"""
Experiment runner for the 60-query research benchmark.

Supports CSV or JSON input, independent single-turn cases, multi-turn
conversations, pilot/full runs, and LLM-only vs Hybrid execution.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.baselines.llm_only import recommend_llm_only
from app.evaluation.logger import EvaluationLogger
from app.recommendation.pipeline import recommend as recommend_hybrid


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list, bool, int, float)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _bool_field(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy benchmark file: {path}")

    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            raw = list(csv.DictReader(f))

        items: List[Dict[str, Any]] = []
        for idx, row in enumerate(raw, 1):
            expected = {
                "required_tags": _json_field(row.get("expected_tags"), []),
                "constraints": _json_field(row.get("expected_constraints_json"), []),
                "preferences": _json_field(row.get("expected_preferences_json"), []),
            }
            items.append({
                "id": row.get("query_id") or f"Q_{idx:02d}",
                "category": row.get("category", ""),
                "conversation_id": row.get("conversation_id", ""),
                "turn_index": int(row.get("turn_index") or 1),
                "query": row.get("query", ""),
                "expected_context_action": row.get("expected_context_action", "REPLACE"),
                "expected_requirements": expected,
                "expected_feasible": _bool_field(row.get("expected_feasible"), True),
                "expected_candidate_count": (
                    int(row["expected_candidate_count"])
                    if str(row.get("expected_candidate_count", "")).strip()
                    else None
                ),
                "evaluation_scope": row.get("evaluation_scope", "main"),
                "compare_with_baseline": _bool_field(row.get("compare_with_baseline"), True),
                "notes": row.get("notes", ""),
            })
        return items

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Benchmark JSON phải là một list.")
    return raw


def run_benchmark(
    queries_file: Path,
    output_file: Path,
    verbose: bool = True,
    systems: str = "both",
    append: bool = False,
) -> List[Dict[str, Any]]:
    queries_data = _load_queries(queries_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger = EvaluationLogger(output_file)
    if not append:
        logger.clear()

    run_llm = systems in {"both", "llm_only"}
    run_hybrid = systems in {"both", "hybrid"}

    all_records: List[Dict[str, Any]] = []
    hybrid_context_by_conversation: Dict[str, Dict[str, Any]] = {}
    baseline_history_by_conversation: Dict[str, List[str]] = {}

    if verbose:
        print(f"=== Benchmark: {len(queries_data)} query-turns ===")
        print(f"Input : {queries_file}")
        print(f"Output: {output_file}")
        print(f"Systems: {systems}\n")

    for idx, item in enumerate(queries_data, 1):
        q_id = str(item.get("id", f"Q_{idx:02d}"))
        query_text = str(item.get("query", "")).strip()
        category = str(item.get("category", ""))
        conversation_id = str(item.get("conversation_id", "") or "")
        turn_index = int(item.get("turn_index", 1) or 1)
        expected_reqs = item.get("expected_requirements", {})
        expected_feasible = bool(item.get("expected_feasible", True))
        expected_action = str(item.get("expected_context_action", "REPLACE"))
        expected_candidate_count = item.get("expected_candidate_count")
        scope = str(item.get("evaluation_scope", "main"))
        compare_with_baseline = bool(item.get("compare_with_baseline", True))

        if verbose:
            prefix = f"[{idx}/{len(queries_data)}] {q_id}"
            if conversation_id:
                prefix += f" ({conversation_id}, turn {turn_index})"
            print(f"{prefix}: {query_text}")

        common = {
            "query_id": q_id,
            "category": category,
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "raw_query": query_text,
            "expected_requirements": expected_reqs,
            "expected_feasible": expected_feasible,
            "expected_context_action": expected_action,
            "expected_candidate_count": expected_candidate_count,
            "evaluation_scope": scope,
            "compare_with_baseline": compare_with_baseline,
        }

        # LLM-only baseline.
        if run_llm and compare_with_baseline:
            llm_query = query_text
            if conversation_id:
                history = baseline_history_by_conversation.setdefault(conversation_id, [])
                history.append(query_text)
                llm_query = "\n".join(
                    f"Lượt {i + 1}: {msg}" for i, msg in enumerate(history)
                )

            if verbose:
                print("  -> LLM-only...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                llm_res = recommend_llm_only(query=llm_query)
                latency = (time.perf_counter() - t0) * 1000.0
                product_name = llm_res.get("product_name")
                record = {
                    **common,
                    "system": "llm_only",
                    "parser": "none",
                    "parsed_requirements": llm_res.get("claimed_specs", {}),
                    "candidate_count": None,
                    "selected_product": product_name,
                    "status": "COMPLETED" if product_name else "FAILED",
                    "hard_constraint_satisfied": None,
                    "has_soft_violation": None,
                    "soft_violation_count": None,
                    "llm_calls": 1,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_latency_ms": round(latency, 2),
                    "raw_output": _sanitize_for_json(llm_res),
                }
                if verbose:
                    print(f" {latency:.1f}ms | {product_name}")
            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000.0
                record = {
                    **common,
                    "system": "llm_only",
                    "parser": "none",
                    "parsed_requirements": {},
                    "candidate_count": None,
                    "selected_product": None,
                    "status": "ERROR",
                    "hard_constraint_satisfied": False,
                    "has_soft_violation": None,
                    "soft_violation_count": None,
                    "llm_calls": 1,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_latency_ms": round(latency, 2),
                    "error": str(exc),
                }
                if verbose:
                    print(f" ERROR: {exc}")
            logger.log(record)
            all_records.append(record)

        # Hybrid pipeline.
        if run_hybrid:
            current_constraints: Optional[Dict[str, Any]] = None
            if conversation_id:
                current_constraints = hybrid_context_by_conversation.get(conversation_id)

            if verbose:
                print("  -> Hybrid...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                hybrid_res = recommend_hybrid(
                    query=query_text,
                    current_constraints=current_constraints,
                )
                latency = (time.perf_counter() - t0) * 1000.0
                opt = hybrid_res.get("optimization", {})
                reqs = hybrid_res.get("requirements", {})
                rec_laptop = hybrid_res.get("recommended_laptop")
                selected_name = rec_laptop.get("laptop_name") if rec_laptop else None

                if conversation_id:
                    hybrid_context_by_conversation[conversation_id] = reqs

                record = {
                    **common,
                    "system": "hybrid_pipeline",
                    "parser": "gemini_with_regex_fallback",
                    "parsed_requirements": _sanitize_for_json(reqs),
                    "context_action": hybrid_res.get("context_action"),
                    "candidate_count": hybrid_res.get("candidates_count", 0),
                    "selected_product": selected_name,
                    "status": opt.get("status", "UNKNOWN"),
                    "hard_constraint_satisfied": bool(opt.get("is_feasible", False)),
                    "has_soft_violation": bool(opt.get("has_soft_violation", False)),
                    "soft_violation_count": len(opt.get("soft_violations", []) or []),
                    "llm_calls": 1,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_latency_ms": round(latency, 2),
                    "optimization_details": _sanitize_for_json(opt),
                }
                if verbose:
                    print(
                        f" {latency:.1f}ms | action={record['context_action']} | "
                        f"status={record['status']} | {selected_name}"
                    )
            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000.0
                record = {
                    **common,
                    "system": "hybrid_pipeline",
                    "parser": "gemini_with_regex_fallback",
                    "parsed_requirements": {},
                    "context_action": None,
                    "candidate_count": 0,
                    "selected_product": None,
                    "status": "ERROR",
                    "hard_constraint_satisfied": False,
                    "has_soft_violation": None,
                    "soft_violation_count": None,
                    "llm_calls": 1,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_latency_ms": round(latency, 2),
                    "error": str(exc),
                }
                if verbose:
                    print(f" ERROR: {exc}")
            logger.log(record)
            all_records.append(record)

        if verbose:
            print()

    if verbose:
        print(f"=== Done. Records written to {output_file} ===")
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LLM-only vs Hybrid laptop benchmark."
    )
    parser.add_argument(
        "--queries",
        default=str(_PROJECT_ROOT / "experiments" / "test_queries.csv"),
        help="CSV or JSON benchmark file.",
    )
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "experiments" / "results" / "results.jsonl"),
        help="JSONL output path.",
    )
    parser.add_argument(
        "--systems",
        choices=["both", "hybrid", "llm_only"],
        default="both",
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        queries_file=Path(args.queries),
        output_file=Path(args.output),
        verbose=not args.quiet,
        systems=args.systems,
        append=args.append,
    )


if __name__ == "__main__":
    main()
