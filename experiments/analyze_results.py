"""
Independent benchmark analysis for the research experiment.

Reads:
- experiments/test_queries.csv
- experiments/results/results.jsonl
- processed laptop catalog

Writes:
- experiments/results/evaluated_results.csv
- experiments/results/summary_metrics.json

The evaluator does not trust the solver's own compliance flag. It maps the
recommended product back to the catalog and re-checks ground-truth hard/soft
requirements independently.
"""

import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _j(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _f(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_name(value: Any) -> str:
    s = str(value or "").lower()
    s = re.sub(r"[^a-z0-9à-ỹ]+", " ", s, flags=re.IGNORECASE)
    return " ".join(s.split())


def _runtime_tags(row: Dict[str, Any]) -> Dict[str, bool]:
    gaming = _b(row.get("is_gaming_laptop"))
    cpu_multi = _f(row.get("geekbench_cpu_multi")) or 0.0
    weight = _f(row.get("laptop_weight"))
    battery = _f(row.get("office_battery_minutes_final"))
    workstation = _b(row.get("is_workstation"))
    return {
        "is_gaming_friendly": gaming,
        "is_office_friendly": (
            (not gaming)
            and weight is not None and weight <= 2.0
            and battery is not None and battery >= 360
        ),
        "is_programming_friendly": cpu_multi >= 6000,
        "is_graphic_friendly": cpu_multi >= 8000 or workstation,
    }


def _is_discrete_gpu(name: Any) -> bool:
    s = str(name or "").lower()
    if any(k in s for k in [
        "tích hợp", "integrated", "adreno", "uhd", "iris",
        "intel graphics", "apple m"
    ]):
        return False
    return any(k in s for k in [
        "rtx", "gtx", "geforce", "nvidia", "radeon rx", "arc b", "discrete"
    ])


def _find_catalog() -> Path:
    candidates = [
        _PROJECT_ROOT / "data" / "processed" / "laptop_dataset_scored.csv",
        _PROJECT_ROOT / "data" / "processed" / "laptop_dataset_tagged.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Không tìm thấy processed laptop catalog.")


def _catalog_index(catalog: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, Dict[str, Any]]]]:
    exact: Dict[str, Dict[str, Any]] = {}
    ordered: List[Tuple[str, Dict[str, Any]]] = []
    for row in catalog:
        row["_eval_tags"] = _runtime_tags(row)
        norm = _normalize_name(row.get("laptop_name"))
        if norm and norm not in exact:
            exact[norm] = row
        ordered.append((norm, row))
    return exact, ordered


def _match_product(
    selected: Any,
    exact: Dict[str, Dict[str, Any]],
    ordered: List[Tuple[str, Dict[str, Any]]],
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    norm = _normalize_name(selected)
    if not norm:
        return None, None
    if norm in exact:
        return exact[norm], 1.0

    # Conservative substring match for model suffix/prefix differences.
    substring = [
        row for name, row in ordered
        if len(norm) >= 8 and (norm in name or name in norm)
    ]
    if len(substring) == 1:
        return substring[0], 0.97

    scored = sorted(
        (
            (difflib.SequenceMatcher(None, norm, name).ratio(), row)
            for name, row in ordered
            if name
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    if not scored:
        return None, None
    best_score, best_row = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0

    # Require high similarity and separation to avoid falsely validating hallucinations.
    if best_score >= 0.90 and best_score - second >= 0.03:
        return best_row, best_score
    return None, best_score


def _constraint_ok(row: Dict[str, Any], con: Dict[str, Any]) -> Optional[bool]:
    field = con.get("field")
    op = con.get("operator")
    value = con.get("value")

    if field == "price":
        actual = _f(row.get("price"))
    elif field == "weight_kg":
        actual = _f(row.get("laptop_weight"))
    elif field == "battery_minutes":
        actual = _f(row.get("office_battery_minutes_final"))
    elif field == "gpu_discrete":
        actual = _is_discrete_gpu(row.get("gpu_name"))
    elif field == "gpu_keyword":
        gpu = str(row.get("gpu_name") or "").lower()
        return all(tok in gpu for tok in str(value).lower().split())
    else:
        return None

    if actual is None:
        return False
    if op == "<=":
        return actual <= float(value)
    if op == ">=":
        return actual >= float(value)
    if op == "=":
        return actual == value
    return None


def _validate_product(
    row: Optional[Dict[str, Any]],
    tags: List[str],
    constraints: List[Dict[str, Any]],
) -> Tuple[bool, Optional[bool], Optional[bool]]:
    if row is None:
        return False, False, None

    product_valid = True
    hard_checks: List[bool] = []
    soft_checks: List[bool] = []

    for tag in tags:
        hard_checks.append(bool(row["_eval_tags"].get(tag, False)))

    for con in constraints:
        ok = _constraint_ok(row, con)
        if ok is None:
            continue
        if con.get("type", "hard") == "hard":
            hard_checks.append(ok)
        else:
            soft_checks.append(ok)

    hard_satisfied = all(hard_checks) if hard_checks else True
    soft_violation = (not all(soft_checks)) if soft_checks else None
    return product_valid, hard_satisfied, soft_violation


def _norm_constraint_value(value: Any) -> str:
    """Normalize semantically equivalent values before requirement comparison.

    JSON distinguishes 15000000 from 15000000.0 textually even though they are
    the same numeric constraint. Evaluation must not penalize that formatting.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return format(round(numeric, 6), ".6f").rstrip("0").rstrip(".")
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _norm_constraints(items: Iterable[Dict[str, Any]]) -> set:
    out = set()
    for con in items or []:
        out.add((
            str(con.get("field")),
            str(con.get("operator")),
            _norm_constraint_value(con.get("value")),
            str(con.get("type", "hard")),
        ))
    return out


def _norm_preferences(items: Iterable[Dict[str, Any]]) -> set:
    return {
        (str(p.get("field")), str(p.get("direction")))
        for p in items or []
    }


def _prf(predicted: set, expected: set) -> Tuple[float, float, float]:
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1


def _requirement_eval(
    parsed: Dict[str, Any],
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    parsed_tags = set(parsed.get("required_tags", []) or [])
    expected_tags = set(expected.get("required_tags", []) or [])
    parsed_constraints = _norm_constraints(parsed.get("constraints", []) or [])
    expected_constraints = _norm_constraints(expected.get("constraints", []) or [])
    parsed_prefs = _norm_preferences(parsed.get("preferences", []) or [])
    expected_prefs = _norm_preferences(expected.get("preferences", []) or [])

    tag_p, tag_r, tag_f1 = _prf(parsed_tags, expected_tags)
    con_p, con_r, con_f1 = _prf(parsed_constraints, expected_constraints)

    tag_exact = parsed_tags == expected_tags
    constraints_exact = parsed_constraints == expected_constraints
    prefs_exact = parsed_prefs == expected_prefs

    return {
        "tag_match_exact": tag_exact,
        "constraint_extraction_exact": constraints_exact,
        "requirement_extraction_exact": tag_exact and constraints_exact and prefs_exact,
        "tag_precision": tag_p,
        "tag_recall": tag_r,
        "tag_f1": tag_f1,
        "constraint_precision": con_p,
        "constraint_recall": con_r,
        "constraint_f1": con_f1,
    }


def _safe_mean(values: List[float]) -> Optional[float]:
    return round(mean(values), 4) if values else None


def _safe_median(values: List[float]) -> Optional[float]:
    return round(median(values), 4) if values else None


def analyze(
    benchmark_csv: Path,
    results_jsonl: Path,
    evaluated_csv: Path,
    summary_json: Path,
    catalog_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    benchmark = _load_csv(benchmark_csv)
    expected_by_id = {r["query_id"]: r for r in benchmark}
    records = _load_jsonl(results_jsonl)

    catalog_path = catalog_csv or _find_catalog()
    catalog = _load_csv(catalog_path)
    exact, ordered = _catalog_index(catalog)

    evaluated: List[Dict[str, Any]] = []

    for rec in records:
        qid = str(rec.get("query_id"))
        gt = expected_by_id.get(qid, {})
        tags = _j(gt.get("expected_tags"), [])
        constraints = _j(gt.get("expected_constraints_json"), [])
        prefs = _j(gt.get("expected_preferences_json"), [])
        expected_req = {
            "required_tags": tags,
            "constraints": constraints,
            "preferences": prefs,
        }

        matched, match_score = _match_product(
            rec.get("selected_product"), exact, ordered
        )
        product_valid, hard_ok, soft_violation = _validate_product(
            matched, tags, constraints
        )

        parsed = rec.get("parsed_requirements", {})
        if not isinstance(parsed, dict):
            parsed = {}

        if rec.get("system") == "hybrid_pipeline":
            req_eval = _requirement_eval(parsed, expected_req)
            tag_exact = req_eval["tag_match_exact"]
            constraints_exact = req_eval["constraint_extraction_exact"]
            requirement_exact = req_eval["requirement_extraction_exact"]
            context_correct = (
                str(rec.get("context_action"))
                == str(gt.get("expected_context_action", "REPLACE"))
            )
            expected_feasible = _b(gt.get("expected_feasible"))
            predicted_feasible = bool(rec.get("hard_constraint_satisfied", False))
            feasibility_correct = expected_feasible == predicted_feasible
            expected_candidates = (
                int(gt["expected_candidate_count"])
                if str(gt.get("expected_candidate_count", "")).strip()
                else None
            )
            candidate_count_correct = (
                expected_candidates is None
                or int(rec.get("candidate_count") or 0) == expected_candidates
            )
        else:
            req_eval = {}
            tag_exact = None
            constraints_exact = None
            requirement_exact = None
            context_correct = None
            feasibility_correct = None
            candidate_count_correct = None

        system_name = str(rec.get("system"))
        status_name = str(rec.get("status"))
        if system_name == "llm_only":
            if status_name == "API_FAILED":
                recommendation_class = "API_FAILED"
            elif status_name == "PARSE_FAILED":
                recommendation_class = "PARSE_FAILED"
            elif status_name == "COMPLETED" and matched is not None:
                recommendation_class = "CATALOG_VALID"
            elif status_name == "COMPLETED":
                recommendation_class = "OUT_OF_CATALOG"
            else:
                recommendation_class = status_name or "UNKNOWN"
        else:
            recommendation_class = (
                "CATALOG_VALID" if matched is not None else "OUT_OF_CATALOG"
            )

        evaluated.append({
            **rec,
            "matched_catalog_product": (
                matched.get("laptop_name") if matched else None
            ),
            "catalog_match_score": match_score,
            "recommendation_class": recommendation_class,
            "product_valid": product_valid,
            "independent_hard_constraint_satisfied": hard_ok,
            "independent_soft_constraint_violation": soft_violation,
            "tag_match_exact": tag_exact,
            "constraint_extraction_exact": constraints_exact,
            "requirement_extraction_exact": requirement_exact,
            "tag_precision": req_eval.get("tag_precision"),
            "tag_recall": req_eval.get("tag_recall"),
            "tag_f1": req_eval.get("tag_f1"),
            "constraint_precision": req_eval.get("constraint_precision"),
            "constraint_recall": req_eval.get("constraint_recall"),
            "constraint_f1": req_eval.get("constraint_f1"),
            "context_action_correct": context_correct,
            "feasibility_detection_correct": feasibility_correct,
            "candidate_count_correct": candidate_count_correct,
        })

    evaluated_csv.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in evaluated:
        for key in row.keys():
            if key not in fields:
                fields.append(key)

    with open(evaluated_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in evaluated:
            flat = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    flat[key] = json.dumps(value, ensure_ascii=False)
                else:
                    flat[key] = value
            writer.writerow(flat)

    summary: Dict[str, Any] = {
        "catalog": str(catalog_path),
        "benchmark_queries": len(benchmark),
        "records": len(evaluated),
        "rq1_technical_feasibility": {},
        "rq2_decision_reliability": {},
        "rq3_operational_feasibility": {},
    }

    hybrid = [r for r in evaluated if r.get("system") == "hybrid_pipeline"]
    if hybrid:
        summary["rq1_technical_feasibility"] = {
            "n": len(hybrid),
            "end_to_end_success_rate": _safe_mean([
                1.0 if r.get("status") != "ERROR" else 0.0 for r in hybrid
            ]),
            "tag_extraction_exact_rate": _safe_mean([
                float(bool(r.get("tag_match_exact"))) for r in hybrid
            ]),
            "constraint_extraction_exact_rate": _safe_mean([
                float(bool(r.get("constraint_extraction_exact"))) for r in hybrid
            ]),
            "full_requirement_exact_rate": _safe_mean([
                float(bool(r.get("requirement_extraction_exact"))) for r in hybrid
            ]),
            "mean_tag_f1": _safe_mean([
                float(r.get("tag_f1") or 0.0) for r in hybrid
            ]),
            "mean_constraint_precision": _safe_mean([
                float(r.get("constraint_precision") or 0.0) for r in hybrid
            ]),
            "mean_constraint_recall": _safe_mean([
                float(r.get("constraint_recall") or 0.0) for r in hybrid
            ]),
            "mean_constraint_f1": _safe_mean([
                float(r.get("constraint_f1") or 0.0) for r in hybrid
            ]),
            "feasibility_detection_accuracy": _safe_mean([
                float(bool(r.get("feasibility_detection_correct"))) for r in hybrid
            ]),
            "candidate_count_exact_rate": _safe_mean([
                float(bool(r.get("candidate_count_correct"))) for r in hybrid
            ]),
        }

        mt = [
            r for r in hybrid
            if str(r.get("evaluation_scope")) == "multiturn"
        ]
        summary["rq1_technical_feasibility"]["multiturn_context_action_accuracy"] = (
            _safe_mean([
                float(bool(r.get("context_action_correct"))) for r in mt
            ])
        )
        summary["rq1_technical_feasibility"]["multiturn_n"] = len(mt)
        nlu_counts: Dict[str, int] = {}
        for row in hybrid:
            source = str(row.get("nlu_source") or "unknown")
            nlu_counts[source] = nlu_counts.get(source, 0) + 1
        summary["rq1_technical_feasibility"]["nlu_source_counts"] = nlu_counts

    main = [
        r for r in evaluated
        if _b(r.get("compare_with_baseline"))
    ]
    by_system: Dict[str, List[Dict[str, Any]]] = {}
    for r in main:
        by_system.setdefault(str(r.get("system")), []).append(r)

    for system, group in by_system.items():
        attempted_group = list(group)
        if system == "llm_only":
            completed_group = [r for r in group if str(r.get("status")) == "COMPLETED"]
            api_failed_group = [r for r in group if str(r.get("status")) == "API_FAILED"]
            parse_failed_group = [r for r in group if str(r.get("status")) == "PARSE_FAILED"]
        else:
            completed_group = list(group)
            api_failed_group = []
            parse_failed_group = []

        feasible_group = [
            r for r in completed_group if _b(r.get("expected_feasible"))
        ]
        soft_group = []
        for r in completed_group:
            gt = expected_by_id.get(str(r.get("query_id")), {})
            cons = _j(gt.get("expected_constraints_json"), [])
            if any(c.get("type", "hard") == "soft" for c in cons):
                soft_group.append(r)

        lat = [
            float(r["total_latency_ms"])
            for r in attempted_group
            if r.get("total_latency_ms") is not None
        ]
        completed_lat = [
            float(r["total_latency_ms"])
            for r in completed_group
            if r.get("total_latency_ms") is not None
        ]
        valid_rate = _safe_mean([
            float(bool(r.get("product_valid"))) for r in completed_group
        ])
        hard_rate = _safe_mean([
            float(bool(r.get("independent_hard_constraint_satisfied")))
            for r in feasible_group
        ])
        soft_violation_rate = _safe_mean([
            float(bool(r.get("independent_soft_constraint_violation")))
            for r in soft_group
        ])

        catalog_valid_n = sum(
            1 for r in completed_group
            if str(r.get("recommendation_class")) == "CATALOG_VALID"
        )
        out_of_catalog_n = sum(
            1 for r in completed_group
            if str(r.get("recommendation_class")) == "OUT_OF_CATALOG"
        )
        catalog_validity_rate = (
            round(catalog_valid_n / len(completed_group), 4)
            if completed_group else None
        )
        out_of_catalog_rate = (
            round(out_of_catalog_n / len(completed_group), 4)
            if completed_group else None
        )

        summary["rq2_decision_reliability"][system] = {
            "comparison_n": len(attempted_group),
            "completed_recommendation_n": len(completed_group),
            "api_failure_n": len(api_failed_group),
            "api_failure_rate": (round(len(api_failed_group) / len(attempted_group), 4) if attempted_group else None),
            "parse_failure_n": len(parse_failed_group),
            "feasible_case_n": len(feasible_group),
            "catalog_valid_n": catalog_valid_n,
            "catalog_validity_rate": catalog_validity_rate,
            "out_of_catalog_n": out_of_catalog_n,
            "out_of_catalog_rate": out_of_catalog_rate,
            "product_validity_rate": catalog_validity_rate,
            "verified_hallucination_n": None,
            "verified_hallucination_rate": None,
            "hallucination_metric_note": (
                "OUT_OF_CATALOG is not automatically treated as hallucination. "
                "True hallucination requires independent external verification."
            ),
            "hard_constraint_satisfaction_rate_on_feasible_cases": hard_rate,
            "hard_constraint_violation_rate_on_feasible_cases": (
                round(1.0 - hard_rate, 4) if hard_rate is not None else None
            ),
            "soft_constraint_violation_rate": soft_violation_rate,
        }
        summary["rq3_operational_feasibility"][system] = {
            "n": len(attempted_group),
            "completed_n": len(completed_group),
            "average_latency_ms_all_attempts": _safe_mean(lat),
            "median_latency_ms_all_attempts": _safe_median(lat),
            "average_latency_ms_completed": _safe_mean(completed_lat),
            "median_latency_ms_completed": _safe_median(completed_lat),
            "average_llm_calls": _safe_mean([
                float(r.get("llm_calls") or 0) for r in attempted_group
            ]),
            "average_input_tokens": _safe_mean([
                float(r["input_tokens"])
                for r in attempted_group if r.get("input_tokens") is not None
            ]),
            "average_output_tokens": _safe_mean([
                float(r["output_tokens"])
                for r in attempted_group if r.get("output_tokens") is not None
            ]),
        }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze research benchmark results.")
    parser.add_argument(
        "--queries",
        default=str(_PROJECT_ROOT / "experiments" / "test_queries.csv"),
    )
    parser.add_argument(
        "--results",
        default=str(_PROJECT_ROOT / "experiments" / "results" / "results.jsonl"),
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Optional processed catalog CSV path.",
    )
    parser.add_argument(
        "--evaluated",
        default=str(_PROJECT_ROOT / "experiments" / "results" / "evaluated_results.csv"),
    )
    parser.add_argument(
        "--summary",
        default=str(_PROJECT_ROOT / "experiments" / "results" / "summary_metrics.json"),
    )
    args = parser.parse_args()

    summary = analyze(
        benchmark_csv=Path(args.queries),
        results_jsonl=Path(args.results),
        evaluated_csv=Path(args.evaluated),
        summary_json=Path(args.summary),
        catalog_csv=Path(args.catalog) if args.catalog else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
