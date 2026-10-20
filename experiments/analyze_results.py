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


def _model_identity_tokens(value: Any) -> set:
    """Extract model-identity tokens conservatively from a product name.

    The goal is to prevent fuzzy matching across different generations/SKUs
    (e.g. M3 Pro -> M5 Pro, Gen 5 -> Gen 7, 16APH8 -> 16AHP9).
    Generic size/year/memory tokens are intentionally ignored.
    """
    text = _normalize_name(value)
    if not text:
        return set()

    tokens = set()

    # Explicit generation markers: "gen 5" -> "gen5".
    for m in re.finditer(r"\bgen\s*(\d{1,2})\b", text):
        tokens.add(f"gen{m.group(1)}")

    # Alphanumeric model/SKU tokens containing both letters and digits.
    for tok in text.split():
        if not (re.search(r"[a-z]", tok) and re.search(r"\d", tok)):
            continue
        # Ignore common capacity/spec tokens that do not define product identity.
        if re.fullmatch(r"\d+(gb|tb|hz|wh|inch)", tok):
            continue
        tokens.add(tok)

    return tokens


def _model_identity_compatible(selected: Any, catalog_name: Any) -> bool:
    """Require model identifiers to agree before accepting fuzzy/substring match.

    If the LLM names a concrete identifier (SKU/generation/model code), the
    catalog entry must contain the same identifier. This deliberately favors
    false negatives over false positives for catalog-validity evaluation.
    """
    selected_tokens = _model_identity_tokens(selected)
    catalog_tokens = _model_identity_tokens(catalog_name)

    # When the LLM provides no concrete model identifier, fuzzy matching is too
    # risky for a scientific catalog-validity metric.
    if not selected_tokens:
        return False

    # Every identifier asserted by the LLM must be represented in the catalog
    # name. This rejects M3->M5, Gen5->Gen7, UX3402ZA->generic family names, etc.
    return selected_tokens.issubset(catalog_tokens)


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
        if len(norm) >= 8
        and (norm in name or name in norm)
        and _model_identity_compatible(selected, row.get("laptop_name"))
    ]
    if len(substring) == 1:
        return substring[0], 0.97

    scored = sorted(
        (
            (difflib.SequenceMatcher(None, norm, name).ratio(), row)
            for name, row in ordered
            if name and _model_identity_compatible(selected, row.get("laptop_name"))
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



def _constraint_ok_claimed_specs(
    specs: Dict[str, Any],
    con: Dict[str, Any],
) -> Optional[bool]:
    """Evaluate only explicit constraints against LLM self-reported specs.

    This is claim-based evidence, not independent factual verification.
    Missing fields are UNVERIFIABLE rather than treated as violations.
    """
    field = con.get("field")
    op = con.get("operator")
    value = con.get("value")

    if field == "price":
        actual = _f(specs.get("price_vnd"))
    elif field == "weight_kg":
        actual = _f(specs.get("weight_kg"))
    elif field == "battery_minutes":
        # The baseline schema does not currently request battery duration.
        actual = _f(specs.get("battery_minutes"))
    elif field == "gpu_discrete":
        gpu = specs.get("gpu")
        if gpu is None:
            return None
        actual = _is_discrete_gpu(gpu)
    elif field == "gpu_keyword":
        gpu = specs.get("gpu")
        if gpu is None:
            return None
        return all(tok in str(gpu).lower() for tok in str(value).lower().split())
    else:
        return None

    if actual is None:
        return None
    if op == "<=":
        return actual <= float(value)
    if op == ">=":
        return actual >= float(value)
    if op == "=":
        return actual == value
    return None


def _explicit_constraint_status(
    constraints: List[Dict[str, Any]],
    checker,
    constraint_type: str,
) -> Tuple[str, int, int]:
    """Return SATISFIED / VIOLATED / UNVERIFIABLE for explicit constraints.

    A known violation is sufficient to label VIOLATED. SATISFIED requires every
    relevant explicit constraint to be assessable and satisfied.
    """
    relevant = [
        c for c in constraints
        if str(c.get("type", "hard")) == constraint_type
    ]
    if not relevant:
        return "NOT_APPLICABLE", 0, 0

    checks = [checker(c) for c in relevant]
    assessable = [x for x in checks if x is not None]
    if any(x is False for x in assessable):
        return "VIOLATED", len(assessable), len(relevant)
    if len(assessable) == len(relevant) and all(x is True for x in assessable):
        return "SATISFIED", len(assessable), len(relevant)
    return "UNVERIFIABLE", len(assessable), len(relevant)


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


def _safe_percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    value = ordered[lower] * (1.0 - frac) + ordered[upper] * frac
    return round(value, 4)


def analyze(
    benchmark_csv: Path,
    results_jsonl: Path,
    evaluated_csv: Path,
    summary_json: Path,
    catalog_csv: Optional[Path] = None,
    external_verification_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    benchmark = _load_csv(benchmark_csv)
    expected_by_id = {r["query_id"]: r for r in benchmark}
    records = _load_jsonl(results_jsonl)

    verification_path = external_verification_csv
    if verification_path is None:
        default_verification = _PROJECT_ROOT / "experiments" / "external_verification_llm_only.csv"
        verification_path = default_verification if default_verification.exists() else None
    verification_by_id: Dict[str, Dict[str, str]] = {}
    if verification_path is not None and verification_path.exists():
        verification_rows = _load_csv(verification_path)
        verification_by_id = {
            str(r.get("query_id")): r for r in verification_rows
            if str(r.get("query_id") or "").strip()
        }

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

        catalog_hard_status, catalog_hard_assessable_n, catalog_hard_total_n = (
            _explicit_constraint_status(
                constraints,
                lambda con: _constraint_ok(matched, con) if matched is not None else None,
                "hard",
            )
        )
        catalog_soft_status, catalog_soft_assessable_n, catalog_soft_total_n = (
            _explicit_constraint_status(
                constraints,
                lambda con: _constraint_ok(matched, con) if matched is not None else None,
                "soft",
            )
        )

        claimed_specs = rec.get("parsed_requirements", {})
        if not isinstance(claimed_specs, dict):
            claimed_specs = {}
        claim_hard_status, claim_hard_assessable_n, claim_hard_total_n = (
            _explicit_constraint_status(
                constraints,
                lambda con: _constraint_ok_claimed_specs(claimed_specs, con),
                "hard",
            )
        )
        claim_soft_status, claim_soft_assessable_n, claim_soft_total_n = (
            _explicit_constraint_status(
                constraints,
                lambda con: _constraint_ok_claimed_specs(claimed_specs, con),
                "soft",
            )
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
            "catalog_explicit_hard_constraint_status": catalog_hard_status,
            "catalog_explicit_hard_assessable_n": catalog_hard_assessable_n,
            "catalog_explicit_hard_total_n": catalog_hard_total_n,
            "catalog_explicit_soft_constraint_status": catalog_soft_status,
            "catalog_explicit_soft_assessable_n": catalog_soft_assessable_n,
            "catalog_explicit_soft_total_n": catalog_soft_total_n,
            "claim_explicit_hard_constraint_status": claim_hard_status,
            "claim_explicit_hard_assessable_n": claim_hard_assessable_n,
            "claim_explicit_hard_total_n": claim_hard_total_n,
            "claim_explicit_soft_constraint_status": claim_soft_status,
            "claim_explicit_soft_assessable_n": claim_soft_assessable_n,
            "claim_explicit_soft_total_n": claim_soft_total_n,
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
            "external_verification_status": (
                verification_by_id.get(qid, {}).get("verification_status")
                if system_name == "llm_only" else None
            ),
            "external_spec_status": (
                verification_by_id.get(qid, {}).get("spec_status")
                if system_name == "llm_only" else None
            ),
            "external_verification_confidence": (
                verification_by_id.get(qid, {}).get("confidence")
                if system_name == "llm_only" else None
            ),
            "external_verification_source_type": (
                verification_by_id.get(qid, {}).get("source_type")
                if system_name == "llm_only" else None
            ),
            "external_verification_source_url": (
                verification_by_id.get(qid, {}).get("source_url")
                if system_name == "llm_only" else None
            ),
            "external_verification_note": (
                verification_by_id.get(qid, {}).get("verification_note")
                if system_name == "llm_only" else None
            ),
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

        # Independent catalog verification is only meaningful when the selected
        # product can actually be mapped to the study catalog. Out-of-catalog
        # recommendations are UNVERIFIABLE, not automatic violations.
        catalog_verified_feasible = [
            r for r in feasible_group
            if str(r.get("recommendation_class")) == "CATALOG_VALID"
        ]
        catalog_verified_hard_rate = _safe_mean([
            float(bool(r.get("independent_hard_constraint_satisfied")))
            for r in catalog_verified_feasible
        ])

        explicit_hard_applicable = [
            r for r in feasible_group
            if int(r.get("catalog_explicit_hard_total_n") or 0) > 0
        ]
        if system == "llm_only":
            explicit_hard_status_field = "claim_explicit_hard_constraint_status"
            explicit_soft_status_field = "claim_explicit_soft_constraint_status"
        else:
            explicit_hard_status_field = "catalog_explicit_hard_constraint_status"
            explicit_soft_status_field = "catalog_explicit_soft_constraint_status"

        explicit_hard_assessable = [
            r for r in explicit_hard_applicable
            if str(r.get(explicit_hard_status_field)) in {"SATISFIED", "VIOLATED"}
        ]
        explicit_hard_satisfied_n = sum(
            1 for r in explicit_hard_assessable
            if str(r.get(explicit_hard_status_field)) == "SATISFIED"
        )
        explicit_hard_rate = (
            round(explicit_hard_satisfied_n / len(explicit_hard_assessable), 4)
            if explicit_hard_assessable else None
        )

        explicit_soft_applicable = [
            r for r in soft_group
            if str(r.get(explicit_soft_status_field)) != "NOT_APPLICABLE"
        ]
        explicit_soft_assessable = [
            r for r in explicit_soft_applicable
            if str(r.get(explicit_soft_status_field)) in {"SATISFIED", "VIOLATED"}
        ]
        explicit_soft_violation_n = sum(
            1 for r in explicit_soft_assessable
            if str(r.get(explicit_soft_status_field)) == "VIOLATED"
        )
        explicit_soft_violation_rate = (
            round(explicit_soft_violation_n / len(explicit_soft_assessable), 4)
            if explicit_soft_assessable else None
        )

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
            "catalog_verified_feasible_n": len(catalog_verified_feasible),
            "catalog_verified_all_hard_requirements_satisfaction_rate": catalog_verified_hard_rate,
            "catalog_verification_note": (
                "This rate is computed only for feasible recommendations that can be "
                "mapped to the study catalog; out-of-catalog outputs are excluded rather "
                "than counted as violations."
            ),
            "explicit_hard_constraint_applicable_n": len(explicit_hard_applicable),
            "explicit_hard_constraint_assessable_n": len(explicit_hard_assessable),
            "explicit_hard_constraint_unverifiable_n": (
                len(explicit_hard_applicable) - len(explicit_hard_assessable)
            ),
            "explicit_hard_constraint_satisfaction_rate_on_assessable_cases": explicit_hard_rate,
            "explicit_hard_constraint_evidence": (
                "llm_self_claimed_specs" if system == "llm_only"
                else "independent_catalog"
            ),
            "explicit_soft_constraint_applicable_n": len(explicit_soft_applicable),
            "explicit_soft_constraint_assessable_n": len(explicit_soft_assessable),
            "explicit_soft_constraint_unverifiable_n": (
                len(explicit_soft_applicable) - len(explicit_soft_assessable)
            ),
            "explicit_soft_constraint_violation_rate_on_assessable_cases": explicit_soft_violation_rate,
            "constraint_metric_note": (
                "LLM-only out-of-catalog constraint compliance is claim-based, not "
                "independently verified. Do not present it as factual accuracy."
                if system == "llm_only"
                else "Hybrid explicit constraints are independently checked against the catalog."
            ),
        }

        if system == "llm_only" and verification_by_id:
            verified_rows = [
                r for r in completed_group
                if str(r.get("external_verification_status") or "").strip()
            ]
            product_rows = [
                r for r in verified_rows
                if str(r.get("external_verification_status")) != "NON_PRODUCT_RESPONSE"
            ]
            verified_real = [
                r for r in product_rows
                if str(r.get("external_verification_status")) == "VERIFIED_REAL"
            ]
            exact_not_confirmed = [
                r for r in product_rows
                if str(r.get("external_verification_status")) == "EXACT_MODEL_NOT_CONFIRMED"
            ]
            verified_hallucinations = [
                r for r in product_rows
                if str(r.get("external_verification_status")) == "VERIFIED_HALLUCINATION"
            ]
            material_spec_errors = [
                r for r in verified_real
                if str(r.get("external_spec_status")) == "MATERIAL_SPEC_ERROR"
            ]
            summary["rq2_decision_reliability"][system]["external_verification"] = {
                "source_file": str(verification_path) if verification_path else None,
                "verified_output_n": len(verified_rows),
                "product_response_n": len(product_rows),
                "non_product_response_n": len(verified_rows) - len(product_rows),
                "verified_real_product_n": len(verified_real),
                "exact_model_not_confirmed_n": len(exact_not_confirmed),
                "verified_product_hallucination_n": len(verified_hallucinations),
                "verified_product_hallucination_rate_on_product_responses": (
                    round(len(verified_hallucinations) / len(product_rows), 4)
                    if product_rows else None
                ),
                "material_spec_error_n": len(material_spec_errors),
                "material_spec_error_rate_on_verified_real_products": (
                    round(len(material_spec_errors) / len(verified_real), 4)
                    if verified_real else None
                ),
                "verification_status_counts": {
                    status: sum(
                        1 for r in verified_rows
                        if str(r.get("external_verification_status")) == status
                    )
                    for status in sorted({
                        str(r.get("external_verification_status"))
                        for r in verified_rows
                    })
                },
                "spec_status_counts": {
                    status: sum(
                        1 for r in verified_rows
                        if str(r.get("external_spec_status")) == status
                    )
                    for status in sorted({
                        str(r.get("external_spec_status"))
                        for r in verified_rows
                    })
                },
                "note": (
                    "EXACT_MODEL_NOT_CONFIRMED is unresolved, not hallucination. "
                    "Material spec errors are counted only among externally VERIFIED_REAL products."
                ),
            }
        api_attempt_values = [
            float(
                r.get("api_attempts")
                if system == "llm_only"
                else r.get("nlu_api_attempts")
                or 0
            )
            for r in attempted_group
        ]
        input_token_values = [
            float(r["input_tokens"])
            for r in attempted_group if r.get("input_tokens") is not None
        ]
        output_token_values = [
            float(r["output_tokens"])
            for r in attempted_group if r.get("output_tokens") is not None
        ]
        thinking_token_values = [
            float(r["thinking_tokens"])
            for r in attempted_group if r.get("thinking_tokens") is not None
        ]
        total_token_values = [
            float(r["total_tokens"])
            for r in attempted_group if r.get("total_tokens") is not None
        ]
        cost_values = [
            float(r["estimated_api_cost_usd"])
            for r in attempted_group if r.get("estimated_api_cost_usd") is not None
        ]
        total_api_cost = sum(cost_values) if cost_values else None
        cost_per_completed = (
            total_api_cost / len(completed_group)
            if total_api_cost is not None and completed_group else None
        )
        cost_per_catalog_valid = (
            total_api_cost / catalog_valid_n
            if total_api_cost is not None and catalog_valid_n > 0 else None
        )

        summary["rq3_operational_feasibility"][system] = {
            "n": len(attempted_group),
            "completed_n": len(completed_group),
            "average_latency_ms_all_attempts": _safe_mean(lat),
            "median_latency_ms_all_attempts": _safe_median(lat),
            "p95_latency_ms_all_attempts": _safe_percentile(lat, 0.95),
            "average_latency_ms_completed": _safe_mean(completed_lat),
            "median_latency_ms_completed": _safe_median(completed_lat),
            "p95_latency_ms_completed": _safe_percentile(completed_lat, 0.95),
            "average_api_attempts": _safe_mean(api_attempt_values),
            "median_api_attempts": _safe_median(api_attempt_values),
            "api_attempt_metric_note": (
                "LLM-only uses api_attempts; Hybrid uses nlu_api_attempts so retries "
                "and model fallbacks are measured consistently."
            ),
            "average_input_tokens": _safe_mean(input_token_values),
            "average_output_tokens": _safe_mean(output_token_values),
            "average_thinking_tokens": _safe_mean(thinking_token_values),
            "average_total_tokens": _safe_mean(total_token_values),
            "average_estimated_api_cost_usd": _safe_mean(cost_values),
            "total_estimated_api_cost_usd": (
                round(total_api_cost, 8) if total_api_cost is not None else None
            ),
            "cost_per_completed_recommendation_usd": (
                round(cost_per_completed, 8) if cost_per_completed is not None else None
            ),
            "cost_per_catalog_valid_recommendation_usd": (
                round(cost_per_catalog_valid, 8) if cost_per_catalog_valid is not None else None
            ),
            "cost_metric_note": (
                "Estimated from Gemini usage_metadata using the pricing snapshot "
                "documented in app.evaluation.cost. Output cost includes thinking tokens."
            ),
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
        "--external-verification",
        default=None,
        help="Optional CSV with external verification labels for LLM-only outputs.",
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
        external_verification_csv=(
            Path(args.external_verification) if args.external_verification else None
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
