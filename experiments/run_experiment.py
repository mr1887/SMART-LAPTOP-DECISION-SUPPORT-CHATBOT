"""
Script thực thi thử nghiệm benchmark so sánh giữa LLM-Only Baseline và Hybrid Pipeline.

Flow:
- Đọc danh sách truy vấn từ file JSON (mặc định: experiments/sample_queries.json)
- Chạy từng query qua:
    1. LLM-Only Baseline (app.baselines.llm_only)
    2. Hybrid Pipeline (app.recommendation.pipeline)
- Đo lường latency, trích xuất selected product, constraints, token usage (nếu có metadata từ API, ngược lại null)
- Ghi log kết quả chi tiết từng query vào file experiments/results/results.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục backend vào sys.path để import các module của hệ thống
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.baselines.llm_only import recommend_llm_only
from app.evaluation.logger import EvaluationLogger, ExperimentRecord
from app.recommendation.pipeline import recommend as recommend_hybrid


def _sanitize_for_json(obj: Any) -> Any:
    """Chuyển đổi các kiểu NumPy / Pandas sang kiểu Python native để serialize JSON an toàn."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


def run_benchmark(
    queries_file: Path,
    output_file: Path,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Thực thi benchmark trên toàn bộ các câu query trong queries_file và lưu vào output_file.
    """
    if not queries_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file queries tại {queries_file}")

    with open(queries_file, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    # Khởi tạo logger lưu vào output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger = EvaluationLogger(output_file)

    all_records: List[Dict[str, Any]] = []
    total_queries = len(queries_data)

    if verbose:
        print(f"=== Bắt đầu chạy benchmark: {total_queries} queries ===")
        print(f"File queries: {queries_file}")
        print(f"Output: {output_file}\n")

    for idx, item in enumerate(queries_data, 1):
        q_id = str(item.get("id", f"Q_{idx:02d}"))
        query_text = str(item.get("query", "")).strip()
        expected_reqs = item.get("expected_requirements", {})

        if verbose:
            print(f"[{idx}/{total_queries}] Query ID: {q_id} | Query: '{query_text}'")

        # ---------------------------------------------------------
        # 1. Chạy LLM-Only Baseline
        # ---------------------------------------------------------
        if verbose:
            print("  -> Chạy [LLM-Only Baseline]...", end="", flush=True)

        t0 = time.perf_counter()
        try:
            llm_res = recommend_llm_only(query=query_text)
            llm_latency_ms = (time.perf_counter() - t0) * 1000.0

            product_name = llm_res.get("product_name")
            claimed_specs = llm_res.get("claimed_specs", {})

            rec_llm = {
                "query_id": q_id,
                "system": "llm_only",
                "parser": "none",
                "raw_query": query_text,
                "parsed_requirements": claimed_specs,
                "expected_requirements": expected_reqs,
                "candidate_count": None,
                "selected_product": product_name,
                "status": "COMPLETED" if product_name else "FAILED",
                "hard_constraint_satisfied": None,  # Cần đối chiếu validation độc lập
                "has_soft_violation": False,
                "soft_violation_count": 0,
                "llm_calls": 1,
                "input_tokens": None,   # Để null nếu API response không cung cấp token metadata
                "output_tokens": None,  # Để null nếu API response không cung cấp token metadata
                "nlu_latency_ms": None,
                "scoring_latency_ms": None,
                "optimization_latency_ms": None,
                "total_latency_ms": round(llm_latency_ms, 2),
                "raw_output": llm_res,
            }
            if verbose:
                print(f" Hoàn tất ({llm_latency_ms:.1f}ms) | Sản phẩm: {product_name}")
        except Exception as e:
            llm_latency_ms = (time.perf_counter() - t0) * 1000.0
            rec_llm = {
                "query_id": q_id,
                "system": "llm_only",
                "parser": "none",
                "raw_query": query_text,
                "parsed_requirements": {},
                "expected_requirements": expected_reqs,
                "candidate_count": None,
                "selected_product": None,
                "status": "ERROR",
                "hard_constraint_satisfied": False,
                "has_soft_violation": False,
                "soft_violation_count": 0,
                "llm_calls": 1,
                "input_tokens": None,
                "output_tokens": None,
                "nlu_latency_ms": None,
                "scoring_latency_ms": None,
                "optimization_latency_ms": None,
                "total_latency_ms": round(llm_latency_ms, 2),
                "error": str(e),
            }
            if verbose:
                print(f" Lỗi: {e}")

        logger.log(rec_llm)
        all_records.append(rec_llm)

        # ---------------------------------------------------------
        # 2. Chạy Hybrid Pipeline (NLU + LightGBM + OR-Tools)
        # ---------------------------------------------------------
        if verbose:
            print("  -> Chạy [Hybrid Pipeline]...", end="", flush=True)

        t0 = time.perf_counter()
        try:
            hybrid_res = recommend_hybrid(query=query_text)
            hybrid_latency_ms = (time.perf_counter() - t0) * 1000.0

            opt = hybrid_res.get("optimization", {})
            reqs = hybrid_res.get("requirements", {})
            cand_count = hybrid_res.get("candidates_count", 0)
            rec_laptop = hybrid_res.get("recommended_laptop")

            if rec_laptop:
                selected_prod_name = rec_laptop.get("laptop_name", f"Laptop #{opt.get('laptop_id')}")
            else:
                selected_prod_name = None

            is_feasible = bool(opt.get("is_feasible", False))
            has_soft_v = bool(opt.get("has_soft_violation", False))
            soft_violations = opt.get("soft_violations", [])

            rec_hybrid = {
                "query_id": q_id,
                "system": "hybrid_pipeline",
                "parser": "gemini",
                "raw_query": query_text,
                "parsed_requirements": reqs,
                "expected_requirements": expected_reqs,
                "candidate_count": cand_count,
                "selected_product": selected_prod_name,
                "status": opt.get("status", "UNKNOWN"),
                "hard_constraint_satisfied": is_feasible,
                "has_soft_violation": has_soft_v,
                "soft_violation_count": len(soft_violations),
                "llm_calls": 1,  # 1 lượt trích xuất NLU bằng Gemini
                "input_tokens": None,   # Để null nếu API response không cung cấp token metadata
                "output_tokens": None,  # Để null nếu API response không cung cấp token metadata
                "nlu_latency_ms": None,
                "scoring_latency_ms": None,
                "optimization_latency_ms": None,
                "total_latency_ms": round(hybrid_latency_ms, 2),
                "optimization_details": _sanitize_for_json(opt),
            }
            if verbose:
                print(f" Hoàn tất ({hybrid_latency_ms:.1f}ms) | Status: {opt.get('status')} | Sản phẩm: {selected_prod_name}")
        except Exception as e:
            hybrid_latency_ms = (time.perf_counter() - t0) * 1000.0
            rec_hybrid = {
                "query_id": q_id,
                "system": "hybrid_pipeline",
                "parser": "gemini",
                "raw_query": query_text,
                "parsed_requirements": {},
                "expected_requirements": expected_reqs,
                "candidate_count": 0,
                "selected_product": None,
                "status": "ERROR",
                "hard_constraint_satisfied": False,
                "has_soft_violation": False,
                "soft_violation_count": 0,
                "llm_calls": 1,
                "input_tokens": None,
                "output_tokens": None,
                "nlu_latency_ms": None,
                "scoring_latency_ms": None,
                "optimization_latency_ms": None,
                "total_latency_ms": round(hybrid_latency_ms, 2),
                "error": str(e),
            }
            if verbose:
                print(f" Lỗi: {e}")

        logger.log(rec_hybrid)
        all_records.append(rec_hybrid)
        if verbose:
            print()

    if verbose:
        print(f"=== Đã hoàn tất benchmark! Kết quả đã ghi vào: {output_file} ===")

    return all_records


def main():
    parser = argparse.ArgumentParser(description="Chạy benchmark so sánh LLM-Only và Hybrid Pipeline.")
    parser.add_argument(
        "--queries",
        type=str,
        default=str(_PROJECT_ROOT / "experiments" / "sample_queries.json"),
        help="Đường dẫn file queries JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "experiments" / "results" / "results.jsonl"),
        help="Đường dẫn file kết quả JSONL",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Không in chi tiết từng bước ra màn hình",
    )

    args = parser.parse_args()
    run_benchmark(
        queries_file=Path(args.queries),
        output_file=Path(args.output),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
