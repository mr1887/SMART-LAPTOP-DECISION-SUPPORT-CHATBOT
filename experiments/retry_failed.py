"""
Retry only failed API calls from a previous benchmark run.

Example (PowerShell, from repository root):
python experiments/retry_failed.py --queries experiments/test_queries_300_final.csv --previous-results experiments/results/results_300.jsonl --output experiments/results/results_300_retry.jsonl --delay 4
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Support imports when launched as: python experiments/retry_failed.py
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EXPERIMENTS_DIR.parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from run_experiment import _load_queries, run_benchmark


BASELINE_SYSTEMS = ("llm_only", "catalog_grounded_llm")
RETRYABLE_STATUSES = {"API_FAILED"}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file kết quả gốc: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL lỗi tại dòng {line_number} của {path}: {exc}"
                ) from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def _query_id(query: Dict[str, Any]) -> str:
    return str(query.get("id", query.get("query_id", ""))).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chỉ chạy lại các cặp query_id + system có status API_FAILED."
    )
    parser.add_argument(
        "--queries",
        default=str(_PROJECT_ROOT / "experiments" / "test_queries_300_final.csv"),
        help="CSV benchmark gốc; cần chứa toàn bộ các query_id trong kết quả cũ.",
    )
    parser.add_argument(
        "--previous-results",
        default=str(_PROJECT_ROOT / "experiments" / "results" / "results_300.jsonl"),
        help="JSONL kết quả benchmark ban đầu; file này chỉ được đọc, không bị sửa.",
    )
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "experiments" / "results" / "results_300_retry.jsonl"),
        help="File JSONL mới để ghi các kết quả retry.",
    )
    parser.add_argument(
        "--merged-output",
        default=str(_PROJECT_ROOT / "experiments" / "results" / "results_300_merged.jsonl"),
        help="JSONL mới chứa kết quả gốc cộng kết quả retry thay thế theo query_id + system.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Số giây chờ giữa các truy vấn baseline để giảm tốc độ gọi API (mặc định: 4).",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Bao gồm cả status ERROR ngoài API_FAILED. Mặc định chỉ retry API_FAILED.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Cho phép ghi đè file output nếu file đó đã tồn tại.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ liệt kê số lượng và query_id cần retry, không gọi API.",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries).resolve()
    previous_path = Path(args.previous_results).resolve()
    output_path = Path(args.output).resolve()
    merged_path = Path(args.merged_output).resolve()

    protected_paths = {queries_path, previous_path}
    if output_path in protected_paths or merged_path in protected_paths:
        raise ValueError("Không được ghi đè benchmark CSV hoặc kết quả gốc.")
    if output_path == merged_path:
        raise ValueError("--output và --merged-output phải là hai file khác nhau.")
    if args.delay < 0:
        raise ValueError("--delay phải >= 0.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"File output đã tồn tại: {output_path}\n"
            "Hãy chọn tên file khác hoặc thêm --overwrite nếu thực sự muốn ghi đè."
        )

    benchmark = _load_queries(queries_path)
    query_by_id: Dict[str, Dict[str, Any]] = {}
    for query in benchmark:
        qid = _query_id(query)
        if qid:
            query_by_id[qid] = query

    original_records = _read_jsonl(previous_path)
    status_targets = set(RETRYABLE_STATUSES)
    if args.include_errors:
        status_targets.add("ERROR")

    # Keep the latest original record for each query/system pair.
    original_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    failed_ids_by_system: Dict[str, set] = {system: set() for system in BASELINE_SYSTEMS}
    for record in original_records:
        system = str(record.get("system", "")).strip()
        qid = str(record.get("query_id", record.get("id", ""))).strip()
        status = str(record.get("status", "")).strip().upper()
        if not qid or system not in BASELINE_SYSTEMS:
            continue
        pair = (qid, system)
        original_by_pair[pair] = record
        if status in status_targets:
            failed_ids_by_system[system].add(qid)

    selected: Dict[str, List[Dict[str, Any]]] = {}
    missing: Dict[str, List[str]] = {}
    for system in BASELINE_SYSTEMS:
        ids = failed_ids_by_system[system]
        selected[system] = [
            query for query in benchmark
            if _query_id(query) in ids
            and bool(query.get("compare_with_baseline", True))
        ]
        selected_ids = {_query_id(query) for query in selected[system]}
        missing[system] = sorted(ids - selected_ids)

    unresolved = {system: ids for system, ids in missing.items() if ids}
    if unresolved:
        details = "; ".join(
            f"{system}: {', '.join(ids[:12])}" for system, ids in unresolved.items()
        )
        raise ValueError(
            "Một số query lỗi trong kết quả cũ không tìm thấy trong benchmark "
            "hoặc không được đánh dấu compare_with_baseline=1. "
            f"Kiểm tra đúng file CSV và đúng phiên bản benchmark. Chi tiết: {details}"
        )

    # A later conversation turn cannot be faithfully replayed alone because the
    # baseline runner reconstructs conversation history from the supplied subset.
    for system, rows in selected.items():
        later_turns = [
            _query_id(row) for row in rows
            if str(row.get("conversation_id", "") or "").strip()
            and int(row.get("turn_index", 1) or 1) > 1
        ]
        if later_turns:
            raise ValueError(
                f"{system} có query nhiều lượt cần ngữ cảnh trước đó: "
                f"{', '.join(later_turns[:10])}. Không nên retry riêng lượt sau; "
                "cần phát lại đầy đủ hội thoại để giữ điều kiện thí nghiệm."
            )

    print("=== RETRY PLAN ===")
    print(f"Benchmark query-turns: {len(benchmark)}")
    print(f"Original result records: {len(original_records)}")
    for system in BASELINE_SYSTEMS:
        print(f"{system}: {len(selected[system])} failed query(s)")
        if selected[system]:
            print("  " + ", ".join(_query_id(q) for q in selected[system]))
    print(f"Delay between baseline calls: {args.delay:.1f}s")
    if args.dry_run:
        print("Dry run: chưa gọi API và chưa tạo file output.")
        return

    total_to_retry = sum(len(rows) for rows in selected.values())
    if total_to_retry == 0:
        print("Không tìm thấy trường hợp nào cần retry. Không tạo file output.")
        return

    output_records: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="laptop_retry_") as temp_dir:
        temp_root = Path(temp_dir)
        for system in BASELINE_SYSTEMS:
            rows = selected[system]
            if not rows:
                continue

            input_path = temp_root / f"{system}_queries.json"
            temp_output = temp_root / f"{system}_results.jsonl"
            input_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"\n=== Retrying {system}: {len(rows)} query(s) ===")
            retried = run_benchmark(
                queries_file=input_path,
                output_file=temp_output,
                verbose=True,
                systems=system,
                append=False,
                baseline_delay_seconds=args.delay,
                comparison_only=False,
            )

            for record in retried:
                qid = str(record.get("query_id", "")).strip()
                prior = original_by_pair.get((qid, system), {})
                record["is_retry"] = True
                record["initial_status"] = prior.get("status")
                record["initial_api_errors"] = prior.get("api_errors", [])
                output_records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: Dict[str, Dict[str, int]] = {}
    for system in BASELINE_SYSTEMS:
        rows = [r for r in output_records if r.get("system") == system]
        counts[system] = {
            "retried": len(rows),
            "completed": sum(1 for r in rows if r.get("status") == "COMPLETED"),
            "still_api_failed": sum(1 for r in rows if r.get("status") == "API_FAILED"),
            "parse_failed": sum(1 for r in rows if r.get("status") == "PARSE_FAILED"),
            "other_status": sum(
                1 for r in rows
                if r.get("status") not in {"COMPLETED", "API_FAILED", "PARSE_FAILED"}
            ),
        }

    print("\n=== RETRY SUMMARY ===")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"Retry-only results saved to: {output_path}")
    print(f"Merged evaluation results saved to: {merged_path}")
    print("Kết quả gốc được giữ nguyên.")


if __name__ == "__main__":
    main()
