"""
Evaluation Metrics Module - Tính toán các chỉ số đánh giá thực nghiệm.

Không phụ thuộc framework ngoài, thuần Python / Pandas.
"""

from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd


def _to_df(records: List[Union[Dict[str, Any], Any]]) -> pd.DataFrame:
    """Chuyển list records (dict hoặc dataclass) thành pandas DataFrame."""
    if not records:
        return pd.DataFrame()
    rows = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in records]
    return pd.DataFrame(rows)


def constraint_satisfaction_rate(records: List[Union[Dict[str, Any], Any]]) -> float:
    """Tỷ lệ các truy vấn thỏa mãn 100% hard constraints (0.0 .. 1.0)."""
    df = _to_df(records)
    if df.empty or "hard_constraint_satisfied" not in df.columns:
        return 0.0
    return float(df["hard_constraint_satisfied"].astype(bool).mean())


def constraint_violation_rate(records: List[Union[Dict[str, Any], Any]]) -> float:
    """Tỷ lệ các truy vấn có vi phạm ràng buộc (1 - constraint_satisfaction_rate)."""
    return 1.0 - constraint_satisfaction_rate(records)


def product_hallucination_rate(
    records: List[Union[Dict[str, Any], Any]],
    catalog_product_names: Optional[Set[str]] = None,
) -> float:
    """
    Tỷ lệ đề xuất sản phẩm không tồn tại trong catalog (hallucination).

    Args:
        records: Danh sách bản ghi thực nghiệm.
        catalog_product_names: Set các tên/ID sản phẩm hợp lệ có trong database.
    """
    df = _to_df(records)
    if df.empty or "selected_product" not in df.columns:
        return 0.0

    if catalog_product_names is None:
        # Nếu không truyền catalog, kiểm tra trường hợp không chọn được sản phẩm hoặc product là None/empty
        missing = df["selected_product"].isna() | (df["selected_product"] == "")
        return float(missing.mean())

    catalog_normalized = {str(name).strip().lower() for name in catalog_product_names}

    def _is_hallucinated(val: Any) -> bool:
        if val is None or pd.isna(val) or val == "":
            return True
        if isinstance(val, dict):
            name = str(val.get("product_name") or val.get("laptop_name") or "").strip().lower()
        else:
            name = str(val).strip().lower()
        return name not in catalog_normalized

    hallucinated = df["selected_product"].apply(_is_hallucinated)
    return float(hallucinated.mean())


def average_llm_calls(records: List[Union[Dict[str, Any], Any]]) -> float:
    """Số lượt gọi LLM trung bình trên mỗi query."""
    df = _to_df(records)
    if df.empty or "llm_calls" not in df.columns:
        return 0.0
    return float(df["llm_calls"].mean())


def average_tokens(records: List[Union[Dict[str, Any], Any]]) -> Dict[str, float]:
    """Số lượng token trung bình (input, output, total) trên mỗi query."""
    df = _to_df(records)
    if df.empty:
        return {"avg_input_tokens": 0.0, "avg_output_tokens": 0.0, "avg_total_tokens": 0.0}

    in_tok = float(df.get("input_tokens", pd.Series(0)).mean())
    out_tok = float(df.get("output_tokens", pd.Series(0)).mean())
    return {
        "avg_input_tokens": round(in_tok, 2),
        "avg_output_tokens": round(out_tok, 2),
        "avg_total_tokens": round(in_tok + out_tok, 2),
    }


def average_latency(records: List[Union[Dict[str, Any], Any]]) -> Dict[str, float]:
    """Thời gian phản hồi trung bình (ms) tổng thể và theo từng thành phần pipeline."""
    df = _to_df(records)
    if df.empty:
        return {
            "avg_nlu_latency_ms": 0.0,
            "avg_scoring_latency_ms": 0.0,
            "avg_optimization_latency_ms": 0.0,
            "avg_total_latency_ms": 0.0,
        }

    return {
        "avg_nlu_latency_ms": round(float(df.get("nlu_latency_ms", pd.Series(0)).mean()), 2),
        "avg_scoring_latency_ms": round(float(df.get("scoring_latency_ms", pd.Series(0)).mean()), 2),
        "avg_optimization_latency_ms": round(float(df.get("optimization_latency_ms", pd.Series(0)).mean()), 2),
        "avg_total_latency_ms": round(float(df.get("total_latency_ms", pd.Series(0)).mean()), 2),
    }


def compute_all_metrics(
    records: List[Union[Dict[str, Any], Any]],
    catalog_product_names: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Tính toán toàn bộ các metric đánh giá trên tập experiment records.
    """
    if not records:
        return {}

    tokens = average_tokens(records)
    latencies = average_latency(records)

    return {
        "sample_count": len(records),
        "constraint_satisfaction_rate": round(constraint_satisfaction_rate(records), 4),
        "constraint_violation_rate": round(constraint_violation_rate(records), 4),
        "product_hallucination_rate": round(product_hallucination_rate(records, catalog_product_names), 4),
        "average_llm_calls": round(average_llm_calls(records), 2),
        **tokens,
        **latencies,
    }
