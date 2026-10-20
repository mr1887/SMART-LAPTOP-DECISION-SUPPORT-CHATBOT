"""Catalog-grounded LLM baseline for the research benchmark.

This baseline gives the LLM the same study catalog used by the Hybrid system,
but it does NOT use LightGBM, retrieval, or OR-Tools. The LLM must choose one
catalog item directly. This isolates the value of catalog grounding from the
additional value of symbolic optimization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from app.ai.gemini_service import _generate_with_fallback, get_last_generation_diagnostics


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

CATALOG_PROMPT_TEMPLATE = """Bạn là hệ thống tư vấn laptop trong một thí nghiệm nghiên cứu.

NHIỆM VỤ:
- Đọc yêu cầu người dùng.
- Chỉ được chọn DUY NHẤT một laptop có trong CATALOG bên dưới.
- Không được tạo model mới, không đổi tên model, không sửa thông số catalog.
- Tự suy luận hard/soft constraints và sở thích của người dùng.
- Nếu không có sản phẩm nào đáp ứng hoàn toàn, vẫn chọn sản phẩm gần nhất trong catalog
  và nêu ngắn gọn tiêu chí nào không được đáp ứng.
- Không sử dụng công cụ tối ưu hóa, solver, LightGBM hay retrieval bên ngoài catalog đã cho.

Yêu cầu người dùng:
"{query}"

CATALOG (mỗi dòng là một sản phẩm; toàn bộ catalog nghiên cứu được cung cấp):
{catalog}

Trả về DUY NHẤT JSON hợp lệ, không markdown, theo schema:
{{
  "product_name": "<phải khớp chính xác laptop_name trong catalog>",
  "claimed_specs": {{
    "price_vnd": <số hoặc null>,
    "cpu": "<chuỗi hoặc null>",
    "gpu": "<chuỗi hoặc null>",
    "ram_gb": <số hoặc null>,
    "storage_gb": <số hoặc null>,
    "weight_kg": <số hoặc null>,
    "battery_minutes": <số hoặc null>
  }},
  "reasoning": "<lý do ngắn gọn>",
  "unmet_requirements": ["<tiêu chí không đáp ứng nếu có>"]
}}
"""


def find_catalog_path(explicit_path: Optional[Path] = None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Không tìm thấy catalog: {path}")

    candidates = [
        _PROJECT_ROOT / "data" / "processed" / "laptop_dataset_scored.csv",
        _PROJECT_ROOT / "data" / "processed" / "laptop_dataset_tagged.csv",
        _PROJECT_ROOT / "data" / "processed" / "laptop_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Không tìm thấy catalog đã xử lý. Kỳ vọng một trong: "
        "laptop_dataset_scored.csv, laptop_dataset_tagged.csv, laptop_dataset.csv"
    )


def _first_existing(columns: Iterable[str], options: Iterable[str]) -> Optional[str]:
    available = set(columns)
    for name in options:
        if name in available:
            return name
    return None


def _clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def build_catalog_summary(catalog_path: Optional[Path] = None) -> Dict[str, Any]:
    """Return a compact text representation containing every catalog row.

    Only decision-relevant columns are included to keep token usage measurable
    and avoid giving the grounded baseline internal Hybrid-only scores.
    """
    path = find_catalog_path(catalog_path)
    df = pd.read_csv(path)

    aliases = {
        "id": ["laptop_model_id", "id"],
        "laptop_name": ["laptop_name", "name"],
        "brand": ["brand_name", "brand"],
        "cpu": ["cpu_name", "cpu"],
        "gpu": ["gpu_name", "gpu"],
        "price": ["price", "price_vnd"],
        "ram_gb": ["ram_gb", "ram", "memory_gb"],
        "storage_gb": ["storage_gb", "storage", "ssd_gb"],
        "weight_kg": ["laptop_weight", "weight_kg", "weight"],
        "battery_minutes": ["office_battery_minutes_final", "battery_minutes"],
        "screen_size": ["screen_size", "screen_size_inch"],
        "gaming": ["is_gaming_friendly", "is_gaming_laptop"],
        "programming": ["is_programming_friendly"],
        "graphics": ["is_graphic_friendly", "is_workstation"],
        "office": ["is_office_friendly"],
    }

    resolved = []
    for label, options in aliases.items():
        col = _first_existing(df.columns, options)
        if col is not None:
            resolved.append((label, col))

    if not any(label == "laptop_name" for label, _ in resolved):
        raise ValueError("Catalog không có cột laptop_name/name để định danh sản phẩm.")

    header = "\t".join(label for label, _ in resolved)
    rows = [header]
    for _, row in df.iterrows():
        rows.append("\t".join(_clean_value(row[col]) for _, col in resolved))

    return {
        "path": str(path),
        "row_count": int(len(df)),
        "columns": [label for label, _ in resolved],
        "text": "\n".join(rows),
    }


def recommend_catalog_grounded_llm(
    query: str,
    catalog_summary: str,
) -> Dict[str, Any]:
    prompt = CATALOG_PROMPT_TEMPLATE.format(
        query=query.strip(),
        catalog=catalog_summary,
    )

    raw_response = _generate_with_fallback(
        prompt=prompt,
        config={"temperature": 0.2, "response_mime_type": "application/json"},
        retry_rounds=3,
        retry_delay_seconds=2.0,
    )
    diagnostics = get_last_generation_diagnostics()

    base = {
        "api_attempts": diagnostics.get("attempts", 0),
        "api_model": diagnostics.get("model"),
        "api_errors": diagnostics.get("errors", []),
        "input_tokens": diagnostics.get("input_tokens", 0),
        "output_tokens": diagnostics.get("output_tokens", 0),
        "thinking_tokens": diagnostics.get("thinking_tokens", 0),
        "total_tokens": diagnostics.get("total_tokens", 0),
        "estimated_api_cost_usd": diagnostics.get("estimated_cost_usd", 0.0),
        "usage_by_model": diagnostics.get("usage_by_model", {}),
    }

    if not raw_response:
        return {
            "product_name": None,
            "claimed_specs": {},
            "reasoning": "",
            "unmet_requirements": [],
            "raw_response": "",
            "api_success": False,
            **base,
        }

    data: Dict[str, Any] = {}
    try:
        data = json.loads(raw_response)
    except Exception:
        # response_mime_type should normally prevent this, but preserve failures
        # instead of silently repairing arbitrary text.
        data = {}

    return {
        "product_name": data.get("product_name"),
        "claimed_specs": data.get("claimed_specs", {}) if isinstance(data, dict) else {},
        "reasoning": data.get("reasoning", "") if isinstance(data, dict) else "",
        "unmet_requirements": data.get("unmet_requirements", []) if isinstance(data, dict) else [],
        "raw_response": raw_response,
        "api_success": True,
        **base,
    }
