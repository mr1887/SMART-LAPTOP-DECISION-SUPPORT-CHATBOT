"""
Baseline module: LLM-Only Direct Recommendation (Không dùng LightGBM & không dùng OR-Tools).

Mục đích nghiên cứu (Research baseline):
- Đo lường khả năng của LLM thuần (Gemini) khi tự gợi ý sản phẩm trực tiếp từ query người dùng.
- Làm cơ sở so sánh (baseline) với kiến trúc lai kết hợp NLU + LightGBM Scoring + OR-Tools Optimizer.
- Phát hiện các hiện tượng hallucination về cấu hình hoặc giá cả khi không có solver/database grounding.
"""

import json
import re
from typing import Any, Dict, Optional

from app.ai.gemini_service import _generate_with_fallback, get_last_generation_diagnostics


LLM_ONLY_PROMPT_TEMPLATE = """Bạn là chuyên gia tư vấn laptop. Dựa vào yêu cầu của người dùng, hãy đề xuất 1 mẫu laptop phù hợp nhất.

Yêu cầu của người dùng:
"{query}"
{catalog_context}
Hãy trả về DUY NHẤT một khối JSON (không thêm text ngoài JSON) với cấu trúc sau:
{{
  "product_name": "<Tên đầy đủ và chính xác của laptop được đề xuất>",
  "claimed_specs": {{
    "price_vnd": <mức giá ước tính VNĐ hoặc số nguyên>,
    "cpu": "<tên CPU>",
    "gpu": "<tên GPU>",
    "ram_gb": <dung lượng RAM GB>,
    "storage_gb": <dung lượng ổ cứng GB>,
    "weight_kg": <trọng lượng kg hoặc null>,
    "screen_size_inch": <kích thước màn hình hoặc null>
  }},
  "reasoning": "<lý do ngắn gọn đề xuất mẫu máy này>"
}}
"""


def recommend_llm_only(
    query: str,
    product_catalog_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Thực hiện gợi ý laptop trực tiếp bằng LLM thuần (Gemini) mà không qua LightGBM và OR-Tools.

    Args:
        query: Câu hỏi / nhu cầu tìm kiếm của người dùng.
        product_catalog_summary: (Tùy chọn) Tóm tắt danh mục sản phẩm nếu muốn cung cấp context.

    Returns:
        dict với format:
        {
            "product_name": str,
            "claimed_specs": dict,
            "raw_response": str
        }
    """
    catalog_context = ""
    if product_catalog_summary:
        catalog_context = f"\nThông tin danh mục tham khảo:\n{product_catalog_summary}\n"

    prompt = LLM_ONLY_PROMPT_TEMPLATE.format(
        query=query.strip(),
        catalog_context=catalog_context,
    )

    raw_response = _generate_with_fallback(
        prompt=prompt,
        config={"temperature": 0.2, "response_mime_type": "application/json"},
        retry_rounds=3,
        retry_delay_seconds=2.0,
    )
    diagnostics = get_last_generation_diagnostics()

    if not raw_response:
        return {
            "product_name": None,
            "claimed_specs": {},
            "raw_response": "",
            "api_success": False,
            "api_attempts": diagnostics.get("attempts", 0),
            "api_model": diagnostics.get("model"),
        }

    # Parse JSON từ response của Gemini
    product_name = None
    claimed_specs = {}

    try:
        data = json.loads(raw_response)
        product_name = data.get("product_name")
        claimed_specs = data.get("claimed_specs", {})
    except Exception:
        # Fallback regex nếu model trả về markdown code block
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                product_name = data.get("product_name")
                claimed_specs = data.get("claimed_specs", {})
            except Exception:
                pass

    return {
        "product_name": product_name,
        "claimed_specs": claimed_specs,
        "raw_response": raw_response,
        "api_success": True,
        "api_attempts": diagnostics.get("attempts", 0),
        "api_model": diagnostics.get("model"),
    }
