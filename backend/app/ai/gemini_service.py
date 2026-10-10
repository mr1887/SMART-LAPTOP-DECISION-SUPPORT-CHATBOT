"""
Module tích hợp Google Gemini API sử dụng SDK mới nhất (google-genai):
1. Trích xuất ràng buộc thông minh (Gemini NLP Extraction)
2. Sinh lời tư vấn cá nhân hóa (Gemini Decision Explanation)
3. Phản hồi giao tiếp tự nhiên (Gemini Conversational Feedback)

Tự động nhận diện GEMINI_API_KEY từ file .env hoặc biến môi trường.
Nếu không có API Key hoặc gặp lỗi mạng, module sẽ trả về None để hệ thống
tự động chuyển sang phương án fallback (Regex & Template gốc).
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Tự động tìm và nạp file .env từ thư mục gốc dự án
_ENV_PATH = Path(__file__).parents[3] / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
else:
    load_dotenv(override=True)

_client = None
_last_generation_diagnostics = {"success": False, "model": None, "attempts": 0, "errors": []}


def get_last_generation_diagnostics() -> dict:
    """Return diagnostics for the most recent Gemini call; never includes secrets."""
    return {
        "success": bool(_last_generation_diagnostics.get("success")),
        "model": _last_generation_diagnostics.get("model"),
        "attempts": int(_last_generation_diagnostics.get("attempts", 0)),
        "errors": list(_last_generation_diagnostics.get("errors", [])),
    }

DEFAULT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]


def _init_gemini():
    """Khởi tạo Gemini client từ thư viện mới google-genai."""
    global _client
    if _client is not None:
        return _client

    if _ENV_PATH.exists():
        load_dotenv(dotenv_path=_ENV_PATH, override=True)
    else:
        load_dotenv(override=True)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "your_gemini_api_key_here":
        return None

    try:
        from google import genai
        _client = genai.Client(api_key=api_key.strip())
        return _client
    except Exception as e:
        print(f"[Google GenAI Warning] Không thể khởi tạo Client: {e}")
        return None


def _generate_with_fallback(
    prompt: str,
    config: Optional[dict] = None,
    retry_rounds: int = 1,
    retry_delay_seconds: float = 2.0,
) -> Optional[str]:
    """Call Gemini with model fallback and optional retry rounds."""
    global _last_generation_diagnostics
    _last_generation_diagnostics = {"success": False, "model": None, "attempts": 0, "errors": []}

    client = _init_gemini()
    if client is None:
        _last_generation_diagnostics["errors"].append("Gemini client unavailable or API key missing")
        return None

    try:
        from google.genai import types
        gen_config = types.GenerateContentConfig(**(config or {}))
    except Exception:
        gen_config = None

    rounds = max(1, int(retry_rounds or 1))
    delay = max(0.0, float(retry_delay_seconds or 0.0))
    for round_idx in range(rounds):
        for m_name in DEFAULT_MODELS:
            _last_generation_diagnostics["attempts"] += 1
            try:
                if gen_config:
                    response = client.models.generate_content(model=m_name, contents=prompt, config=gen_config)
                else:
                    response = client.models.generate_content(model=m_name, contents=prompt)
                if response and response.text:
                    _last_generation_diagnostics["success"] = True
                    _last_generation_diagnostics["model"] = m_name
                    return response.text.strip()
                _last_generation_diagnostics["errors"].append(f"{m_name}: empty response")
            except Exception as exc:
                msg = str(exc).replace("\n", " ")[:240]
                _last_generation_diagnostics["errors"].append(f"{m_name}: {type(exc).__name__}: {msg}")
        if round_idx < rounds - 1 and delay > 0:
            time.sleep(delay * (2 ** round_idx))
    return None


# ==============================================================================
# 1. Trích xuất ràng buộc thông minh bằng Gemini
# ==============================================================================
EXTRACTION_SYSTEM_PROMPT = """
Bạn là bộ phân tích ngôn ngữ tự nhiên cho chatbot tư vấn chọn laptop.
Nhiệm vụ của bạn là đọc tin nhắn người dùng (tiếng Việt) và trích xuất các ràng buộc (constraints), sở thích (preferences) và nhãn nhu cầu (required_tags).

CÁC TRƯỜNG ĐƯỢC PHÉP (allowed fields):
- price: Ngân sách tính theo VNĐ đầy đủ (VD: 20 triệu -> 20000000, 15tr -> 15000000, 25 củ -> 25000000).
- ram_gb: Dung lượng RAM tính theo GB (VD: 16GB -> 16, 32GB -> 32).
- storage_gb: Dung lượng ổ cứng tính theo GB (VD: 512GB -> 512, 1TB -> 1024).
- weight_kg: Trọng lượng máy tính theo kg (VD: 1.5kg -> 1.5, dưới 2kg -> 2.0).
- battery_minutes: Thời lượng pin tính theo PHÚT (VD: 6 tiếng -> 360, 8h -> 480).
- gpu_discrete: Boolean (true nếu cần card rời/GPU rời NVIDIA, RTX, GTX; false nếu card tích hợp).
- gpu_keyword: Tên dòng GPU cụ thể nếu có (VD: "RTX 4060", "RTX 3050", "RTX 4070", "M4"...).

QUY TẮC PHÂN LOẠI SEMANTIC:
1. Ràng buộc (constraints):
   - operator: chỉ một trong các toán tử "<=", ">=", "="
   - type:
     + "hard": Các từ khóa thể hiện tính bắt buộc, giới hạn cứng: "bắt buộc", "không quá", "ít nhất", "phải có", "tối đa", "tối thiểu", "dưới", "trên".
       (VD: "bắt buộc dưới 20 triệu" -> field: price, operator: "<=", value: 20000000, type: "hard")
       (VD: "RAM ít nhất 16GB" -> field: ram_gb, operator: ">=", value: 16, type: "hard")
       (VD: "SSD ít nhất 512GB" -> field: storage_gb, operator: ">=", value: 512, type: "hard")
     + "soft": Các từ khóa ước lượng, mềm dẻo: "tầm", "khoảng", "tầm khoảng", "xung quanh", "nếu được".
       (VD: "tầm 20 triệu" -> field: price, operator: "<=", value: 20000000, type: "soft")
   - source_text: Trích xuất chính xác đoạn văn bản gốc tương ứng của người dùng.

2. Sở thích tối ưu (preferences):
   - direction: "minimize", "maximize", "prefer"
   - Sử dụng khi người dùng muốn tối ưu hóa nhưng không đặt ngưỡng cứng cụ thể:
     + "càng nhẹ càng tốt", "ưu tiên mỏng nhẹ" -> field: "weight_kg", direction: "minimize"
     + "càng rẻ càng tốt", "tiết kiệm nhất có thể" -> field: "price", direction: "minimize"
     + "pin càng trâu càng tốt", "ưu tiên pin lâu" -> field: "battery_minutes", direction: "maximize"
     + "ưu tiên RAM nhiều", "RAM càng to càng tốt" -> field: "ram_gb", direction: "maximize"
   - source_text: Trích xuất chính xác đoạn văn bản gốc tương ứng.

3. Nhãn nhu cầu (required_tags):
   - Chọn từ các tag chuẩn: "is_gaming_friendly", "is_programming_friendly", "is_graphic_friendly", "is_office_friendly".

ĐỊNH DẠNG JSON ĐẦU RA BẮT BUỘC:
{
  "constraints": [
    {
      "field": "price|ram_gb|storage_gb|weight_kg|battery_minutes|gpu_discrete|gpu_keyword",
      "operator": "<=|>=|=",
      "value": 20000000,
      "type": "hard|soft",
      "source_text": "tầm 20 triệu"
    }
  ],
  "preferences": [
    {
      "field": "weight_kg|price|battery_minutes|ram_gb|storage_gb",
      "direction": "minimize|maximize|prefer",
      "source_text": "càng nhẹ càng tốt"
    }
  ],
  "required_tags": ["is_gaming_friendly"]
}

QUY TẮC QUAN TRỌNG:
- Chỉ trả về duy nhất 1 JSON object hợp lệ, không kèm theo bất kỳ văn bản giải thích nào khác ngoài JSON.
- Nếu không có ràng buộc/sở thích nào, trả về danh sách rỗng [] cho các trường.
"""


def extract_constraints_gemini(user_message: str, current_constraints: Optional[dict] = None) -> Optional[dict]:
    """Sử dụng Google GenAI API để trích xuất ràng buộc theo schema mới. Trả về None nếu không khả dụng."""
    try:
        context_str = ""
        if current_constraints:
            context_str = f"\nRàng buộc hiện tại trước đó: {json.dumps(current_constraints, ensure_ascii=False)}"

        prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n{context_str}\nTin nhắn người dùng: \"{user_message}\"\n\nJSON output:"
        
        raw_text = _generate_with_fallback(
            prompt=prompt,
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        )

        if not raw_text:
            return None

        # Clean markdown formatting nếu có
        raw_text = re.sub(r"^```json\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        
        data = json.loads(raw_text)

        constraints = []
        for c in data.get("constraints", []):
            if isinstance(c, dict) and "field" in c and "operator" in c and "value" in c:
                constraints.append({
                    "field": c.get("field"),
                    "operator": c.get("operator"),
                    "value": c.get("value"),
                    "type": c.get("type", "hard"),
                    "source_text": c.get("source_text"),
                })

        preferences = []
        for p in data.get("preferences", []):
            if isinstance(p, dict) and "field" in p and "direction" in p:
                preferences.append({
                    "field": p.get("field"),
                    "direction": p.get("direction"),
                    "source_text": p.get("source_text"),
                })

        required_tags = [
            tag for tag in data.get("required_tags", []) if isinstance(tag, str)
        ]

        return {
            "constraints": constraints,
            "preferences": preferences,
            "required_tags": required_tags,
        }
    except Exception as e:
        print(f"[Google GenAI NLP Error] Lỗi trích xuất ({e}), chuyển sang fallback.")
        return None


# ==============================================================================
# 2. Sinh lời tư vấn cá nhân hóa & Tiếp nhận ý kiến bằng Gemini
# ==============================================================================
CONSULTATION_SYSTEM_PROMPT = """
Bạn là Chuyên gia Tư vấn Laptop AI cao cấp, nhiệt tình, am hiểu sâu sắc về công nghệ và tâm lý khách hàng của hệ thống "Smart Laptop Decision Support System".

Hệ thống đã phân tích dữ liệu kỹ thuật và chọn ra mẫu laptop phù hợp nhất cho người dùng.

NHIỆM VỤ CỦA BẠN:
1. Giao tiếp tự nhiên, lôi cuốn và chuyên nghiệp:
   - Xưng hô lịch sự, thân thiện (mình/em - bạn/anh/chị).
   - Tuyệt đối KHÔNG sử dụng các thuật ngữ kỹ thuật khô khan hoặc nội bộ hệ thống như: "[Gurobi]", "[PuLP]", "Laptop ID #...", "AI_Score (0.xxx)", "Binary Integer Programming".
   - QUY TẮC BẮT BUỘC: KHÔNG ĐƯỢC DÙNG CÁC KÝ HIỆU EMOJI ĐẶC BIỆT (như 💰, 🎯, ⚙️, 🖥️, 🏢, 🏆, 📺, 🖼️, 👉, ✨, 💡, ⚠️, ⭐). Hãy dùng định dạng gạch đầu dòng, in đậm Markdown chuẩn và sạch sẽ.
   - Thay vì nói "AI_Score", hãy diễn đạt một cách tự nhiên: "Điểm đánh giá tối ưu: X / 10" (dựa trên cân bằng giữa cấu hình, màn hình và tầm giá).

2. Trình bày sản phẩm rõ ràng, nổi bật:
   - Nêu bật tên mẫu laptop được đề xuất.
   - Trình bày thông số chính bằng gạch đầu dòng Markdown rõ ràng:
     * Mức giá tham khảo (VNĐ)
     * Cấu hình (CPU, GPU - nêu rõ card rời hay tích hợp)
     * Màn hình hiển thị (kích thước, độ phân giải nếu có)
     * Thương hiệu sản xuất & Năm ra mắt
     * Nhu cầu sử dụng phù hợp nhất (Lập trình, Đồ họa, Gaming, Văn phòng...)
   - QUY TẮC QUAN TRỌNG: TUYỆT ĐỐI KHÔNG đề cập đến Thời lượng pin hoặc Trọng lượng/Cân nặng.
   - Phân tích thực tế: Vì sao chiếc máy này là lựa chọn tuyệt vời cho nhu cầu của khách hàng.

3. Xử lý trường hợp nới lỏng tiêu chí (is_relaxed = True / Nghiệm nới lỏng):
   - Bạn BẮT BUỘC phải thông báo minh bạch cho khách hàng rằng KHÔNG có sản phẩm nào trên thị trường đáp ứng 100% tất cả các tiêu chí ban đầu.
   - Chỉ ra lý do và mốc chênh lệch cụ thể (VD: "Thị trường hiện chưa có laptop nào giá 10 triệu mà sở hữu card RTX 4060, các dòng laptop có RTX 4060 khởi điểm từ 22 triệu VNĐ, vượt 12 triệu so với ngân sách của bạn").
   - Khẳng định đây là **NGHIỆM GẦN NHẤT (Nghiệm nới lỏng ràng buộc)** do bộ giải toán chọn ra.

4. Đa phương tiện & Gợi ý hành động:
   - Nhắc đến việc khách hàng có thể xem video review trên YouTube và xem ảnh chi tiết ở các đường liên kết đính kèm.
   - Đặt câu hỏi mở để tiếp tục hỗ trợ nếu khách muốn thay đổi tiêu chí.
"""

CONVERSATIONAL_FEEDBACK_PROMPT = """
Bạn là Chuyên gia Tư vấn Laptop AI thân thiện, thông minh và chu đáo.
Người dùng vừa gửi một tin nhắn chào hỏi, trò chuyện phiếm, hỏi đáp chung hoặc phản hồi/nhận xét (không phải là câu yêu cầu lọc cấu hình máy cụ thể).

NHIỆM VỤ CỦA BẠN:
1. Nếu là Chào hỏi ("hello", "chào bạn", "alo"...):
   - Chào lại một cách niềm nở, tươi vui và lịch sự.
   - Giới thiệu ngắn gọn bạn là trợ lý AI chuyên tư vấn laptop tối ưu.
   - Hướng dẫn khách hàng nêu nhu cầu (ví dụ: tầm giá bao nhiêu, mua để học tập/văn phòng, lập trình IT, làm đồ họa 3D hay chơi game, cần pin trâu hay siêu mỏng nhẹ...).
2. Nếu là phản hồi/góp ý (chê đắt, chê nặng, khen hay, cảm ơn...):
   - Lắng nghe và đồng cảm, phản hồi cầu thị.
   - Gợi ý điều chỉnh (VD: "Nếu bạn muốn mức giá mềm hơn, mình có thể tìm giúp bạn các dòng tầm 15-20 triệu nhé!").
3. Nếu là câu hỏi ngoài lề hoặc không phù hợp:
   - Trả lời khéo léo, lịch thiệp và nhẹ nhàng dẫn dắt người dùng quay lại chủ đề tư vấn laptop.
"""


def generate_gemini_consultation(
    user_message: str,
    constraints: dict,
    result: dict,
    laptop_details: Optional[dict] = None
) -> Optional[str]:
    """Sinh lời tư vấn thông minh từ Google GenAI dựa trên kết quả giải toán và đa phương tiện."""
    try:
        prompt_data = {
            "user_message": user_message,
            "applied_constraints": constraints,
            "solver_result": {
                "laptop_id": result.get("laptop_id"),
                "is_feasible": result.get("is_feasible"),
                "is_relaxed": result.get("is_relaxed"),
                "ai_score": result.get("ai_score"),
                "solver_explanation": result.get("explanation"),
            },
            "laptop_specs": laptop_details or {},
        }

        prompt = (
            f"{CONSULTATION_SYSTEM_PROMPT}\n\n"
            f"DỮ LIỆU ĐẦU VÀO TỪ HỆ THỐNG:\n"
            f"{json.dumps(prompt_data, ensure_ascii=False, indent=2)}\n\n"
            f"Hãy viết câu trả lời tư vấn hoàn chỉnh, tự nhiên và chuyên nghiệp gửi tới khách hàng:"
        )

        return _generate_with_fallback(
            prompt=prompt,
            config={"temperature": 0.5}
        )
    except Exception as e:
        print(f"[Google GenAI Advisory Error] Lỗi sinh tư vấn ({e}), chuyển sang fallback.")
        return None


def generate_conversational_reply(
    user_message: str,
    current_constraints: Optional[dict] = None,
    last_laptop_details: Optional[dict] = None
) -> Optional[str]:
    """Tiếp nhận ý kiến khách hàng, chào hỏi và giải đáp thắc mắc tự nhiên khi không cần gọi solver."""
    try:
        context = {
            "user_message": user_message,
            "current_constraints": current_constraints or {},
            "last_viewed_laptop": last_laptop_details or {},
        }

        prompt = (
            f"{CONVERSATIONAL_FEEDBACK_PROMPT}\n\n"
            f"NGỮ CẢNH HỘI THOẠI:\n"
            f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
            f"Hãy phản hồi khách hàng một cách tự nhiên, duyên dáng nhất:"
        )

        return _generate_with_fallback(
            prompt=prompt,
            config={"temperature": 0.6}
        )
    except Exception as e:
        print(f"[Google GenAI Conversational Error] Lỗi phản hồi giao tiếp ({e}).")
        return None
