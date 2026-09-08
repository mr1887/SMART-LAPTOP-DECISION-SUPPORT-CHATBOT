"""
Route /api/chat - Endpoint chính cho chatbot tư vấn laptop.

Flow mỗi lượt chat:
    1. Nhận message từ user (+ session_id nếu tiếp tục hội thoại cũ)
    2. NLP (nl2constraint) trích xuất ràng buộc từ câu hỏi
    3. Merge ràng buộc vào session (tích lũy multi-turn)
    4. Load dataset laptop đã chấm điểm (AI_Score)
    5. Gọi solver để tìm laptop tối ưu
    6. Trả về kết quả + câu trả lời tự nhiên
"""

import uuid
import urllib.parse
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.nlp import nl2constraint
from app.optimizer import solver
from app.session import session_manager

router = APIRouter()

_df_cache: pd.DataFrame | None = None
_video_url_cache: dict[int, str] | None = None


def _find_scored_csv() -> Path:
    candidates = [
        Path.cwd() / "data" / "processed" / "laptop_dataset_scored.csv",
        Path("/app/data/processed/laptop_dataset_scored.csv"),
        Path(__file__).resolve().parents[4] / "data" / "processed" / "laptop_dataset_scored.csv",
        Path(__file__).resolve().parents[3] / "data" / "processed" / "laptop_dataset_scored.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _find_benchmark_csv() -> Path:
    candidates = [
        Path.cwd() / "data" / "raw" / "laptop_benchmark_result.csv",
        Path("/app/data/raw/laptop_benchmark_result.csv"),
        Path(__file__).resolve().parents[4] / "data" / "raw" / "laptop_benchmark_result.csv",
        Path(__file__).resolve().parents[3] / "data" / "raw" / "laptop_benchmark_result.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _load_scored_df() -> pd.DataFrame:
    """Lazy-load và cache dataset laptop đã chấm điểm."""
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    csv_path = _find_scored_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset tại {csv_path}. "
            "Vui lòng kiểm tra thư mục data/processed/laptop_dataset_scored.csv."
        )
    _df_cache = pd.read_csv(csv_path)
    return _df_cache


def _get_benchmark_video_urls() -> dict[int, str]:
    """Cache link video review YouTube từ dữ liệu benchmark."""
    global _video_url_cache
    if _video_url_cache is not None:
        return _video_url_cache
    _video_url_cache = {}
    csv_path = _find_benchmark_csv()
    if csv_path.exists():
        try:
            bdf = pd.read_csv(csv_path)
            if "laptop_model_id" in bdf.columns and "review_video_url" in bdf.columns:
                valid = bdf.dropna(subset=["laptop_model_id", "review_video_url"])
                for _, row in valid.iterrows():
                    url_str = str(row["review_video_url"]).strip()
                    if url_str.startswith("http"):
                        _video_url_cache[int(row["laptop_model_id"])] = url_str
        except Exception:
            pass
    return _video_url_cache


# ---------- Schema ----------

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    constraints: dict
    result: dict | None = None
    laptop_details: dict | None = None


# ---------- Helper ----------

def _get_laptop_details(df: pd.DataFrame, laptop_id: int | None) -> dict | None:
    """Trích xuất thông số kỹ thuật chi tiết của laptop kèm link YouTube review & hình ảnh (bỏ Pin & Cân nặng)."""
    if laptop_id is None or df is None or df.empty:
        return None
    
    matches = df[df["laptop_model_id"] == laptop_id]
    if matches.empty:
        return None
    
    row = matches.iloc[0]
    
    laptop_name = str(row.get("laptop_name", f"Laptop #{laptop_id}"))
    brand_name = str(row.get("brand_name", ""))
    year = int(row.get("year_introduce")) if pd.notnull(row.get("year_introduce")) else None

    # Màn hình
    screen_size = row.get("screen_size")
    screen_w = row.get("screen_dimension_width")
    screen_h = row.get("screen_dimension_height")
    screen_str = None
    if pd.notnull(screen_size):
        if pd.notnull(screen_w) and pd.notnull(screen_h):
            screen_str = f"{float(screen_size):.1f}\" ({int(screen_w)}x{int(screen_h)})"
        else:
            screen_str = f"{float(screen_size):.1f} inch"

    # Phù hợp cho nhu cầu gì
    suitable_roles = []
    if row.get("is_gaming_friendly") == 1 or row.get("is_gaming_friendly") is True:
        suitable_roles.append("Gaming / Chơi game")
    if row.get("is_programming_friendly") == 1 or row.get("is_programming_friendly") is True:
        suitable_roles.append("Lập trình / CNTT")
    if row.get("is_graphic_friendly") == 1 or row.get("is_graphic_friendly") is True:
        suitable_roles.append("Đồ họa / Thiết kế")
    if row.get("is_office_friendly") == 1 or row.get("is_office_friendly") is True:
        suitable_roles.append("Văn phòng / Học tập")

    # 1. Link YouTube Review: Ưu tiên link từ benchmark, fallback sang YouTube Search query
    video_map = _get_benchmark_video_urls()
    if int(laptop_id) in video_map:
        review_video_url = video_map[int(laptop_id)]
    else:
        search_kw = urllib.parse.quote(f"Đánh giá {brand_name} {laptop_name}".strip())
        review_video_url = f"https://www.youtube.com/results?search_query={search_kw}"
    
    # 2. Link tìm kiếm hình ảnh sản phẩm chi tiết
    img_kw = urllib.parse.quote(f"{brand_name} {laptop_name} laptop".strip())
    image_search_url = f"https://www.google.com/search?tbm=isch&q={img_kw}"

    return {
        "laptop_model_id": int(laptop_id),
        "laptop_name": laptop_name,
        "brand_name": brand_name,
        "year": year,
        "screen": screen_str,
        "suitable_roles": ", ".join(suitable_roles) if suitable_roles else "Đa dụng",
        "price_vnd": float(row.get("price", 0)),
        "cpu": str(row.get("cpu_name", row.get("laptop_cpu_note", ""))),
        "gpu": str(row.get("gpu_name", row.get("laptop_gpu_note", ""))),
        "ai_score": float(row.get("AI_Score", 0)) if pd.notnull(row.get("AI_Score")) else None,
        "review_video_url": review_video_url,
        "image_search_url": image_search_url,
    }


def _build_reply(constraints: dict, result: dict | None, laptop_details: dict | None = None) -> str:
    """Tạo câu trả lời tự nhiên, thân thiện và chuyên nghiệp từ kết quả solver (không dùng emoji đặc biệt)."""
    if result is None:
        return (
            "Chào bạn! Mình là trợ lý AI chuyên tư vấn laptop tối ưu.\n\n"
            "Bạn có thể cho mình biết **khoảng ngân sách** (ví dụ: tầm 15-20 triệu) và "
            "**nhu cầu sử dụng** (học tập, văn phòng, đồ họa, lập trình hay chơi game...) để mình gợi ý mẫu máy phù hợp nhất nhé!"
        )

    laptop_id = result.get("laptop_id")
    is_relaxed = result.get("is_relaxed", False)
    is_feasible = result.get("is_feasible", False)
    score = result.get("ai_score")

    if not is_feasible and not is_relaxed and laptop_id is None:
        return (
            "Hiện tại hệ thống chưa tìm thấy mẫu laptop nào đáp ứng trọn vẹn tất cả các yêu cầu trên. "
            "Bạn có thể thử nới lỏng ngân sách hoặc giảm bớt một số điều kiện khắt khe để mình tìm thêm nhé!"
        )

    # Lấy thông tin chi tiết
    laptop_name = (laptop_details.get("laptop_name") if laptop_details else None) or f"Laptop #{laptop_id}"
    price_vnd = laptop_details.get("price_vnd") if laptop_details else None
    cpu = laptop_details.get("cpu", "") if laptop_details else ""
    gpu = laptop_details.get("gpu", "") if laptop_details else ""
    screen = laptop_details.get("screen") if laptop_details else None
    brand_name = laptop_details.get("brand_name", "") if laptop_details else ""
    year = laptop_details.get("year") if laptop_details else None
    suitable_roles = laptop_details.get("suitable_roles") if laptop_details else None

    # Điểm đánh giá thân thiện
    score_display = f"{score * 10:.1f} / 10" if score is not None else "Đánh giá cao"

    # Header & Cảnh báo nới lỏng
    violation_text = result.get("violation_text") or result.get("explanation") or ""
    if is_feasible:
        header = f"**Em tìm thấy mẫu laptop tối ưu nhất thỏa mãn 100% tiêu chí của bạn:**\n\n### **{laptop_name}**"
    else:
        header = (
            f"**Thông báo tiêu chí (Nghiệm nới lỏng ràng buộc):**\n"
            f"Rất tiếc, trên thị trường hiện không có mẫu laptop nào thỏa mãn 100% đồng thời mọi tiêu chí của bạn "
            f"({violation_text}).\n\n"
            f"**Nghiệm gần nhất từ bộ giải toán (Soft Constraint):**\n"
            f"Hệ thống đã tự động nới lỏng tiêu chí và tìm ra mẫu laptop tiệm cận nhất với yêu cầu của bạn:\n\n"
            f"### **{laptop_name}**"
        )

    # Thông số cấu hình (Không chứa pin và cân nặng, không chứa emoji)
    specs = []
    if price_vnd and price_vnd > 0:
        price_note = ""
        max_p = constraints.get("max_price")
        if is_relaxed and max_p and price_vnd > max_p:
            diff = price_vnd - max_p
            price_note = f" *(Vượt ngân sách {diff:,.0f} VNĐ)*"
        specs.append(f"- **Mức giá tham khảo:** {price_vnd:,.0f} VNĐ{price_note}")
    if cpu or gpu:
        specs.append(f"- **Cấu hình:** CPU {cpu} | GPU {gpu}")
    if screen:
        specs.append(f"- **Màn hình hiển thị:** {screen}")
    if brand_name or year:
        brand_str = brand_name
        if year:
            brand_str += f" (Năm ra mắt: {year})"
        specs.append(f"- **Thương hiệu:** {brand_str}")
    if suitable_roles:
        specs.append(f"- **Phù hợp tốt nhất cho:** {suitable_roles}")

    specs.append(f"- **Điểm tương thích & tối ưu:** {score_display}")

    specs_str = "\n".join(specs)

    # Media links
    media_info = ""
    if laptop_details:
        v_url = laptop_details.get("review_video_url")
        i_url = laptop_details.get("image_search_url")
        if v_url:
            media_info += f"\n\n- **Video đánh giá thực tế:** [Xem trên YouTube]({v_url})"
        if i_url:
            media_info += f"\n- **Hình ảnh sản phẩm:** [Xem bộ sưu tập ảnh]({i_url})"

    footer = "\n\n*Bạn thấy mẫu máy này thế nào? Nếu muốn đổi tầm giá, hãng máy hoặc cấu hình khác, hãy nói cho mình biết nhé!*"

    return f"{header}\n\n{specs_str}{media_info}{footer}"


# ---------- Endpoint ----------

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Nhận tin nhắn từ user, phân loại ý định (Intent) và phản hồi tự nhiên."""

    # 1. Lấy hoặc tạo session_id
    session_id = req.session_id or str(uuid.uuid4())

    # 2. Lấy trạng thái session hiện tại
    state = session_manager.get_state(session_id)
    old_constraints = state.get("constraints", {})
    last_laptop_details = state.get("last_laptop_details")

    # 3. Trích xuất ràng buộc từ câu hỏi
    new_delta = nl2constraint.parse(req.message)
    has_new_constraints = any(
        v is not None and v != [] and v
        for v in new_delta.values()
    )

    # 4. Xác định ý định người dùng (Intent Detection)
    intent = nl2constraint.detect_intent(req.message, has_extracted_constraints=has_new_constraints)

    # XỬ LÝ THEO Ý ĐỊNH:
    # A. Ý định RESET / Xóa bộ lọc
    if intent == "RESET":
        session_manager.clear_session(session_id)
        reply = (
            "Đã làm mới toàn bộ bộ lọc và tiêu chí tìm kiếm! ✨\n\n"
            "Bạn đang cần tìm laptop với ngân sách khoảng bao nhiêu và phục vụ nhu cầu gì "
            "(học tập, văn phòng, đồ họa, lập trình hay gaming...)?"
        )
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            constraints={},
            result=None,
            laptop_details=None,
        )

    # B. Ý định CHÀO HỎI thuần túy (không có tiêu chí laptop cụ thể)
    if intent == "GREETING" and not has_new_constraints:
        reply = None
        try:
            from app.ai.gemini_service import generate_conversational_reply
            reply = generate_conversational_reply(
                user_message=req.message,
                current_constraints=old_constraints,
                last_laptop_details=last_laptop_details,
            )
        except Exception:
            reply = None

        if not reply:
            reply = (
                "Chào bạn! 👋 Mình là trợ lý AI chuyên tư vấn lựa chọn laptop tối ưu.\n\n"
                "Để mình giúp bạn chọn được chiếc laptop ưng ý nhất, bạn có thể chia sẻ thêm:\n"
                "- 💰 **Ngân sách dự kiến:** (VD: 15-20 triệu, dưới 25tr...)\n"
                "- 🎯 **Nhu cầu chính:** (VD: Học tập/văn phòng, Lập trình IT, Đồ họa thiết kế, hay Gaming...)\n"
                "- ⚡ **Yêu cầu mong muốn:** (VD: Pin trâu, mỏng nhẹ, có card đồ họa rời RTX...)"
            )

        return ChatResponse(
            session_id=session_id,
            reply=reply,
            constraints=old_constraints,
            result=None,
            laptop_details=None,
        )

    # C. Ý định TRÒ CHUYỆN / HỎI NGOÀI LỀ / GÓP Ý (nhưng không bổ sung tiêu chí tìm máy)
    if intent == "SMALLTALK" and not has_new_constraints:
        reply = None
        try:
            from app.ai.gemini_service import generate_conversational_reply
            reply = generate_conversational_reply(
                user_message=req.message,
                current_constraints=old_constraints,
                last_laptop_details=last_laptop_details,
            )
        except Exception:
            reply = None

        if not reply:
            reply = (
                "Cảm ơn chia sẻ của bạn! 😊 Nếu bạn cần tìm mẫu laptop phù hợp hoặc muốn điều chỉnh tiêu chí "
                "(ngân sách, cấu hình, hãng máy), bạn cứ nhắn cho mình nhé!"
            )

        return ChatResponse(
            session_id=session_id,
            reply=reply,
            constraints=old_constraints,
            result=None,
            laptop_details=None,
        )

    # D. Ý định TÌM KIẾM / LỌC LAPTOP (SEARCH)
    merged_constraints = session_manager.merge_constraints(old_constraints, new_delta)

    has_any_constraint = any(
        v is not None and v != [] and v
        for v in merged_constraints.values()
    )

    result = None
    df = None
    laptop_details = None

    if has_any_constraint:
        try:
            df = _load_scored_df()
            result = solver.solve(merged_constraints, df, backend="gurobi")
            laptop_details = _get_laptop_details(df, result.get("laptop_id"))
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi solver: {e}")

    # Lưu lại session
    session_manager.save_state(session_id, {
        "constraints": merged_constraints,
        "last_laptop_details": laptop_details or last_laptop_details,
    })

    # Xây dựng câu trả lời tư vấn chuyên sâu
    reply = None
    if result is not None:
        try:
            from app.ai.gemini_service import generate_gemini_consultation
            reply = generate_gemini_consultation(
                user_message=req.message,
                constraints=merged_constraints,
                result=result,
                laptop_details=laptop_details,
            )
        except Exception:
            reply = None

    if reply is None:
        reply = _build_reply(merged_constraints, result, laptop_details)

    # Lưu lịch sử chat vào Google Firebase Firestore (không làm gián đoạn phản hồi nếu gặp lỗi)
    try:
        from app.db.firebase_service import save_chat_message
        save_chat_message(
            session_id=session_id,
            user_message=req.message,
            assistant_reply=reply,
            constraints=merged_constraints,
            laptop_details=laptop_details,
            result=result,
        )
    except Exception:
        pass

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        constraints=merged_constraints,
        result=result,
        laptop_details=laptop_details,
    )


@router.delete("/{session_id}")
def clear_session(session_id: str):
    """Xóa/reset lịch sử hội thoại của một session."""
    session_manager.clear_session(session_id)
    return {"message": f"Session '{session_id}' đã được reset."}
