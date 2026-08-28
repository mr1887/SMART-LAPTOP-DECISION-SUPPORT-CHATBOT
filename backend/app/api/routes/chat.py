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
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.nlp import nl2constraint
from app.optimizer import solver
from app.session import session_manager

router = APIRouter()

# Đường dẫn dataset đã có AI_Score (được tạo bởi train_lgbm.py)
_SCORED_CSV = Path(__file__).parents[4] / "data" / "processed" / "laptop_dataset_scored.csv"

_df_cache: pd.DataFrame | None = None


def _load_scored_df() -> pd.DataFrame:
    """Lazy-load và cache dataset laptop đã chấm điểm."""
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    if not _SCORED_CSV.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {_SCORED_CSV}. "
            "Hãy chạy `python backend/app/scoring/train_lgbm.py` trước."
        )
    _df_cache = pd.read_csv(_SCORED_CSV)
    return _df_cache


# ---------- Schema ----------

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    constraints: dict
    result: dict | None = None


# ---------- Helper ----------

def _build_reply(constraints: dict, result: dict | None) -> str:
    """Tạo câu trả lời tự nhiên từ kết quả solver."""
    if result is None:
        return (
            "Xin lỗi, tôi chưa đủ thông tin để gợi ý laptop. "
            "Bạn có thể cho biết ngân sách và nhu cầu sử dụng không?"
        )

    laptop_id = result.get("laptop_id")
    explanation = result.get("explanation", "")
    is_relaxed = result.get("is_relaxed", False)
    is_feasible = result.get("is_feasible", False)

    if not is_feasible and not is_relaxed and laptop_id is None:
        return f"Tôi không tìm được laptop nào phù hợp. {explanation}"

    if is_relaxed:
        return (
            f"Không có laptop nào thỏa mãn 100% yêu cầu của bạn. "
            f"Gợi ý gần nhất: Laptop ID #{laptop_id}. {explanation}"
        )

    return f"Tôi tìm được laptop phù hợp nhất: Laptop ID #{laptop_id}. {explanation}"


# ---------- Endpoint ----------

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Nhận tin nhắn từ user, trả về gợi ý laptop."""

    # 1. Lấy hoặc tạo session_id
    session_id = req.session_id or str(uuid.uuid4())

    # 2. Lấy trạng thái session hiện tại
    state = session_manager.get_state(session_id)
    old_constraints = state.get("constraints", {})

    # 3. Trích xuất ràng buộc mới từ câu hỏi
    new_delta = nl2constraint.parse(req.message)

    # 4. Merge ràng buộc
    merged_constraints = session_manager.merge_constraints(old_constraints, new_delta)

    # 5. Lưu lại session
    session_manager.save_state(session_id, {"constraints": merged_constraints})

    # 6. Kiểm tra có đủ ràng buộc để chạy solver chưa
    has_any_constraint = any(
        v is not None and v != [] and v
        for v in merged_constraints.values()
    )

    result = None
    if has_any_constraint:
        try:
            df = _load_scored_df()
            result = solver.solve(merged_constraints, df, backend="gurobi")
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi solver: {e}")

    # 7. Xây dựng câu trả lời
    reply = _build_reply(merged_constraints, result)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        constraints=merged_constraints,
        result=result,
    )


@router.delete("/{session_id}")
def clear_session(session_id: str):
    """Xóa/reset lịch sử hội thoại của một session."""
    session_manager.clear_session(session_id)
    return {"message": f"Session '{session_id}' đã được reset."}
