"""
Tầng 0 - Session Context Manager.
Quản lý trạng thái ngữ cảnh hội thoại đa lượt (Multi-turn Context).
"""

from typing import Optional

_STORE: dict = {}  # session_id -> {"constraints": {...}, "seed_laptop_id": ..., ...}


def get_state(session_id: str) -> dict:
    """Lấy trạng thái đã lưu của session, hoặc rỗng nếu là lượt chat đầu tiên."""
    return _STORE.get(session_id, {"constraints": {}})


def save_state(session_id: str, state: dict) -> None:
    """Lưu lại trạng thái mới nhất sau khi đã merge, dùng cho lượt chat kế tiếp."""
    _STORE[session_id] = state


def merge_constraints(old_constraints: dict, new_delta: dict) -> dict:
    """
    Trộn ràng buộc mới (delta từ NLP hoặc UI) vào ràng buộc cũ đang lưu trong session.

    Quy tắc:
    - value == "REMOVE"  -> xóa field đó khỏi trạng thái
    - value is not None  -> ghi đè lên giá trị cũ
    - value is None      -> giữ nguyên giá trị cũ
    - Nếu khách chuyển đổi nhu cầu chính (ví dụ từ gaming sang văn phòng) mà câu mới không nhắc đến GPU rời / từ khóa GPU cũ,
      tự động làm sạch các ràng buộc GPU cũ để tránh xung đột tiêu chí.
    """
    merged = old_constraints.copy()

    # Kiểm tra nếu đổi nhu cầu chính
    new_tags = new_delta.get("required_tags")
    old_tags = old_constraints.get("required_tags")
    if new_tags and old_tags and set(new_tags) != set(old_tags):
        # Nếu chuyển sang văn phòng / học tập thuần túy và lượt này không yêu cầu GPU cụ thể
        if "is_office_friendly" in new_tags and "is_gaming_friendly" not in new_tags:
            if new_delta.get("gpu_keyword") is None:
                merged.pop("gpu_keyword", None)
            if new_delta.get("require_discrete_gpu") is None:
                merged.pop("require_discrete_gpu", None)

    for key, value in new_delta.items():
        if value == "REMOVE":
            merged.pop(key, None)
        elif value is not None:
            merged[key] = value

    return merged


def clear_session(session_id: str) -> None:
    """Xóa hẳn session - dùng khi người dùng bấm 'làm lại từ đầu'."""
    _STORE.pop(session_id, None)
