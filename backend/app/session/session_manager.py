"""
Tầng 0 - Session Context Manager.
MVP: lưu trong RAM (dict). Khi triển khai nhiều instance backend hoặc cần
bền hơn 1 phiên chạy, thay _STORE bằng Redis (đã có sẵn service redis
trong docker-compose.yml).
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
    """Trộn ràng buộc mới (delta, từ Regex hoặc từ nút bấm UI) vào ràng
    buộc cũ đang lưu trong session.

    Quy tắc:
    - value == "REMOVE"  -> xóa field đó khỏi trạng thái (khách nói "bỏ...")
    - value is not None  -> ghi đè lên giá trị cũ (khách đổi/thêm ràng buộc)
    - value is None       -> giữ nguyên giá trị cũ (field này không được
                              nhắc tới trong lượt chat hiện tại)
    """
    merged = old_constraints.copy()
    for key, value in new_delta.items():
        if value == "REMOVE":
            merged.pop(key, None)
        elif value is not None:
            merged[key] = value
    return merged


def clear_session(session_id: str) -> None:
    """Xóa hẳn session - dùng khi người dùng bấm 'làm lại từ đầu'."""
    _STORE.pop(session_id, None)
