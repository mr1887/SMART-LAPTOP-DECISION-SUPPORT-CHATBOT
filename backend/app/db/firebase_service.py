"""
Module quản lý kết nối và lưu trữ dữ liệu với Google Firebase Firestore.
Hỗ trợ file chứng thực firebase_key.json và Application Default Credentials (ADC) trên Cloud Run.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

_db_client: Optional[Any] = None


def _find_key_path() -> Optional[Path]:
    """Tìm đường dẫn tới file chứng thực firebase_key.json."""
    env_cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if env_cred and Path(env_cred).exists():
        return Path(env_cred)

    candidates = [
        Path.cwd() / "firebase_key.json",
        Path(__file__).parents[3] / "firebase_key.json",
        Path(__file__).parents[2] / "firebase_key.json",
        Path("/app/firebase_key.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def init_firebase() -> Optional[Any]:
    """Khởi tạo Firebase Admin SDK và trả về client Firestore."""
    global _db_client
    if _db_client is not None:
        return _db_client

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            key_path = _find_key_path()
            if key_path and key_path.exists():
                cred = credentials.Certificate(str(key_path))
                firebase_admin.initialize_app(cred)
            else:
                # Tự động dùng Default Credentials của Google Cloud Run
                firebase_admin.initialize_app()

        _db_client = firestore.client()
        return _db_client
    except Exception as e:
        logger.warning(f"Không thể khởi tạo Firebase Firestore: {e}")
        return None


def save_chat_message(
    session_id: str,
    user_message: str,
    assistant_reply: str,
    constraints: Optional[Dict[str, Any]] = None,
    laptop_details: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> bool:
    """Lưu vết một lượt trao đổi chat vào Firestore."""
    try:
        db = init_firebase()
        if db is None:
            return False

        now_utc = datetime.now(timezone.utc)
        doc_data = {
            "session_id": session_id,
            "user_message": user_message,
            "assistant_reply": assistant_reply,
            "constraints": constraints or {},
            "laptop_details": laptop_details or {},
            "is_feasible": result.get("is_feasible") if result else None,
            "is_relaxed": result.get("is_relaxed") if result else None,
            "timestamp": now_utc,
            "created_at_iso": now_utc.isoformat(),
        }

        db.collection("sessions").document(session_id).collection("messages").add(doc_data)
        db.collection("chat_logs").add(doc_data)
        return True
    except Exception as e:
        logger.warning(f"Lỗi khi lưu Firestore: {e}")
        return False


def get_session_history(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Lấy lịch sử hội thoại của một session từ Firestore."""
    try:
        db = init_firebase()
        if db is None:
            return []
        from firebase_admin import firestore

        query = (
            db.collection("sessions")
            .document(session_id)
            .collection("messages")
            .order_by("timestamp", direction=firestore.Query.ASCENDING)
            .limit(limit)
        )
        docs = query.stream()
        history = []
        for doc in docs:
            history.append(doc.to_dict())
        return history
    except Exception as e:
        logger.warning(f"Lỗi khi đọc Firestore: {e}")
        return []
