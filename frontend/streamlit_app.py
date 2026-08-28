"""
Frontend - Streamlit UI, hướng HYBRID:
- Sidebar: slider/checkbox cho ràng buộc cứng (giá, cân nặng, pin, ngành)
  -> gửi thẳng /api/constraints, KHÔNG qua Regex, luôn chính xác 100%.
- Ô chat chính giữa: vẫn giữ để khách tinh chỉnh bằng câu tự nhiên
  ("đổi ngân sách thành 25tr", "bỏ điều kiện pin"...) -> qua /api/chat,
  dùng Regex như thiết kế gốc.
Cả 2 luồng dùng chung session_id, chung session_manager ở backend nên
trạng thái luôn đồng bộ dù khách dùng cách nào.
"""
import os
import uuid

import requests
import streamlit as st

# Dev local: http://localhost:8000 | Docker: đặt env BACKEND_URL=http://backend:8000
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Smart Laptop Chatbot", page_icon="💻", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_constraints" not in st.session_state:
    st.session_state.current_constraints = {}


def _call_backend(endpoint: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Không kết nối được Backend ({BACKEND_URL}): {e}")
        return None


def _build_reply(data: dict) -> str:
    """Xây dựng câu trả lời từ response của backend.

    /api/chat    trả về: {session_id, reply, constraints, result}
    /api/constraints trả về: {constraints, result}
    result có dạng: {laptop_id, is_feasible, is_relaxed, ai_score, explanation}
    """
    # Ưu tiên lấy reply đã được backend format sẵn (từ /api/chat)
    if "reply" in data and data["reply"]:
        return data["reply"]

    result = data.get("result") or {}
    laptop_id = result.get("laptop_id")
    explanation = result.get("explanation", "")

    if result.get("is_feasible"):
        return f" Gợi ý: **Laptop #{laptop_id}**\n\n{explanation}"
    elif laptop_id:
        return f" Không có máy 100% phù hợp, gần nhất: **Laptop #{laptop_id}**\n\n{explanation}"
    else:
        return f" Không tìm được máy phù hợp.\n\n{explanation}"


def _append_result_to_chat(user_text: str, data: dict) -> None:
    reply = _build_reply(data)
    st.session_state.chat_history.append(("user", user_text))
    st.session_state.chat_history.append(("assistant", reply))
    st.session_state.current_constraints = data.get("constraints", {})


# ---------- SIDEBAR: bộ lọc nhanh bằng slider/checkbox ----------
with st.sidebar:
    st.header(" Bộ lọc nhanh")
    st.caption("Chỉnh ở đây luôn chính xác 100%, không cần gõ câu.")

    price_range = st.slider(
        "Khoảng ngân sách (triệu đồng)",
        min_value=5, max_value=50, value=(10, 25), step=1,
    )

    max_weight = st.slider(
        "Cân nặng tối đa (kg)",
        min_value=0.8, max_value=3.0, value=1.6, step=0.1,
    )

    min_battery_hours = st.slider(
        "Pin tối thiểu (giờ, dùng văn phòng)",
        min_value=2, max_value=15, value=6, step=1,
    )

    st.subheader("Nhu cầu sử dụng")
    tag_options = {
        "is_programming_friendly": "Lập trình / CNTT",
        "is_graphic_friendly": "Đồ họa / Thiết kế",
        "is_office_friendly": "Văn phòng / Mỏng nhẹ",
        "is_gaming_friendly": "Gaming",
    }
    selected_labels = st.multiselect(
        "Chọn nhu cầu (có thể chọn nhiều)",
        options=list(tag_options.values()),
    )
    selected_tags = [k for k, v in tag_options.items() if v in selected_labels]

    if st.button(" Tìm laptop theo bộ lọc này", use_container_width=True):
        payload = {
            "min_price": price_range[0] * 1_000_000,
            "max_price": price_range[1] * 1_000_000,
            "max_weight": max_weight,
            "min_battery": min_battery_hours * 60,
            "required_tags": selected_tags,
        }
        data = _call_backend("/api/constraints", payload)
        if data:
            summary_text = (
                f"[Bộ lọc] {price_range[0]}-{price_range[1]}tr, "
                f"nhẹ dưới {max_weight}kg, pin trên {min_battery_hours}h"
                + (f", nhu cầu: {', '.join(selected_labels)}" if selected_labels else "")
            )
            _append_result_to_chat(summary_text, data)
            st.rerun()

    st.divider()
    if st.session_state.current_constraints:
        st.caption("Ràng buộc hiện tại (đồng bộ với chat):")
        st.json(st.session_state.current_constraints, expanded=False)


# ---------- MAIN: chat tự nhiên để tinh chỉnh thêm ----------
st.title(" Trợ lý tư vấn chọn Laptop")
st.caption("Dùng bộ lọc bên trái cho nhanh, hoặc gõ tự nhiên ở đây để tinh chỉnh thêm.")

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("VD: 'đổi ngân sách thành 25tr' hoặc 'bỏ điều kiện pin'")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang phân tích..."):
            data = _call_backend("/api/chat", {
                "session_id": st.session_state.session_id,
                "message": user_input,
            })
            if data:
                reply = _build_reply(data)
                st.markdown(reply)
                st.caption(f"Ràng buộc hiện tại: {data.get('constraints', {})}")
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", reply))
                st.session_state.current_constraints = data.get("constraints", {})
