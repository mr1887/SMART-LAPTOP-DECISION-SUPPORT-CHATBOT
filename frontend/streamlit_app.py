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


def _call_backend_bytes(endpoint: str, payload: dict) -> bytes | None:
    try:
        resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.content
    except Exception:
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
    explanation = result.get("explanation", "")
    score = result.get("ai_score")
    score_str = f" ({score * 10:.1f}/10 ⭐)" if score else ""

    if result.get("is_feasible"):
        return f"✨ **Gợi ý lựa chọn tối ưu nhất{score_str}:**\n\n{explanation}"
    elif result.get("laptop_id"):
        return f"💡 **Gợi ý lựa chọn cân bằng nhất{score_str}:**\n\n{explanation}"
    else:
        return f"⚠️ **Thông báo:** {explanation or 'Không tìm thấy laptop phù hợp với tiêu chí hiện tại.'}"


def _format_constraints_summary(c: dict) -> str:
    """Chuyển đổi dict ràng buộc kỹ thuật sang câu tóm tắt dễ hiểu cho người dùng."""
    parts = []
    min_p = c.get("min_price")
    max_p = c.get("max_price")
    if min_p and max_p:
        parts.append(f"Ngân sách {min_p/1e6:.0f}tr - {max_p/1e6:.0f}tr")
    elif max_p:
        parts.append(f"Ngân sách dưới {max_p/1e6:.0f}tr")
    elif min_p:
        parts.append(f"Ngân sách trên {min_p/1e6:.0f}tr")

    if c.get("max_weight"):
        parts.append(f"Cân nặng ≤ {c['max_weight']}kg")

    if c.get("min_battery"):
        parts.append(f"Pin ≥ {c['min_battery']/60:.1f}h")

    if c.get("require_discrete_gpu") is True:
        parts.append("Có card đồ họa rời")
    elif c.get("require_discrete_gpu") is False:
        parts.append("Card onboard")

    if c.get("gpu_keyword"):
        parts.append(f"GPU {c['gpu_keyword']}")

    tags = c.get("required_tags") or []
    tag_map = {
        "is_programming_friendly": "IT / Code",
        "is_graphic_friendly": "Đồ họa",
        "is_office_friendly": "Văn phòng",
        "is_gaming_friendly": "Gaming",
    }
    tag_names = [tag_map.get(t, t) for t in tags if t in tag_map]
    if tag_names:
        parts.append(f"Nhu cầu: {', '.join(tag_names)}")

    if not parts:
        return ""
    return "📊 **Tiêu chí đang áp dụng:** " + " • ".join(parts)


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

    st.subheader("Card đồ họa (GPU)")
    gpu_type_option = st.selectbox(
        "Loại Card đồ họa",
        options=["Tùy chọn (Tất cả)", "Bắt buộc Card rời (Discrete GPU - Gaming/Đồ họa)", "Chỉ Card tích hợp (Onboard - Tiết kiệm pin)"],
        index=0,
    )
    require_discrete_gpu = None
    if "Bắt buộc Card rời" in gpu_type_option:
        require_discrete_gpu = True
    elif "Chỉ Card tích hợp" in gpu_type_option:
        require_discrete_gpu = False

    gpu_keyword_input = st.text_input(
        "Dòng GPU cụ thể (tùy chọn)",
        placeholder="VD: RTX 4060, RTX 4050, NVIDIA, Apple M4...",
    ).strip()

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
            "require_discrete_gpu": require_discrete_gpu,
            "gpu_keyword": gpu_keyword_input if gpu_keyword_input else None,
            "required_tags": selected_tags,
        }
        data = _call_backend("/api/constraints", payload)
        if data:
            gpu_info = ""
            if require_discrete_gpu is True:
                gpu_info = ", Card rời"
            elif require_discrete_gpu is False:
                gpu_info = ", Card tích hợp"
            if gpu_keyword_input:
                gpu_info += f", GPU '{gpu_keyword_input}'"

            summary_text = (
                f"[Bộ lọc] Ngân sách {price_range[0]}-{price_range[1]} triệu{gpu_info}"
                + (f", nhu cầu: {', '.join(selected_labels)}" if selected_labels else "")
            )
            _append_result_to_chat(summary_text, data)
            st.rerun()

    st.divider()
    enable_voice = st.checkbox("🔊 Tự động phát giọng nói (gTTS)", value=False)

    if st.session_state.current_constraints:
        st.caption("Ràng buộc hiện tại (đồng bộ với chat):")
        st.json(st.session_state.current_constraints, expanded=False)


# ---------- MAIN: chat tự nhiên để tinh chỉnh thêm ----------
st.title("💻 Trợ lý AI Tư vấn Chọn Laptop")
st.caption("Trò chuyện tự nhiên hoặc dùng bộ lọc bên trái để tìm kiếm chiếc laptop tối ưu nhất cho bạn.")

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("VD: 'Mình cần tìm laptop gaming tầm 25tr có card RTX 4050' hoặc 'Chào bạn'")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Gemini AI & Solver đang phân tích..."):
            data = _call_backend("/api/chat", {
                "session_id": st.session_state.session_id,
                "message": user_input,
            })
            if data:
                reply = _build_reply(data)
                st.markdown(reply)

                # Hiển thị card media trực quan nếu có gợi ý laptop
                laptop_details = data.get("laptop_details")
                if laptop_details:
                    col1, col2 = st.columns(2)
                    v_url = laptop_details.get("review_video_url")
                    i_url = laptop_details.get("image_search_url")
                    if v_url:
                        with col1:
                            st.link_button("📺 Xem Đánh Giá Trên YouTube", v_url, use_container_width=True)
                    if i_url:
                        with col2:
                            st.link_button("🖼️ Xem Hình Ảnh Chi Tiết", i_url, use_container_width=True)

                # Tự động phát giọng nói gTTS hoặc nút nghe lại
                if reply:
                    audio_bytes = _call_backend_bytes("/api/tts", {"text": reply, "lang": "vi"})
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=enable_voice)

                c_summary = _format_constraints_summary(data.get('constraints', {}))
                if c_summary:
                    st.caption(c_summary)
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", reply))
                st.session_state.current_constraints = data.get("constraints", {})

