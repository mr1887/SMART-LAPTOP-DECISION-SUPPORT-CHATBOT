"""
Module Google Cloud Text-to-Speech (TTS) - Chuyển đổi văn bản câu trả lời sang giọng nói tiếng Việt chất lượng cao.
Sử dụng Google Cloud REST API v1:
POST https://texttospeech.googleapis.com/v1/text:synthesize?key={API_KEY}
"""

import base64
import os
import re
from pathlib import Path
from typing import Optional, Tuple
import requests
from dotenv import load_dotenv


def _reload_env():
    """Tự động tìm và nạp file .env từ các thư mục tiềm năng."""
    possible_paths = [
        Path(__file__).parents[3] / ".env",
        Path(__file__).parents[2] / ".env",
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
    ]
    for p in possible_paths:
        if p.exists():
            load_dotenv(dotenv_path=p, override=True)
            return


# Nạp khi khởi động
_reload_env()


def _get_tts_api_key() -> Optional[str]:
    """Tìm API Key phù hợp cho Text-to-Speech từ biến môi trường."""
    _reload_env()

    key = (
        os.getenv("GOOGLE_TTS_API_KEY")
        or os.getenv("GOOGLE_CLOUD_API_KEY")
        or os.getenv("GOOGLE_SPEECH_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if key and key.strip() and not key.strip().startswith("your_"):
        return key.strip()
    return None


def clean_text_for_speech(text: str) -> str:
    """Loại bỏ ký tự Markdown, đường dẫn link, icon emoji để đọc giọng nói tự nhiên nhất."""
    if not text:
        return ""

    clean = text
    # 1. Bỏ markdown links [text](url) -> giữ lại text
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)

    # 2. Bỏ URLs
    clean = re.sub(r"https?://\S+", "", clean)

    # 3. Bỏ markdown headers (###, ##, #), bold (**), italic (*), bullets (- )
    clean = re.sub(r"#+\s*", "", clean)
    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
    clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
    clean = re.sub(r"`([^`]+)`", r"\1", clean)
    clean = re.sub(r"^[\s\*\-\+]+\s*", "", clean, flags=re.MULTILINE)

    # 4. Bỏ hầu hết các emoji
    clean = re.sub(r"[\U00010000-\U0010ffff]", "", clean)
    clean = re.sub(r"[\u2600-\u27BF]", "", clean)

    # 5. Chuẩn hóa khoảng trắng dư thừa
    clean = re.sub(r"\n+", ". ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    return clean


def text_to_speech_bytes(
    text: str,
    lang: str = "vi-VN",
    voice_name: Optional[str] = "vi-VN-Neural2-A",
    speaking_rate: float = 1.0,
    pitch: float = 0.0
) -> Optional[bytes]:
    """
    Tạo file âm thanh MP3 (Bytes) từ chuỗi văn bản sử dụng Google Cloud Text-to-Speech API.

    Args:
        text: Chuỗi văn bản cần đọc
        lang: Mã ngôn ngữ (mặc định 'vi-VN')
        voice_name: Tên giọng đọc (VD: 'vi-VN-Neural2-A', 'vi-VN-Wavenet-A', 'vi-VN-Standard-A')
        speaking_rate: Tốc độ đọc (0.25 -> 4.0, mặc định 1.0)
        pitch: Độ cao giọng (-20.0 -> 20.0, mặc định 0.0)

    Returns:
        Dữ liệu nhị phân MP3 (bytes) hoặc None nếu có lỗi.
    """
    spoken_text = clean_text_for_speech(text)
    if not spoken_text:
        return None

    # Giới hạn an toàn khoảng 4000 ký tự cho 1 request
    if len(spoken_text) > 4000:
        spoken_text = spoken_text[:4000] + "..."

    api_key = _get_tts_api_key()
    if not api_key:
        print("[Google Cloud TTS] Chưa cấu hình GOOGLE_CLOUD_API_KEY hoặc GOOGLE_TTS_API_KEY trong file .env.")
        return None

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"

    # Chuẩn hóa mã ngôn ngữ
    lang_code = lang if "-" in lang else f"{lang}-VN" if lang == "vi" else lang

    voice_config = {
        "languageCode": lang_code
    }
    if voice_name:
        voice_config["name"] = voice_name

    payload = {
        "input": {
            "text": spoken_text
        },
        "voice": voice_config,
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": speaking_rate,
            "pitch": pitch
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=20)

        # Nếu lỗi do giọng Neural2 không khả dụng, fallback về giọng mặc định theo languageCode
        if resp.status_code != 200 and voice_name:
            fallback_payload = {
                "input": {
                    "text": spoken_text
                },
                "voice": {
                    "languageCode": lang_code
                },
                "audioConfig": {
                    "audioEncoding": "MP3",
                    "speakingRate": speaking_rate,
                    "pitch": pitch
                }
            }
            resp_fallback = requests.post(url, json=fallback_payload, timeout=20)
            if resp_fallback.status_code == 200:
                resp = resp_fallback

        if resp.status_code != 200:
            err_data = resp.json().get("error", {}) if resp.headers.get("content-type", "").startswith("application/json") else {}
            err_msg = err_data.get("message", f"HTTP {resp.status_code}: {resp.text}")
            print(f"[Google Cloud TTS Error] {err_msg}")
            return None

        data = resp.json()
        audio_content_b64 = data.get("audioContent")
        if not audio_content_b64:
            return None

        return base64.b64decode(audio_content_b64)

    except Exception as e:
        print(f"[Google Cloud TTS Exception] {e}")
        return None
