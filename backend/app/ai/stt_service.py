"""
Module tích hợp Google Cloud Speech-to-Text (STT) API.
Chuyển đổi âm thanh giọng nói (WebM, WAV, MP3, OGG, FLAC) thành văn bản tiếng Việt.

Sử dụng Google Cloud REST API v1:
POST https://speech.googleapis.com/v1/speech:recognize?key={API_KEY}
"""

import base64
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
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


def _get_stt_api_key() -> Optional[str]:
    """Tìm API Key phù hợp cho Speech-to-Text từ biến môi trường."""
    _reload_env()

    key = (
        os.getenv("GOOGLE_STT_API_KEY")
        or os.getenv("GOOGLE_CLOUD_API_KEY")
        or os.getenv("GOOGLE_SPEECH_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if key and key.strip() and not key.strip().startswith("your_"):
        return key.strip()
    return None


def speech_to_text(
    audio_bytes: bytes,
    mime_type: str = "audio/webm",
    lang: str = "vi-VN"
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Chuyển đổi dữ liệu âm thanh (bytes) sang văn bản tiếng Việt.

    Args:
        audio_bytes: Dữ liệu nhị phân file âm thanh (WebM, MP3, WAV, OGG,...)
        mime_type: MIME type của audio (ví dụ 'audio/webm', 'audio/wav', 'audio/mp3', 'audio/ogg')
        lang: Mã ngôn ngữ BCP-47 (mặc định 'vi-VN')

    Returns:
        (transcript, confidence, error_message):
            - transcript: Chuỗi văn bản nhận dạng được
            - confidence: Độ chính xác (0.0 -> 1.0)
            - error_message: Thông báo lỗi nếu thất bại
    """
    if not audio_bytes:
        return None, None, "Dữ liệu âm thanh rỗng."

    api_key = _get_stt_api_key()
    if not api_key:
        return None, None, "Chưa cấu hình GOOGLE_CLOUD_API_KEY hoặc GOOGLE_STT_API_KEY trong file .env."

    url = f"https://speech.googleapis.com/v1/speech:recognize?key={api_key}"

    # Cấu hình encoding dựa trên MIME type
    config: Dict[str, Any] = {
        "languageCode": lang,
        "enableAutomaticPunctuation": True,
        "model": "default"
    }

    mime_lower = mime_type.lower() if mime_type else ""
    if "webm" in mime_lower:
        config["encoding"] = "WEBM_OPUS"
        config["sampleRateHertz"] = 48000
    elif "ogg" in mime_lower:
        config["encoding"] = "OGG_OPUS"
        config["sampleRateHertz"] = 48000
    elif "wav" in mime_lower or "wave" in mime_lower:
        config["encoding"] = "LINEAR16"
    elif "mp3" in mime_lower or "mpeg" in mime_lower:
        config["encoding"] = "MP3"
    elif "flac" in mime_lower:
        config["encoding"] = "FLAC"

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "config": config,
        "audio": {
            "content": audio_b64
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=20)

        # Nếu gửi config encoding không khớp, thử lại với config tối giản
        if resp.status_code != 200 and "encoding" in config:
            fallback_payload = {
                "config": {
                    "languageCode": lang,
                    "enableAutomaticPunctuation": True
                },
                "audio": {
                    "content": audio_b64
                }
            }
            resp_fallback = requests.post(url, json=fallback_payload, timeout=20)
            if resp_fallback.status_code == 200:
                resp = resp_fallback

        if resp.status_code != 200:
            err_data = resp.json().get("error", {}) if resp.headers.get("content-type", "").startswith("application/json") else {}
            err_msg = err_data.get("message", f"HTTP {resp.status_code}: {resp.text}")
            print(f"[Google Cloud STT Error] {err_msg}")
            return None, None, f"Lỗi Google Cloud Speech-to-Text: {err_msg}"

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "", 0.0, None

        transcripts = []
        confidences = []

        for r in results:
            alts = r.get("alternatives", [])
            if alts:
                best_alt = alts[0]
                transcripts.append(best_alt.get("transcript", "").strip())
                if "confidence" in best_alt:
                    confidences.append(float(best_alt["confidence"]))

        final_transcript = " ".join(transcripts).strip()
        avg_confidence = (sum(confidences) / len(confidences)) if confidences else 1.0

        return final_transcript, avg_confidence, None

    except Exception as e:
        print(f"[Google Cloud STT Exception] {e}")
        return None, None, f"Lỗi kết nối Speech-to-Text: {str(e)}"
