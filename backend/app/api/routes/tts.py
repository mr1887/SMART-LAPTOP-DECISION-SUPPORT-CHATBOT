"""
Route /api/tts - Endpoint chuyển đổi văn bản câu trả lời thành âm thanh tiếng Việt sử dụng Google Cloud Text-to-Speech.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.ai.tts_service import text_to_speech_bytes

router = APIRouter()


class TTSRequest(BaseModel):
    text: str
    lang: Optional[str] = "vi-VN"
    voice_name: Optional[str] = "vi-VN-Neural2-A"
    speaking_rate: Optional[float] = 1.0
    pitch: Optional[float] = 0.0


@router.post("")
def generate_tts(req: TTSRequest):
    """Nhận văn bản tiếng Việt và trả về file âm thanh MP3 từ Google Cloud TTS."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Văn bản không được để rỗng.")

    audio_bytes = text_to_speech_bytes(
        text=req.text,
        lang=req.lang or "vi-VN",
        voice_name=req.voice_name,
        speaking_rate=req.speaking_rate or 1.0,
        pitch=req.pitch or 0.0
    )
    if not audio_bytes:
        raise HTTPException(
            status_code=500,
            detail="Không thể khởi tạo âm thanh từ Google Cloud TTS. Vui lòng kiểm tra GOOGLE_CLOUD_API_KEY trong file .env."
        )

    return Response(content=audio_bytes, media_type="audio/mpeg")
