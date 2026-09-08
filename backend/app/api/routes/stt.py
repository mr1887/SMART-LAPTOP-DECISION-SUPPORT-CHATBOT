"""
Route /api/stt - Endpoint nhận diện giọng nói tiếng Việt bằng Google Cloud Speech-to-Text.
"""

import base64
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.ai.stt_service import speech_to_text

router = APIRouter()


class STTJsonRequest(BaseModel):
    audio_base64: str
    mime_type: Optional[str] = "audio/webm"
    lang: Optional[str] = "vi-VN"


@router.post("")
async def transcribe_audio_file(
    file: Optional[UploadFile] = File(None),
    lang: str = Form("vi-VN")
):
    """
    Nhận file âm thanh qua FormData (WebM, WAV, MP3, OGG) và trả về văn bản nhận diện.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="Chưa có file âm thanh được tải lên.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File âm thanh rỗng.")

    mime_type = file.content_type or "audio/webm"
    transcript, confidence, err = speech_to_text(audio_bytes, mime_type=mime_type, lang=lang)

    if err:
        raise HTTPException(status_code=500, detail=err)

    return {
        "transcript": transcript or "",
        "confidence": confidence or 0.0,
        "language": lang
    }


@router.post("/base64")
def transcribe_audio_base64(req: STTJsonRequest):
    """
    Nhận chuỗi âm thanh Base64 và trả về văn bản nhận diện tiếng Việt.
    """
    if not req.audio_base64 or not req.audio_base64.strip():
        raise HTTPException(status_code=400, detail="audio_base64 không được để rỗng.")

    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Chuỗi audio_base64 không hợp lệ.")

    transcript, confidence, err = speech_to_text(
        audio_bytes,
        mime_type=req.mime_type or "audio/webm",
        lang=req.lang or "vi-VN"
    )

    if err:
        raise HTTPException(status_code=500, detail=err)

    return {
        "transcript": transcript or "",
        "confidence": confidence or 0.0,
        "language": req.lang or "vi-VN"
    }
