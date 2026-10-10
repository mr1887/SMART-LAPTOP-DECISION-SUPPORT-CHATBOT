"""
Entry point cho FastAPI backend - Laptop Recommendation Chatbot.

Khởi chạy:
    cd backend
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, constraints, tts, stt

APP_BUILD_ID = "dataset-aware-2026-10-10-v1"

app = FastAPI(
    title="Laptop Recommendation Chatbot API",
    description="Tư vấn laptop thông minh sử dụng NLP + LightGBM + Google OR-Tools CP-SAT + Gemini Search Grounding + Firestore + Google Cloud STT & TTS",
    version=APP_BUILD_ID,
)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Đổi lại thành domain frontend khi deploy production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Routers ----------
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(constraints.router, prefix="/api/constraints", tags=["Constraints"])
app.include_router(tts.router, prefix="/api/tts", tags=["TTS"])
app.include_router(stt.router, prefix="/api/stt", tags=["STT"])

# ---------- Static Web App ----------
_WEB_DIR = Path(__file__).parents[2] / "frontend" / "web"

@app.on_event("startup")
def _print_build_id():
    print(f">>> APP BUILD: {APP_BUILD_ID}", flush=True)

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "build": APP_BUILD_ID}

if _WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="static_web")
