"""
Entry point cho FastAPI backend - Laptop Recommendation Chatbot.

Khởi chạy:
    cd backend
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, constraints

app = FastAPI(
    title="Laptop Recommendation Chatbot API",
    description="Tư vấn laptop thông minh sử dụng NLP + LightGBM + Gurobi/PuLP",
    version="1.0.0",
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


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Laptop Recommendation Chatbot API đang chạy."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
