FROM python:3.11-slim

WORKDIR /app

# Cài đặt thư viện hệ thống cần thiết cho LightGBM, C/C++, curl và Nginx (Reverse Proxy cho Cloud Run)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    nginx \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt dependencies Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn backend, frontend, dataset và cấu hình
COPY . .

# Cấp quyền thực thi cho script start.sh
RUN chmod +x ./start.sh

# Cấu hình biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend:/app \
    BACKEND_URL=http://127.0.0.1:8000 \
    PORT=8080
# Thêm dòng này vào ngay bên dưới phần ENV trong Dockerfile
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/firebase_key.json
# Mở cổng Cloud Run (8080)
EXPOSE 8080

# Khởi chạy toàn bộ hệ thống (FastAPI + Streamlit + Nginx)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
