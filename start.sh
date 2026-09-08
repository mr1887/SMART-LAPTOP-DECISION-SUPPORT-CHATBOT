#!/bin/bash
set -e

export PORT=${PORT:-8080}
echo "==> Khởi động ứng dụng trên PORT: $PORT"

# 1. Tạo file cấu hình Nginx từ template với biến $PORT động
envsubst '$PORT' < /app/nginx.conf.template > /etc/nginx/nginx.conf

# 2. Khởi động FastAPI Backend (port 8000)
echo "==> [1/3] Đang khởi động Backend FastAPI trên cổng 8000..."
cd /app/backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# 3. Khởi động Streamlit Frontend (port 8501)
echo "==> [2/3] Đang khởi động Frontend Streamlit trên cổng 8501..."
cd /app
streamlit run frontend/streamlit_app.py --server.port=8501 --server.address=127.0.0.1 --server.enableCORS=false --server.enableXsrfProtection=false &
FRONTEND_PID=$!

# 4. Khởi động Nginx Reverse Proxy (cổng $PORT)
echo "==> [3/3] Đang khởi động Nginx Reverse Proxy..."
nginx -g 'daemon off;' &
NGINX_PID=$!

# Bắt tín hiệu dừng (SIGTERM/SIGINT) để dừng tất cả tiến trình an toàn khi Cloud Run scale down
trap "echo 'Đang dừng các tiến trình...'; kill -TERM $BACKEND_PID $FRONTEND_PID $NGINX_PID 2>/dev/null" SIGTERM SIGINT

# Chờ bất kỳ tiến trình nào dừng
wait -n $BACKEND_PID $FRONTEND_PID $NGINX_PID

exit $?
