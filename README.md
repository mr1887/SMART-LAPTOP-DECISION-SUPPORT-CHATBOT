#  Smart Laptop Chatbot

> **"Không chỉ Chat, chúng tôi giải toán để chọn Laptop cho bạn."**

Trợ lý AI tư vấn chọn laptop, kết hợp khả năng hiểu ngôn ngữ tự nhiên của **Gemini** với độ chính xác tuyệt đối của **tối ưu hóa toán học (Gurobi/PuLP)** — đảm bảo mọi gợi ý luôn tuân thủ 100% ràng buộc về ngân sách và cấu hình, thay vì "đoán" như chatbot AI thông thường.

Link dự án: https://smart-laptop-chatbot-61261939576.asia-southeast1.run.app/

---

## 📖 Mục lục

- [Vấn đề & Giải pháp](#-vấn-đề--giải-pháp)
- [Demo](#-demo)
- [Tính năng](#-tính-năng)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Tech Stack](#-tech-stack)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt & Chạy thử](#-cài-đặt--chạy-thử)
- [Biến môi trường](#-biến-môi-trường)
- [API Endpoints](#-api-endpoints)
- [Data Pipeline](#-data-pipeline)
- [Kiểm thử](#-kiểm-thử)
- [Triển khai lên Cloud Run](#-triển-khai-lên-cloud-run)
- [Giới hạn hiện tại](#-giới-hạn-hiện-tại)
- [Roadmap](#-roadmap)
- [License](#-license)

---

##  Vấn đề & Giải pháp

Chọn mua laptop là bài toán phức tạp: hàng trăm thông số kỹ thuật, ngân sách giới hạn, nhu cầu sử dụng khác nhau. Chatbot AI dựa hoàn toàn vào mô hình ngôn ngữ tiềm ẩn rủi ro **hallucination** — có thể gợi ý sản phẩm vượt ngân sách hoặc sai cấu hình yêu cầu.

**Nguyên lý thiết kế cốt lõi:**

> LLM là giao diện. Toán học tối ưu hóa là bộ não quyết định.

Hệ thống tách bạch rõ ràng: **Gemini** chỉ đảm nhận việc hiểu ý định người dùng, còn quyết định cuối cùng do **Gurobi Optimizer** đảm nhận — một bộ giải toán học đảm bảo tuân thủ tuyệt đối mọi ràng buộc, không có khái niệm "đoán sai".

---

##  Demo

- **Video demo đầy đủ:** [dán link YouTube của bạn]
- **Demo trực tiếp (Cloud Run):** liên hệ trực tiếp để nhận link — bản demo giới hạn quy mô nhỏ

---

##  Tính năng

- 🗣️ **Giao tiếp đa phương thức** — text hoặc giọng nói (Google Speech-to-Text / Text-to-Speech)
- 🎛️ **Giao diện Hybrid** — sidebar nút bấm/slider (chính xác 100%) song song với chat tự nhiên (linh hoạt)
- 🧠 **Hiểu ngôn ngữ tự nhiên tiếng Việt** — trích xuất ngân sách, cân nặng, pin, nhu cầu sử dụng
- 🔄 **Ghi nhớ ngữ cảnh hội thoại** — hợp nhất ràng buộc qua nhiều lượt chat (thêm/sửa/xóa có chủ đích), lưu bền trên Firestore
- 🎯 **Tối ưu hóa đảm bảo 100% ràng buộc** — không bao giờ gợi ý sản phẩm vi phạm yêu cầu cứng
- 🛡️ **Xử lý thông minh khi vô nghiệm (Infeasible)** — tự chẩn đoán ràng buộc xung đột (`computeIIS`), đề xuất phương án gần đúng nhất thay vì chỉ báo lỗi
- 📊 **Chấm điểm phù hợp bằng ML** — LightGBM học từ dữ liệu hành vi người dùng thực tế
- ☁️ **Serverless, tự động mở rộng** — triển khai trên Google Cloud Run

---

##  Kiến trúc hệ thống

```
[ Người dùng: Text / Giọng nói / Nút bấm-Slider ]
                    │
                    ▼
┌───────────────────────────────────────────┐
│ TẦNG 0 — SESSION CONTEXT MANAGER (Firestore)│
│ Hợp nhất ràng buộc qua nhiều lượt hội thoại  │
└──────────────────┬──────────────────────────┘
                    ▼
┌───────────────────────────────────────────┐
│ TẦNG 1 — HIỂU Ý ĐỊNH                        │
│ Gemini 2.5 Flash (+ Regex dự phòng)         │
└──────────────────┬──────────────────────────┘
                    ▼
┌───────────────────────────────────────────┐
│ TẦNG 2 — CHẤM ĐIỂM PHÙ HỢP                  │
│ LightGBM → AI_Score cho từng laptop         │
└──────────────────┬──────────────────────────┘
                    ▼
┌───────────────────────────────────────────┐
│ TẦNG 3 — TỐI ƯU HÓA QUYẾT ĐỊNH              │
│ Gurobi / PuLP → chọn laptop tối ưu           │
│ Infeasible → chẩn đoán + nới lỏng có kiểm soát│
└──────────────────┬──────────────────────────┘
                    ▼
┌───────────────────────────────────────────┐
│ TẦNG 4 — GROUNDED RESPONSE                  │
│ Gemini sinh câu trả lời bám sát kết quả      │
│ + Google Text-to-Speech                      │
└───────────────────────────────────────────┘
```

---

##  Tech Stack

| Thành phần | Công nghệ |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| AI & NLU | Google GenAI SDK, Gemini 2.5 Flash |
| Optimization | Gurobi (`gurobipy`, WLS License) + PuLP (CBC, dự phòng miễn phí) |
| ML Scoring | LightGBM |
| Voice | Google Cloud Speech-to-Text / Text-to-Speech |
| Database / Session | Firebase / Firestore |
| Frontend | Streamlit |
| DevOps | Docker, Docker Compose |
| Cloud | Google Cloud Run (asia-southeast1) |

---

##  Cấu trúc thư mục

```
smart-laptop-chatbot/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes/
│   │   │   ├── chat.py                # POST /api/chat
│   │   │   └── constraints.py         # POST /api/constraints/set
│   │   ├── session/session_manager.py # Tầng 0
│   │   ├── nlp/nl2constraint.py       # Tầng 1 (Regex dự phòng)
│   │   ├── tagging/auto_tag.py        # Auto-tagging
│   │   ├── scoring/
│   │   │   ├── train_lgbm.py          # Train model (offline)
│   │   │   ├── predict.py             # Inference (runtime)
│   │   │   └── model.pkl
│   │   ├── optimizer/solver.py        # Tầng 3
│   │   └── data_pipeline/
│   │       ├── merge_laptop.py
│   │       └── clean_dataset.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── streamlit_app.py
│   └── Dockerfile
├── data/
│   ├── raw/ external/ processed/ sql/
├── docker-compose.yml
└── .env.example
```

---

##  Cài đặt & Chạy thử

### Cách 1 — Docker (khuyến nghị)

```bash
cp .env.example .env   
docker compose up --build
```

- Frontend: `http://localhost:8501`
- Backend API docs (Swagger): `http://localhost:8000/docs`

### Cách 2 — Chạy local (dev, không dùng Docker)

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend (terminal khác):**
```bash
cd frontend
pip install streamlit requests
$env:BACKEND_URL="http://localhost:8000"    # PowerShell
streamlit run streamlit_app.py
```

---

##  Biến môi trường

Copy `.env.example` thành `.env`:

```env
GUROBI_WLSACCESSID=
GUROBI_WLSSECRET=
GUROBI_LICENSEID=
GEMINI_API_KEY=
FIREBASE_CREDENTIALS_PATH=firebase_key.json
```

> Gurobi tự động dò license file cục bộ trước (nếu đã chạy `grbgetkey`); chỉ cần khai báo biến môi trường trên nếu chạy trong môi trường không lưu được license file (VD container mới).

---

## API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/chat` | Nhận câu tự nhiên, qua Gemini/Regex → merge ràng buộc → giải → trả kết quả |
| `POST` | `/api/constraints/set` | Nhận ràng buộc có cấu trúc từ sidebar (nút bấm/slider), bỏ qua NLU |
| `GET` | `/health` | Health check |

Xem chi tiết schema request/response tại `http://localhost:8000/docs` (Swagger UI tự sinh).

---

##  Data Pipeline

```bash
cd backend
python app/data_pipeline/merge_laptop.py      # JOIN các bảng ClickHouse → laptop_dataset.csv
python app/data_pipeline/clean_dataset.py      # Coalesce, fallback, auto-tag → laptop_dataset_tagged.csv
python app/scoring/train_lgbm.py               # Train model → model.pkl + laptop_dataset_scored.csv
```

>  Trước khi chạy, kiểm tra đơn vị đo (`price` phải là VNĐ đầy đủ, `laptop_weight` phải là kg) — dự án từng gặp lỗi lệch đơn vị khiến solver luôn trả về cùng 1 kết quả bất kể ràng buộc.

---

##  Kiểm thử

```bash
cd backend
pytest tests/ -v
```

Đã kiểm thử thủ công với người dùng thật (2 người), xác nhận đúng điểm khác biệt cốt lõi: câu trả lời không "lan man" như chatbot AI thuần LLM, tốc độ phản hồi nhanh nhờ cache Gurobi Environment (Singleton Pattern).

---

##  Triển khai lên Cloud Run

```bash
gcloud run deploy smart-laptop-chatbot \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi
```

Đảm bảo đã cấu hình đúng biến môi trường (Gurobi WLS, Gemini API key, Firebase credentials) trên Cloud Run trước khi deploy.

---

##  Giới hạn hiện tại

- License Gurobi đang dùng là **WLS Academic** — chỉ hợp lệ cho mục đích học thuật, không dùng được cho production thương mại (cần chuyển sang PuLP/OR-Tools hoặc mua license thương mại nếu thương mại hóa)
- Auto-tagging và Regex hiện thiết kế riêng cho ngành hàng laptop, cần tùy chỉnh nếu áp dụng ngành hàng khác
- Dữ liệu engagement cần đủ lớn để LightGBM chấm điểm chính xác (vấn đề Cold-Start với dữ liệu mới/ít)

---

##  Roadmap

- [ ] Mở rộng sang ngành hàng khác (điện thoại, đồ gia dụng)
- [ ] Cá nhân hóa theo lịch sử người dùng
- [ ] Dashboard self-service cho doanh nghiệp vừa & nhỏ
- [ ] Chuyển hoàn toàn sang PuLP/OR-Tools cho bản production
- [ ] Hỗ trợ đa ngôn ngữ

---

##  License

Dự án phát triển cho mục đích học tập và tham gia chương trình AI Riser Vietnam 2026. Chưa xác định license mã nguồn mở chính thức — liên hệ trực tiếp nếu muốn sử dụng lại code.

---

##  Liên hệ

Muốn trải nghiệm demo trực tiếp hoặc trao đổi thêm về kiến trúc dự án — inbox trực tiếp qua LinkedIn.
