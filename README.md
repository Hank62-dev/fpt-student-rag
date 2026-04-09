# 🎓 FPT Student RAG Assistant

AI RAG hỗ trợ sinh viên FPT · **Claude API** + LlamaIndex + FastAPI

---

## Stack

| Layer     | Tech                                  |
|-----------|---------------------------------------|
| LLM       | **Anthropic Claude** (claude-sonnet)  |
| Embedding | BAAI/bge-m3 (HuggingFace)            |
| RAG       | LlamaIndex                            |
| Backend   | FastAPI                               |
| Frontend  | HTML/CSS/JS thuần (không dependency)  |

---

## Cấu trúc thư mục

```
fpt-rag/
├── backend/
│   ├── agent.py          ← RAG pipeline (Claude LLM)
│   ├── ingest.py         ← Build index từ PDF
│   ├── main.py           ← FastAPI server
│   ├── requirements.txt
│   ├── .env              ← bạn tự tạo từ .env.example
│   ├── data/             ← bỏ PDF của FAP vào đây
│   └── indexes/          ← tự sinh sau khi chạy ingest.py
└── frontend/
    └── index.html        ← Mở thẳng trên browser
```

---

## Cài đặt & Chạy

### 1. Clone & cài dependencies

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Tạo file .env

```bash
cp .env.example .env
# Mở .env và điền ANTHROPIC_API_KEY
```

Lấy API key tại: https://console.anthropic.com/

### 3. Build index từ PDF

```bash
# Bỏ file PDF của FAP vào thư mục data/
python ingest.py
```

### 4. Khởi động backend

```bash
uvicorn main:app --reload --port 8000
```

### 5. Mở Frontend

Mở file `frontend/index.html` trực tiếp trên trình duyệt.  
Hoặc dùng Live Server (VS Code extension).

---

## API Endpoints

| Method | Path     | Mô tả                    |
|--------|----------|--------------------------|
| GET    | /health  | Kiểm tra server          |
| POST   | /ask     | Gửi câu hỏi, nhận trả lời |

### Request `/ask`
```json
{ "question": "Quy chế điểm danh của FPT?" }
```

### Response
```json
{
  "answer": "...",
  "sources": [
    { "file_name": "quy_che.pdf", "score": 0.87 }
  ],
  "elapsed": 1.23
}
```

---
