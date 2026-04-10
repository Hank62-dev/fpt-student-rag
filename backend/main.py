# backend/main.py
import os
import json
import time
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, BackgroundTasks
from agent import (
    query_engine,
    get_cached, set_cache,
    check_faq, Settings,
    rebuild_index
)

load_dotenv()

app = FastAPI(title="FPT RAG API", version="3.0.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://fpt-student-rag.vercel.app").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Rate limiting ─────────────────────────────────────────────
_rate_store: dict = defaultdict(list)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))

def rate_limit(request: Request):
    ip = request.client.host
    now = time.time()
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < 60]
    if len(_rate_store[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Quá nhiều yêu cầu. Vui lòng chờ 1 phút.")
    _rate_store[ip].append(now)

# ── Session store ─────────────────────────────────────────────
_sessions: dict = defaultdict(list)
SESSION_MAX = 20

# ── Schema ────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    session_id: str | None = None

    @field_validator("question")
    @classmethod
    def clean_question(cls, v):
        import re
        v = v.strip()
        if not v:
            raise ValueError("Câu hỏi không được để trống")
        if len(v) > 500:
            raise ValueError("Câu hỏi quá dài (tối đa 500 ký tự)")
        v = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[email]', v)
        return v

# ── Health ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# ── Session ───────────────────────────────────────────────────
@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    return {"history": _sessions.get(session_id, [])}

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"cleared": True}

# ── Upload file ───────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt", ".tex"}
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    try:
        # Tạo thư mục data nếu chưa có
        data_path = Path("data")
        data_path.mkdir(exist_ok=True)
        
        file_path = data_path / file.filename
        
        # Lưu file xuống đĩa 
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        #  yêu cầu FastAPI chạy nó ngầm (background)
        background_tasks.add_task(rebuild_index)
        
        return {
            "success": True, 
            "message": f"File {file.filename} đã được tải lên thành công và đang được xử lý ngầm.",
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
def list_files():
    files = []
    for f in DATA_DIR.iterdir():
        if f.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "ext": f.suffix.lower()
            })
    return {"files": files}

@app.delete("/files/{filename}")
def delete_file(filename: str):
    f = DATA_DIR / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="File không tồn tại")
    f.unlink()
    try:
        rebuild_index()
    except Exception:
        pass
    return {"deleted": True}

# ── Main streaming endpoint ───────────────────────────────────
@app.post("/ask/stream")
def ask_stream(req: AskRequest, _=Depends(rate_limit)):
    question = req.question
    session_id = req.session_id or str(uuid.uuid4())

    def generate():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        # FAQ
        faq_answer = check_faq(question)
        if faq_answer:
            for token in faq_answer:
                yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': [], 'cached': False, 'faq': True})}\n\n"
            _append_session(session_id, question, faq_answer)
            return

        # Cache
        cached = get_cached(question)
        if cached:
            for token in cached["answer"]:
                yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"
                time.sleep(0.005)
            yield f"data: {json.dumps({'type': 'done', 'sources': cached['sources'], 'cached': True})}\n\n"
            _append_session(session_id, question, cached["answer"])
            return

        # Retrieval
        try:
            source_nodes = query_engine.retriever.retrieve(question)
        except Exception:
            source_nodes = []

        # Stream LLM
        full_answer = ""
        try:
            streaming_resp = query_engine.query(question)
            for token in streaming_resp.response_gen:
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
            return

        sources = []
        for node in source_nodes[:5]:
            file_name = node.metadata.get("file_name", "Tài liệu FAP")
            score = round(node.score, 3) if hasattr(node, "score") and node.score else "N/A"
            sources.append({"file_name": file_name, "score": score})

        yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'cached': False}, ensure_ascii=False)}\n\n"

        if full_answer:
            set_cache(question, full_answer, sources)
            _append_session(session_id, question, full_answer)

    return StreamingResponse(generate(), media_type="text/event-stream")


def _append_session(session_id: str, question: str, answer: str):
    history = _sessions[session_id]
    history.append({"role": "user", "content": question, "ts": time.time()})
    history.append({"role": "assistant", "content": answer, "ts": time.time()})
    if len(history) > SESSION_MAX * 2:
        _sessions[session_id] = history[-(SESSION_MAX * 2):]