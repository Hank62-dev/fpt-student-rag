# backend/main.py
import os
import json
import time
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from collections import defaultdict

from agent import (
    query_engine,
    get_cached, set_cache,
    rewrite_query, check_faq,
    Settings
)

load_dotenv()

app = FastAPI(title="FPT RAG API", version="2.0.0")

# ── CORS ──────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Rate limiting (in-memory) ─────────────────────────────────
_rate_store: dict = defaultdict(list)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))

def rate_limit(request: Request):
    ip = request.client.host
    now = time.time()
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < 60]
    if len(_rate_store[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Quá nhiều yêu cầu. Vui lòng chờ 1 phút.")
    _rate_store[ip].append(now)

# ── Session store (in-memory) ─────────────────────────────────
# { session_id: [ {role, content, ts}, ... ] }
_sessions: dict = defaultdict(list)
SESSION_MAX = 20  # tối đa 20 lượt/session

# ── Schema ────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    session_id: str | None = None

    @field_validator("question")
    @classmethod
    def clean_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Câu hỏi không được để trống")
        if len(v) > 500:
            raise ValueError("Câu hỏi quá dài (tối đa 500 ký tự)")
        # strip sensitive patterns
        import re
        v = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[email]', v)
        return v

# ── Health ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# ── Session ───────────────────────────────────────────────────
@app.get("/session/new")
def new_session():
    sid = str(uuid.uuid4())
    return {"session_id": sid}

@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    return {"history": _sessions.get(session_id, [])}

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"cleared": True}

# ── Main streaming endpoint ───────────────────────────────────
@app.post("/ask/stream")
def ask_stream(req: AskRequest, _=Depends(rate_limit)):
    question = req.question
    session_id = req.session_id or str(uuid.uuid4())

    def generate():
        # 1. Gửi session_id
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        # 2. Kiểm tra FAQ shortcuts (không tốn token)
        faq_answer = check_faq(question)
        if faq_answer:
            for token in faq_answer:
                yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': [], 'cached': False, 'faq': True})}\n\n"
            _append_session(session_id, question, faq_answer)
            return

        # 3. Kiểm tra cache
        cached = get_cached(question)
        if cached:
            for token in cached["answer"]:
                yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"
                time.sleep(0.005)  # fake stream cho UX
            yield f"data: {json.dumps({'type': 'done', 'sources': cached['sources'], 'cached': True})}\n\n"
            _append_session(session_id, question, cached["answer"])
            return

        # 4. Rewrite query để tăng retrieval accuracy
        rewritten = question

        # 5. Lấy source nodes
        try:
            source_nodes = query_engine.retriever.retrieve(rewritten)
        except Exception:
            source_nodes = []

        # 6. Stream LLM response
        full_answer = ""
        try:
            streaming_resp = query_engine.query(rewritten)
            for token in streaming_resp.response_gen:
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"
        except Exception as e:
            err_msg = f"Đã xảy ra lỗi: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'text': err_msg})}\n\n"
            return

        # 7. Gửi sources + done
        sources = []
        for node in source_nodes[:5]:
            file_name = node.metadata.get("file_name", "Tài liệu FAP")
            score = round(node.score, 3) if hasattr(node, "score") and node.score else "N/A"
            sources.append({"file_name": file_name, "score": score})

        yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'cached': False}, ensure_ascii=False)}\n\n"

        # 8. Lưu cache + session
        if full_answer:
            set_cache(question, full_answer, sources)
            _append_session(session_id, question, full_answer)

    return StreamingResponse(generate(), media_type="text/event-stream")


def _append_session(session_id: str, question: str, answer: str):
    history = _sessions[session_id]
    history.append({"role": "user", "content": question, "ts": time.time()})
    history.append({"role": "assistant", "content": answer, "ts": time.time()})
    # giữ tối đa SESSION_MAX lượt
    if len(history) > SESSION_MAX * 2:
        _sessions[session_id] = history[-(SESSION_MAX * 2):]