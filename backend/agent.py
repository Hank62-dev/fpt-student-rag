# backend/agent.py
import os
import time
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    load_index_from_storage, VectorStoreIndex,
    SimpleDirectoryReader, StorageContext, Settings
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.prompts import PromptTemplate

load_dotenv()

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Groq(
    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.9,
    max_tokens=1024,
)

SYSTEM_PROMPT = PromptTemplate(
    "Bạn là trợ lý AI chuyên hỗ trợ sinh viên Đại học FPT.\n"
    "Chỉ dùng thông tin từ tài liệu bên dưới. Trả lời bằng tiếng Việt, rõ ràng.\n"
    "Nếu không có thông tin: 'Tôi không có thông tin về vấn đề này.'\n\n"
    "Tài liệu:\n{context_str}\n\n"
    "Câu hỏi: {query_str}\n"
    "Trả lời:"
)

def _build_query_engine(idx):
    return idx.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
        text_qa_template=SYSTEM_PROMPT,
        streaming=True,
    )

# Load index
storage_context = StorageContext.from_defaults(persist_dir="indexes")
index = load_index_from_storage(storage_context)
query_engine = _build_query_engine(index)

def rebuild_index():
    """Rebuild index sau khi upload file mới"""
    global index, query_engine
    docs = SimpleDirectoryReader(
        "data",
        required_exts=[".pdf", ".docx", ".xlsx", ".txt", ".tex"],
        recursive=True,
    ).load_data()
    index = VectorStoreIndex.from_documents(docs, show_progress=False)
    index.storage_context.persist(persist_dir="indexes")
    query_engine = _build_query_engine(index)

# ── Cache ─────────────────────────────────────────────────────
CACHE_FILE = Path("cache.json")
_cache: dict = {}

def _load_cache():
    global _cache
    if CACHE_FILE.exists():
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}

def _save_cache():
    CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False, indent=2), encoding="utf-8")

def _cache_key(q: str) -> str:
    return hashlib.md5(q.strip().lower().encode()).hexdigest()

def get_cached(question: str):
    _load_cache()
    return _cache.get(_cache_key(question))

def set_cache(question: str, answer: str, sources: list):
    _load_cache()
    key = _cache_key(question)
    _cache[key] = {"answer": answer, "sources": sources, "ts": time.time()}
    if len(_cache) > 200:
        oldest = sorted(_cache.items(), key=lambda x: x[1].get("ts", 0))
        for k, _ in oldest[:50]:
            del _cache[k]
    _save_cache()

# ── FAQ ───────────────────────────────────────────────────────
# FAQ_MAP = {
#     "lịch thi": "Lịch thi được đăng trên FAP tại mục Examination. Sinh viên cần kiểm tra phòng thi, ca thi trước ít nhất 1 tuần.",
#     "điểm danh": "Quy chế điểm danh FPT: nghỉ quá 20% buổi học bị cấm thi. Sinh viên cần xin phép trước khi nghỉ.",
#     "đăng ký môn": "Đăng ký môn học trên FAP tại mục Registration trong thời gian mở đăng ký. Sinh viên cần đảm bảo đủ điều kiện tiên quyết.",
# }

def check_faq(question: str):
    q_lower = question.lower()
    for keyword, answer in FAQ_MAP.items():
        if keyword in q_lower:
            return answer
    return None