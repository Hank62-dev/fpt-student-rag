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
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
load_dotenv()

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=50)]
Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
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
if Path("indexes").exists() and any(Path("indexes").iterdir()):
    storage_context = StorageContext.from_defaults(persist_dir="indexes")
    index = load_index_from_storage(storage_context)
else:
    index = VectorStoreIndex.from_documents([])
    index.storage_context.persist(persist_dir="indexes")

query_engine = _build_query_engine(index)

def rebuild_index():
    global index, query_engine
    
    if Path("indexes").exists() and any(Path("indexes").iterdir()):
        storage_context = StorageContext.from_defaults(persist_dir="indexes")
        index = load_index_from_storage(storage_context)
    else:
        index = VectorStoreIndex.from_documents([])

    indexed_files = set()
    for doc_id, doc_info in index.docstore.docs.items():
        fname = (
            doc_info.metadata.get("file_name") or
            doc_info.metadata.get("filename") or
            doc_info.metadata.get("source") or ""
        )
        if fname:
            indexed_files.add(Path(fname).name)

    print(f"Files đã index: {indexed_files}")
    
    new_docs = []
    for f in Path("data").iterdir():
        if f.suffix.lower() in {".pdf", ".docx", ".txt", ".tex"}:
            if f.name not in indexed_files:
                print(f"Loading file mới: {f.name}")
                docs = SimpleDirectoryReader(input_files=[str(f)]).load_data()
                new_docs.extend(docs)
        elif f.suffix.lower() == ".xlsx":
            if f.name not in indexed_files:
                print(f"Loading Excel: {f.name}")
                docs = load_excel(str(f))
                print(f"Loaded {len(docs)} sheets từ {f.name}")
                new_docs.extend(docs)

    print(f"Tổng new_docs: {len(new_docs)}")
    
    if new_docs:
        for doc in new_docs:
            index.insert(doc)
        index.storage_context.persist(persist_dir="indexes")

    # Luôn update query_engine dù có file mới hay không
    query_engine = _build_query_engine(index)
    print(f"Query engine rebuilt! Tổng docs: {len(index.docstore.docs)}")

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
FAQ_MAP = {
    "đăng ký môn": "Đăng ký môn học trên FAP tại mục Registration trong thời gian mở đăng ký. Sinh viên cần đảm bảo đủ điều kiện tiên quyết.",
}

def check_faq(question: str):
    q_lower = question.lower()
    for keyword, answer in FAQ_MAP.items():
        if keyword in q_lower:
            return answer
    return None

import pandas as pd
from llama_index.core import Document

def load_excel(file_path: str) -> list:
    """Đọc Excel giữ nguyên cấu trúc bảng"""
    docs = []
    xl = pd.ExcelFile(file_path)
    
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Chuyển bảng thành text có cấu trúc
        text = f"Sheet: {sheet_name}\n\n"
        text += df.to_markdown(index=False)
        
        docs.append(Document(
            text=text,
            metadata={"file_name": Path(file_path).name, "sheet": sheet_name}
        ))
    
    return docs