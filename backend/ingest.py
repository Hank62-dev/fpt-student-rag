# backend/ingest.py
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

print("📄 Đang load tài liệu từ folder data/...")
documents = SimpleDirectoryReader(
    "data",
    required_exts=[".pdf", ".docx", ".txt", ".xlsx"],
    recursive=True,
).load_data()

print(f"✅ Load xong {len(documents)} documents. Đang build index...")
index = VectorStoreIndex.from_documents(documents, show_progress=True)
index.storage_context.persist(persist_dir="indexes")
print("🎉 Index đã lưu vào 'indexes/'")