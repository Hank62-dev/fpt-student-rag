# src/ingest.py
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

# Cấu hình embedding (bge-m3 - tốt nhất cho tiếng Việt 2026)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

print("📄 Đang load PDF từ folder data/...")
documents = SimpleDirectoryReader("data").load_data()

print(f"✅ Load xong {len(documents)} documents. Đang build index...")
index = VectorStoreIndex.from_documents(documents)

# Lưu index để sau không phải build lại
index.storage_context.persist(persist_dir="indexes")
print("🎉 Index đã lưu vào folder 'indexes/' - Sẵn sàng dùng!")