import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding

load_dotenv()

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=1024,
)

storage_context = StorageContext.from_defaults(persist_dir="indexes")
index = load_index_from_storage(storage_context)

nodes = index.as_retriever(similarity_top_k=3).retrieve("học phí")
print(f"Retrieved {len(nodes)} nodes")
for n in nodes:
    print(f"  - {n.metadata.get('file_name')} score={n.score}")
    print(f"    {n.text[:100]}")