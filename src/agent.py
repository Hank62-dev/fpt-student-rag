# src/agent.py
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from dotenv import load_dotenv
import os

load_dotenv()

# Cấu hình
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
llm = OpenAI(model="gpt-4o-mini", temperature=0.3)   # thay "gemini" nếu dùng Gemini

# Load index đã build sẵn
storage_context = StorageContext.from_defaults(persist_dir="indexes")
index = load_index_from_storage(storage_context)

# Tạo query engine + tool cho Agent
query_engine = index.as_query_engine(similarity_top_k=6)
query_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="fpt_student_knowledge_base",
        description="Dùng để trả lời mọi câu hỏi về quy chế, lịch học, thủ tục sinh viên FPT. Luôn trả lời bằng tiếng Việt và trích dẫn nguồn rõ ràng."
    )
)

# Tạo Agentic RAG (ReAct Agent)
agent = ReActAgent.from_tools(
    tools=[query_tool],
    llm=llm,
    verbose=True,           # xem agent suy nghĩ từng bước (rất hay khi debug)
    max_iterations=8
)

def ask_question(question: str):
    response = agent.chat(question)
    return response