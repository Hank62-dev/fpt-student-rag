import os
import time
from dotenv import load_dotenv

from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.prompts import PromptTemplate

load_dotenv()

#  EMBEDDING 
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

#  LLM + RETRY 
class RetryGoogleGenAI(GoogleGenAI):
    """Gemini với auto-retry khi bị rate limit 429"""

    def _retry(self, fn, *args, **kwargs):
        delays = [5, 15, 30]
        for i, delay in enumerate(delays):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate" in err or "quota" in err:
                    print(f" Rate limit hit, thử lại sau {delay}s... (lần {i+1}/3)")
                    time.sleep(delay)
                else:
                    raise
        raise Exception("Vẫn bị rate limit sau 3 lần retry. Vui lòng thử lại sau.")

    def complete(self, prompt, **kwargs):
        return self._retry(super().complete, prompt, **kwargs)

    def chat(self, messages, **kwargs):
        return self._retry(super().chat, messages, **kwargs)

    def stream_complete(self, prompt, **kwargs):
        return self._retry(super().stream_complete, prompt, **kwargs)

    def stream_chat(self, messages, **kwargs):
        return self._retry(super().stream_chat, messages, **kwargs)


from llama_index.llms.groq import Groq

Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.05,
    max_tokens=1024,
)

#  LOAD INDEX 
storage_context = StorageContext.from_defaults(persist_dir="indexes")
index = load_index_from_storage(storage_context)

SYSTEM_PROMPT = PromptTemplate(
    "Bạn là trợ lý thông minh chuyên hỗ trợ sinh viên Đại học FPT.\n"
    "Chỉ sử dụng thông tin từ tài liệu được cung cấp bên dưới để trả lời.\n"
    "Trả lời bằng tiếng Việt, rõ ràng, lịch sự và trích dẫn nguồn nếu có thể.\n"
    "Nếu không tìm thấy thông tin trong tài liệu, hãy nói thẳng: "
    "'Tôi không có thông tin về vấn đề này trong tài liệu FAP.'\n\n"
    "Thông tin từ tài liệu:\n{context_str}\n\n"
    "Câu hỏi: {query_str}\n"
    "Trả lời:"
)

#  QUERY ENGINE 
query_engine = index.as_query_engine(
    similarity_top_k=6,
    response_mode="tree_summarize",
    text_qa_template=SYSTEM_PROMPT,
)


def ask_question(question: str):
    """Trả lời câu hỏi từ RAG pipeline"""
    try:
        response = query_engine.query(question)
        return response
    except Exception as e:
        return f"Đã xảy ra lỗi: {str(e)}"