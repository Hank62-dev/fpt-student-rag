# backend/main.py
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import ask_question

load_dotenv()

app = FastAPI(title="FPT Student RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class SourceNode(BaseModel):
    file_name: str
    score: float | str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceNode]
    elapsed: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()
    response = ask_question(req.question)
    elapsed = round(time.time() - start, 2)

    if hasattr(response, "response") and response.response:
        answer_text = response.response
        source_nodes = getattr(response, "source_nodes", [])
    else:
        answer_text = str(response)
        source_nodes = []

    sources = []
    for node in source_nodes[:5]:
        file_name = node.metadata.get("file_name", "Tài liệu FAP")
        score = round(node.score, 3) if hasattr(node, "score") and node.score else "N/A"
        sources.append(SourceNode(file_name=file_name, score=score))

    return AnswerResponse(answer=answer_text, sources=sources, elapsed=elapsed)