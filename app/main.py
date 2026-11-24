"""FastAPI service for the Modern Furniture Pacific AI sales assistant."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent import MODEL_ID, run_chat

app = FastAPI(title="Modern Furniture Pacific — AI Sales Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(default="user", description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    answer: str
    products: list[dict] = []
    whatsapp_shortlist_link: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = run_chat([m.model_dump() for m in req.messages])
    return ChatResponse(
        answer=result["answer"],
        products=result["products"],
        whatsapp_shortlist_link=result.get("whatsapp_shortlist_link"),
    )
