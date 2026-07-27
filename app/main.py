"""
FastAPI entrypoint. Run with: uvicorn app.main:app --reload

Single endpoint for Phase 1: POST /ask
    { "question": "...", "policy_type": "health" }  -> grounded answer + citations

This is deliberately just the RAG pipeline end-to-end -- no agentic
routing yet (that's Phase 2, LangGraph) and no guardrails yet
(Phase 3: fallback routing, prompt-injection defense, token budgets).
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from app.db import get_pool, close_pool
from app.retrieval import hybrid_search
from app.rerank import rerank
from app.llm import generate_answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_pool()  # warm the connection pool + register pgvector codec on startup
    yield
    await close_pool()


app = FastAPI(title="Insurance Policy & Claims Assistant - Phase 1 (RAG)", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    policy_type: str | None = None  # optional metadata filter, e.g. "health" or "motor"


class Citation(BaseModel):
    doc_title: str
    section: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    pool = await get_pool()

    # Stage 1: hybrid retrieval (BM25 + vector, fused with RRF)
    candidates = await hybrid_search(pool, req.question, policy_type=req.policy_type)

    # Stage 2: cross-encoder reranking down to the top few, high-precision chunks
    top_chunks = rerank(req.question, candidates)

    # Stage 3: generation, grounded in the reranked chunks
    answer = generate_answer(req.question, top_chunks)

    return AskResponse(
        answer=answer,
        citations=[Citation(doc_title=c.doc_title, section=c.section) for c in top_chunks],
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
