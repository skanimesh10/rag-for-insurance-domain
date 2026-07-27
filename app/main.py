"""
FastAPI entrypoint. Run with: uvicorn app.main:app --reload

Phase 1 endpoint: POST /ask
    { "question": "...", "policy_type": "health" }  -> grounded answer + citations
    Straight-line RAG pipeline, no branching, no tools.

Phase 2 endpoint: POST /agent/ask
    Same request shape, but routed through a LangGraph agent that
    classifies intent, branches to RAG or a claims-API tool call, and
    pauses for human approval on high-value claims. See app/agent.py.

Phase 3 guardrails (active on both endpoints, see app/llm.py):
    - fallback routing: primary model retried, then a fallback model,
      then a canned response -- callers never see a raw exception
    - prompt-injection defense: retrieved context is delimited and the
      model is told to treat it as inert data, never instructions
    - token budgets: pass "session_id" in the request to accumulate
      usage across calls; once a session hits max_tokens_per_session,
      further calls short-circuit to a budget-exceeded message without
      spending another token

Phase 4 observability (now active, see app/telemetry.py and
app/langfuse_client.py):
    - OpenTelemetry: every request gets a trace with child spans for
      each pipeline stage (embed, vector search, BM25, RRF fusion,
      rerank, generate) -- exported to Jaeger via docker-compose, or
      any OTLP backend in production
    - Langfuse: the exact prompt/output/token usage for every LLM call,
      including which model (primary or fallback) actually answered --
      set LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY to enable; safely
      no-ops if unset
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from app.db import get_pool, close_pool
from app.retrieval import hybrid_search
from app.rerank import rerank
from app.llm import generate_answer
from app.agent import build_agent_graph
from app.telemetry import setup_telemetry
from app.langfuse_client import flush as flush_langfuse


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_telemetry(app)              # Phase 4: OTel tracing, wraps FastAPI's request handling
    pool = await get_pool()           # warm the connection pool + register pgvector codec on startup
    app.state.agent_graph = build_agent_graph(pool)   # Phase 2: build the LangGraph agent once
    yield
    flush_langfuse()                  # make sure any buffered Langfuse traces get sent before exit
    await close_pool()


app = FastAPI(title="Insurance Policy & Claims Assistant - Phase 4 (RAG + Agent + Guardrails + Observability)", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    policy_type: str | None = None    # optional metadata filter, e.g. "health" or "motor"
    session_id: str = "anonymous"      # used for per-session token budget tracking (Phase 3)


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

    # Stage 3: generation, grounded in the reranked chunks, with guardrails
    answer = generate_answer(req.question, top_chunks, session_id=req.session_id)

    return AskResponse(
        answer=answer,
        citations=[Citation(doc_title=c.doc_title, section=c.section) for c in top_chunks],
    )


class AgentAskResponse(BaseModel):
    answer: str
    intent: str
    needs_human_approval: bool
    citations: list[Citation]


@app.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(req: AskRequest) -> AgentAskResponse:
    """
    Phase 2: routes through the LangGraph agent instead of always
    running the fixed RAG pipeline. Try a policy question, a claim
    status question (e.g. 'What's the status of CLM-2024-8817?'), and
    a high-value claim (e.g. 'What's the status of CLM-2024-9021?')
    to see all three paths through the graph.
    """
    result = await app.state.agent_graph.ainvoke(
        {"question": req.question, "policy_type": req.policy_type, "session_id": req.session_id}
    )

    return AgentAskResponse(
        answer=result["answer"],
        intent=result["intent"],
        needs_human_approval=result.get("needs_human_approval", False),
        citations=[
            Citation(doc_title=c.doc_title, section=c.section)
            for c in result.get("citations", [])
        ],
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
