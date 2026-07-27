# Insurance Policy & Claims Assistant — Phase 1 (Core RAG)

A hybrid-search RAG pipeline over sample insurance policy documents, built to
map directly onto an Infocusp-style Senior Backend Engineer JD: Python/FastAPI,
PostgreSQL + pgvector, BM25 + vector hybrid search, and cross-encoder reranking.

## What this covers

| JD requirement | Where it lives |
|---|---|
| Python + FastAPI, async services | `app/main.py` |
| PostgreSQL at production scale | `sql/init.sql` |
| pgvector, hybrid retrieval (BM25 + vector) | `app/retrieval.py` |
| Semantic search / reranking | `app/rerank.py` |
| RAG pipeline design | `app/chunking.py`, `app/embeddings.py`, `app/ingest.py` |
| LangGraph, agentic workflows, tool calling | `app/agent.py`, `app/agent_state.py`, `app/tools.py` |
| Human-in-the-loop guardrails | `app/agent.py` (`pause_for_human` node) |

Phase 3 (guardrails: fallback routing, prompt-injection defense, token
budgets) and Phase 4 (OpenTelemetry + Langfuse observability) build on
top of this in later phases.

## Phase 2: the agentic layer

`POST /agent/ask` routes the same `{"question": "...", "policy_type": "..."}`
request through a LangGraph agent instead of the fixed Phase 1 pipeline.
The graph:

```
classify_intent
    -> policy_question -> retrieve_policy (Phase 1 RAG) -> generate_answer
    -> claim_status    -> call_claims_api (tool call)
                             -> high value / pending?  -> pause_for_human
                             -> otherwise               -> generate_answer
```

Three requests to try, each exercising a different path through the graph:

```bash
# Path 1: policy question -> RAG (same as Phase 1, just through the agent)
curl -X POST http://localhost:8000/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the waiting period for maternity coverage under policy HP-2024-119?", "policy_type": "health"}'

# Path 2: claim status, auto-answered (amount below the human-approval threshold)
curl -X POST http://localhost:8000/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the status of claim CLM-2024-8817?"}'

# Path 3: claim status, high-value -> pauses for human approval instead of answering
curl -X POST http://localhost:8000/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the status of claim CLM-2024-9021?"}'
```

Path 3 should return `needs_human_approval: true` and a generic "a claims
officer has been notified" message -- the agent deliberately does NOT
generate a specific answer about that claim, which is the point of the
human-in-the-loop checkpoint.

### Design notes worth being able to explain out loud (Phase 2)

- **Why intent classification is rule-based, not an LLM call**: keeps
  routing deterministic and free for this demo. In production you'd
  weigh an LLM/embedding classifier's better generalization against its
  added latency and cost on every single request -- a good "it depends"
  answer to have ready.
- **Why `generate_answer_node` is shared across both paths**: both the
  RAG path and the claims path end up producing the same `RetrievedChunk`
  shape before generation, so the prompt-building and grounding logic in
  `app/llm.py` never needs a claims-specific branch. One code path, two
  sources of "context."
- **Why the graph is built once at startup, not per-request**: compiling
  a LangGraph graph has overhead; `app.state.agent_graph` is built once
  in the `lifespan` handler and reused across every `/agent/ask` call.
- **Why `pause_for_human` is a dead-end node, not a retry loop**: a real
  system would pair this with a task/notification write (e.g. to a
  ticketing system) so a human actually sees it -- that's a natural
  "what would you add for production" follow-up to have an answer for.

## Setup

1. **Start Postgres with pgvector**
   ```bash
   docker compose up -d
   ```
   This runs `sql/init.sql` automatically on first boot, creating the
   `chunks` table (with both a `vector` column and a `tsvector` column)
   and a mock `claims` table for later phases.

2. **Install Python dependencies**
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # then edit .env with your LLM server's base URL, API key, and model id
   ```

4. **Ingest the sample policy documents**
   ```bash
   python -m app.ingest
   ```
   This chunks `sample_docs/*.txt` by `Section X.Y` headers (falling back
   to fixed-size + overlap splitting for any section that's too long),
   embeds each chunk locally via `sentence-transformers`, and inserts
   everything into Postgres.

5. **Run the API**
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Try it**
   ```bash
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the waiting period for maternity coverage under policy HP-2024-119?", "policy_type": "health"}'
   ```

   This question is deliberately designed to need BOTH retrieval signals:
   the exact policy number (BM25 catches this) and the semantic concept
   "maternity waiting period" (vector search catches this) — a good way
   to demonstrate hybrid search is actually doing something, not just
   window dressing.

## Try breaking it (worth doing before an interview)

- Ask a question with **no exact terms** ("when will my claim money arrive") —
  see how vector search alone handles it.
- Ask with **only an exact code** ("CLM-2024-8817") — see how BM25 dominates.
- Temporarily disable the `_bm25_search` or `_vector_search` call in
  `hybrid_search()` and compare answer quality — this is the fastest way
  to *feel* why hybrid search matters rather than just being able to
  explain it.

## Design notes worth being able to explain out loud

- **Why HNSW, not IVFFlat**: this corpus is small and static-ish (policy
  documents don't change hourly), so paying the build cost for HNSW's
  better recall/speed makes sense. See `sql/init.sql`.
- **Why RRF, not averaging scores**: `ts_rank` and cosine distance are on
  incomparable scales — RRF uses rank position instead. See
  `_reciprocal_rank_fusion` in `app/retrieval.py`.
- **Why rerank at all**: hybrid search is a cheap, fast first pass across
  the whole corpus; the cross-encoder reranker is expensive per-pair, so
  it only runs on the narrowed candidate set (`hybrid_candidate_k`, default
  20) rather than the whole table. See `app/rerank.py`.
- **Why chunk by Section header**: policy documents are naturally
  clause-structured, so structural chunking keeps each chunk semantically
  whole, rather than cutting a waiting-period clause in half. See
  `app/chunking.py`.
