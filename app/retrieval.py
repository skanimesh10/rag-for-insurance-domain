"""
Hybrid retrieval = BM25 (lexical) + vector (semantic) search, fused via
Reciprocal Rank Fusion (RRF).

Why RRF and not averaging raw scores: ts_rank (BM25-ish) scores and
cosine distances live on completely different, incomparable scales.
RRF sidesteps that by using each result's *rank position* in its own
list instead of its raw score -- see app/README section on RRF.
"""
from dataclasses import dataclass
import asyncpg

from app.config import settings
from app.embeddings import embed_text
from app.telemetry import get_tracer


@dataclass
class RetrievedChunk:
    id: int
    doc_id: str
    doc_title: str
    section: str
    content: str
    rrf_score: float


async def _vector_search(conn: asyncpg.Connection, query_vector: list[float], k: int,
                          policy_type: str | None) -> list[asyncpg.Record]:
    """Semantic search: nearest neighbors by cosine distance (`<=>` operator)."""
    if policy_type:
        return await conn.fetch(
            """
            SELECT id, doc_id, doc_title, section, content
            FROM chunks
            WHERE policy_type = $2
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            query_vector, policy_type, k,
        )
    return await conn.fetch(
        """
        SELECT id, doc_id, doc_title, section, content
        FROM chunks
        ORDER BY embedding <=> $1
        LIMIT $2
        """,
        query_vector, k,
    )


async def _bm25_search(conn: asyncpg.Connection, query_text: str, k: int,
                        policy_type: str | None) -> list[asyncpg.Record]:
    """Lexical search: Postgres full-text ranking (BM25-like via ts_rank)."""
    if policy_type:
        return await conn.fetch(
            """
            SELECT id, doc_id, doc_title, section, content,
                   ts_rank(content_tsv, plainto_tsquery('english', $1)) AS rank
            FROM chunks
            WHERE policy_type = $3 AND content_tsv @@ plainto_tsquery('english', $1)
            ORDER BY rank DESC
            LIMIT $2
            """,
            query_text, k, policy_type,
        )
    return await conn.fetch(
        """
        SELECT id, doc_id, doc_title, section, content,
               ts_rank(content_tsv, plainto_tsquery('english', $1)) AS rank
        FROM chunks
        WHERE content_tsv @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC
        LIMIT $2
        """,
        query_text, k,
    )


def _reciprocal_rank_fusion(
    vector_results: list[asyncpg.Record],
    bm25_results: list[asyncpg.Record],
    k: int,
) -> list[RetrievedChunk]:
    """
    RRF score for a doc = sum over each ranked list it appears in of
    1 / (k + rank), where rank is 1-indexed position in that list.
    A doc that ranks well in BOTH lists gets boosted the most.
    """
    scores: dict[int, float] = {}
    records: dict[int, asyncpg.Record] = {}

    for rank, row in enumerate(vector_results, start=1):
        scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (k + rank)
        records[row["id"]] = row

    for rank, row in enumerate(bm25_results, start=1):
        scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (k + rank)
        records[row["id"]] = row

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    return [
        RetrievedChunk(
            id=cid,
            doc_id=records[cid]["doc_id"],
            doc_title=records[cid]["doc_title"],
            section=records[cid]["section"],
            content=records[cid]["content"],
            rrf_score=scores[cid],
        )
        for cid in ranked_ids
    ]


async def hybrid_search(
    pool: asyncpg.Pool,
    query_text: str,
    policy_type: str | None = None,
    candidate_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Run BM25 + vector search in parallel (conceptually -- here sequentially
    for simplicity), fuse with RRF, and return a fused ranked list.
    This is the *candidate* set -- reranking (app/rerank.py) narrows it
    further before it goes into the LLM prompt.
    """
    tracer = get_tracer()

    k = candidate_k or settings.hybrid_candidate_k

    with tracer.start_as_current_span("embed_query") as span:
        span.set_attribute("model", settings.embedding_model_name)
        query_vector = embed_text(query_text)

    async with pool.acquire() as conn:
        with tracer.start_as_current_span("vector_search") as span:
            span.set_attribute("candidate_k", k)
            span.set_attribute("policy_type", policy_type or "none")
            vector_results = await _vector_search(conn, query_vector, k, policy_type)
            span.set_attribute("result_count", len(vector_results))

        with tracer.start_as_current_span("bm25_search") as span:
            span.set_attribute("candidate_k", k)
            bm25_results = await _bm25_search(conn, query_text, k, policy_type)
            span.set_attribute("result_count", len(bm25_results))

    with tracer.start_as_current_span("rrf_fusion") as span:
        fused = _reciprocal_rank_fusion(vector_results, bm25_results, k=settings.rrf_k)
        span.set_attribute("fused_result_count", len(fused))

    return fused
