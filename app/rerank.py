"""
Cross-encoder reranking: takes the (fast, approximate) hybrid search
candidates and re-scores them with a model that processes the query
and each chunk TOGETHER -- much more accurate, but too slow to run
against the whole corpus, hence why it only runs on the narrowed
candidate set from hybrid_search().
"""
from functools import lru_cache
from sentence_transformers import CrossEncoder

from app.config import settings
from app.retrieval import RetrievedChunk
from app.telemetry import get_tracer


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model_name)


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
    """
    Score every (query, chunk) pair jointly, then return only the
    top_k highest-scoring chunks -- this is what actually goes into
    the LLM's context window.
    """
    if not candidates:
        return []

    top_k = top_k or settings.rerank_top_k
    tracer = get_tracer()

    with tracer.start_as_current_span("rerank") as span:
        span.set_attribute("candidate_count", len(candidates))
        span.set_attribute("top_k", top_k)

        model = get_reranker()
        pairs = [(query, c.content) for c in candidates]
        scores = model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        result = [chunk for chunk, _score in scored[:top_k]]

        span.set_attribute("top_score", float(scored[0][1]) if scored else 0.0)

    return result
