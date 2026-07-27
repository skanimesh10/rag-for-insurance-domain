"""
Central config. Everything reads from environment variables (via .env),
so nothing is hardcoded -- this matters once you deploy anywhere.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://rag_user:rag_password@localhost:5432/insurance_rag"

    # Embedding model: runs locally via sentence-transformers, no external API needed.
    # 384-dim output -- must match the `vector(384)` column in sql/init.sql.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Cross-encoder reranker: also local, no external API needed.
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM for generation. Uses the OpenAI-compatible API format, which covers
    # most open-source model servers (vLLM, Ollama, Together, LM Studio, etc.)
    # as well as OpenAI itself -- just point base_url at whichever server
    # you're running.
    llm_base_url: str = "http://localhost:8080/v1"   # e.g. your vLLM/Ollama/Together endpoint
    llm_api_key: str = ""                             # some local servers accept any string here
    primary_model: str = "your-model-id"               # the model id your server expects
    fallback_model: str = "your-fallback-model-id"     # used in Phase 3 guardrails

    # Retrieval tuning
    hybrid_candidate_k: int = 20      # how many candidates each of BM25/vector search returns
    rerank_top_k: int = 4             # how many chunks survive reranking into the final prompt
    rrf_k: int = 60                   # RRF fusion constant (60 is the common default)

    # Guardrails
    max_tokens_per_session: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()
