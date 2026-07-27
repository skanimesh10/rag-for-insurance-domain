"""
Embedding model wrapper. Loaded once (singleton) since loading the
model from disk/HF hub is expensive -- you never want to reload it
per-request.

Uses a local sentence-transformers model, so this runs with no external
API calls and no per-request cost -- a reasonable default for a
learning project, and a real option in production for cost control.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


def embed_text(text: str) -> list[float]:
    """Embed a single piece of text (query or chunk)."""
    model = get_embedder()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed many chunks at once -- much faster than one-by-one during ingestion."""
    model = get_embedder()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return vectors.tolist()
