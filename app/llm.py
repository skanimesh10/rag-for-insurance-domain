"""
Phase 1 generation: a single, straightforward LLM call that assembles
retrieved chunks into a grounded prompt.

Uses the `openai` Python client purely as an HTTP client for the
OpenAI-compatible chat completions API -- this works against OpenAI
itself, or against any open-source model server that speaks the same
API shape (vLLM, Ollama, Together, LM Studio, text-generation-inference,
etc.). Point `llm_base_url` at your server and `primary_model` at
whatever model id that server expects.

NOTE: this intentionally has no fallback routing, prompt-injection
delimiting, or token budgeting yet -- those are Phase 3 (guardrails).
"""
from openai import OpenAI
from app.config import settings
from app.retrieval import RetrievedChunk

_client = OpenAI(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key or "not-needed",  # many local servers ignore this but the client requires a non-empty string
)

SYSTEM_PROMPT = """You are an assistant that answers insurance policy questions
using ONLY the context provided below. If the answer is not contained in the
context, say you don't know rather than guessing. Always cite the section
each fact comes from."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c.doc_title} - {c.section}]\n{c.content}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context_block(chunks)

    response = _client.chat.completions.create(
        model=settings.primary_model,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        ],
    )
    return response.choices[0].message.content
