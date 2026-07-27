"""
Guardrail helpers that don't belong inside app/llm.py itself:

1. Token budget tracking per session -- a simple in-memory counter.
   Good enough for a single-process demo; a real deployment would back
   this with Redis (or similar) so the budget survives restarts and is
   shared across multiple app instances.

2. A lightweight heuristic flag for obviously-suspicious retrieved
   content. This is NOT a substitute for the structural defense in
   app/llm.py (delimiting untrusted context + instructing the model to
   treat it as data) -- it's a cheap signal for logging/alerting so a
   suspicious document gets flagged for review, not silently trusted.
"""
import re

_session_usage: dict[str, int] = {}

INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|the)?\s*(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard (all|any|the)?\s*(previous|prior|above)", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"reveal (your|the) (instructions|system prompt)", re.IGNORECASE),
]


def get_session_usage(session_id: str) -> int:
    return _session_usage.get(session_id, 0)


def record_usage(session_id: str, tokens: int) -> None:
    _session_usage[session_id] = _session_usage.get(session_id, 0) + tokens


def is_budget_exceeded(session_id: str, max_tokens: int) -> bool:
    return get_session_usage(session_id) >= max_tokens


def flag_suspicious_content(chunks_text: list[str]) -> list[str]:
    """
    Returns the subset of chunk texts that match an obvious injection
    pattern. Used purely as a signal to log/flag -- the actual defense
    is that the model is instructed to treat ALL retrieved content as
    inert data regardless of what it contains (see app/llm.py).
    """
    flagged = []
    for text in chunks_text:
        if any(pattern.search(text) for pattern in INJECTION_PATTERNS):
            flagged.append(text)
    return flagged
