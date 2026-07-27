"""
Phase 4, layer 2: LLM-specific observability via Langfuse.

Where OpenTelemetry (app/telemetry.py) tells you "the generate step
took 800ms", Langfuse tells you exactly what prompt went in, what
came out, which model actually answered (primary or fallback), and
the token/cost breakdown -- the detail you need to debug *why* a
specific answer was wrong, not just that it was slow.

Feature-flagged: if LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY aren't
set, this module simply no-ops everywhere it's used. That means you
can leave Langfuse wired into app/llm.py permanently, before you've
even created a Langfuse account, without anything breaking.
"""
import logging
from contextlib import contextmanager
from app.config import settings

logger = logging.getLogger("app.langfuse_client")

_enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
_client = None

if _enabled:
    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse tracing enabled (host=%s)", settings.langfuse_host)
    except Exception as e:
        logger.warning("Failed to initialize Langfuse, continuing without it: %s", e)
        _enabled = False


def is_enabled() -> bool:
    return _enabled


@contextmanager
def trace_generation(*, name: str, model: str, input_messages: list[dict], metadata: dict | None = None):
    """
    Wraps one LLM call. Usage:

        with trace_generation(name="generate_answer", model=model_id,
                               input_messages=messages, metadata={"session_id": sid}) as gen:
            response = call_the_model(...)
            gen.record(output=response_text, usage=response.usage)

    No-ops cleanly (yields a stub with a no-op .record()) if Langfuse
    isn't configured, so callers never need an `if is_enabled()` check
    of their own.
    """
    if not _enabled:
        yield _NoOpGeneration()
        return

    try:
        with _client.start_as_current_observation(
            name=name, as_type="generation", model=model,
            input=input_messages, metadata=metadata or {},
        ) as generation:
            yield _LangfuseGeneration(generation)
    except Exception as e:
        logger.warning("Langfuse tracing failed, continuing without it: %s", e)
        yield _NoOpGeneration()


class _LangfuseGeneration:
    def __init__(self, generation):
        self._generation = generation

    def record(self, output: str, usage=None) -> None:
        try:
            usage_details = None
            if usage:
                usage_details = {
                    "input": getattr(usage, "prompt_tokens", None),
                    "output": getattr(usage, "completion_tokens", None),
                    "total": getattr(usage, "total_tokens", None),
                }
            self._generation.update(output=output, usage_details=usage_details)
        except Exception as e:
            logger.warning("Langfuse record() failed: %s", e)


class _NoOpGeneration:
    def record(self, output: str, usage=None) -> None:
        pass


def flush() -> None:
    """Call on app shutdown so any buffered traces actually get sent."""
    if _enabled and _client:
        try:
            _client.flush()
        except Exception as e:
            logger.warning("Langfuse flush() failed: %s", e)
