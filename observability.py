"""
Langfuse Observability Module
==============================
Single shared client, context-variable-based propagation.

All spans/generations are attached as direct children of the root trace so
the module works safely in both asyncio tasks AND ThreadPoolExecutor workers.

Usage pattern
-------------
1. Request enters  → start_trace(...)
2. Each component  → new_span() / end_span()  or  record_generation()
3. Request exits   → finish_trace(trace, ...)

Context propagation
-------------------
- asyncio.to_thread() copies the current contextvars.Context automatically.
- ThreadPoolExecutor workers receive a *per-task* copy via
  `contextvars.copy_context()` (see toc_router.py parallel block).
- Workers only READ _active_trace, never modify it — so sharing is safe.
"""

import os
import time
import logging
from typing import Optional
from contextvars import ContextVar
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

SLOW_REQUEST_THRESHOLD_MS: int = int(os.getenv("SLOW_REQUEST_THRESHOLD_MS", "5000"))

# ── Shared Langfuse client (lazy-initialized once) ─────────────────────────

_langfuse_client = None


def _get_client():
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    # Accept both LANGFUSE_HOST and LANGFUSE_BASE_URL (common aliases)
    host = (
        os.getenv("LANGFUSE_HOST")
        or os.getenv("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    )

    if not public_key or not secret_key:
        logger.error(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — "
            "observability disabled. Check your .env file."
        )
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info(f"Langfuse observability initialized (host={host})")
        return _langfuse_client
    except ImportError:
        logger.error("langfuse package not installed — run: pip install langfuse")
        return None
    except Exception as exc:
        logger.error(f"Langfuse init failed (tracing disabled): {exc}")
        return None


# ── ContextVar: active trace ──────────────────────────────────────────────────
# Propagated automatically by asyncio.to_thread() via context copy.
# ThreadPoolExecutor workers get a copy via contextvars.copy_context().

_active_trace: ContextVar = ContextVar("langfuse_active_trace", default=None)


def get_trace():
    """Return the active Langfuse trace for the current execution context."""
    return _active_trace.get()


# ── Trace lifecycle ───────────────────────────────────────────────────────────

def start_trace(
    name: str,
    session_id: Optional[str] = None,
    input_query: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """
    Create a root trace for one user request.
    Sets _active_trace so all downstream code (including threads) can see it.
    Returns the trace object, or None if Langfuse is not configured.
    """
    client = _get_client()
    if client is None:
        _active_trace.set(None)
        return None

    try:
        trace = client.trace(
            name=name,
            session_id=session_id,
            input=input_query,
            metadata=metadata or {},
        )
        _active_trace.set(trace)
        return trace
    except Exception as exc:
        logger.debug(f"start_trace error: {exc}")
        _active_trace.set(None)
        return None


def finish_trace(
    trace,
    output: Optional[str] = None,
    metadata: Optional[dict] = None,
    total_duration_ms: Optional[int] = None,
):
    """
    Finalize the root trace with output + summary metadata, then flush.
    Always call this at the end of each request, even on error.
    """
    if trace is None:
        return

    extra: dict = dict(metadata or {})
    if total_duration_ms is not None:
        extra["total_duration_ms"] = total_duration_ms
        if total_duration_ms > SLOW_REQUEST_THRESHOLD_MS:
            extra["slow_request"] = True

    try:
        trace.update(output=output, metadata=extra)
    except Exception as exc:
        logger.debug(f"finish_trace update error: {exc}")

    _flush()


def _flush():
    client = _get_client()
    if client:
        try:
            client.flush()
        except Exception:
            pass


# ── Span helpers ──────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_span(
    name: str,
    metadata: Optional[dict] = None,
) -> tuple:
    """
    Open a span as a direct child of the active trace.

    Returns (span, start_monotonic).
    Always call end_span() with the returned values.
    If Langfuse is disabled, returns (None, time.monotonic()) — end_span is a no-op.
    """
    trace = get_trace()
    start_mono = time.monotonic()
    if trace is None:
        return None, start_mono

    try:
        span = trace.span(
            name=name,
            metadata=metadata or {},
            start_time=_now(),
        )
        return span, start_mono
    except Exception as exc:
        logger.debug(f"new_span '{name}' error: {exc}")
        return None, start_mono


def end_span(
    span,
    start_monotonic: float,
    metadata: Optional[dict] = None,
    error: Optional[str] = None,
):
    """
    Close a span with duration_ms, status, and optional extra metadata.
    Pass error=<message> to mark the span as failed.
    """
    if span is None:
        return

    duration_ms = int((time.monotonic() - start_monotonic) * 1000)
    extra: dict = {"duration_ms": duration_ms}
    if duration_ms > SLOW_REQUEST_THRESHOLD_MS:
        extra["slow_request"] = True
    if error:
        extra["error"] = error

    try:
        span.end(
            end_time=_now(),
            level="ERROR" if error else "DEFAULT",
            status_message=error,
            metadata={**(metadata or {}), **extra},
        )
    except Exception as exc:
        logger.debug(f"end_span error: {exc}")


# ── Generation helper ─────────────────────────────────────────────────────────

def record_generation(
    name: str,
    model: str,
    input_messages: list,
    output: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    temperature: Optional[float] = None,
    duration_ms: Optional[int] = None,
    metadata: Optional[dict] = None,
    error: Optional[str] = None,
):
    """
    Record a completed LLM generation (Groq call) on the active trace.

    Call this AFTER the API call returns with token counts from response.usage.
    Does nothing if Langfuse is not configured.
    """
    trace = get_trace()
    if trace is None:
        return

    extra: dict = dict(metadata or {})
    if duration_ms is not None:
        extra["duration_ms"] = duration_ms
        if duration_ms > SLOW_REQUEST_THRESHOLD_MS:
            extra["slow_request"] = True

    model_params: dict = {}
    if temperature is not None:
        model_params["temperature"] = temperature

    try:
        trace.generation(
            name=name,
            model=model,
            model_parameters=model_params,
            input=input_messages,
            output=output if not error else None,
            usage={
                "promptTokens": prompt_tokens,
                "completionTokens": completion_tokens,
                "totalTokens": total_tokens,
            },
            metadata=extra,
            level="ERROR" if error else "DEFAULT",
            status_message=error,
        )
    except Exception as exc:
        logger.debug(f"record_generation '{name}' error: {exc}")
