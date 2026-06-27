"""Lightweight pipeline tracing.

Always emits structured per-stage timing logs. If `opentelemetry` is
installed, it also opens a real span per stage; otherwise it's a no-op span.
No hard dependency — Talon runs identically with or without OTel.

Usage:
    from src.obs.trace import stage

    with stage("market_scan", tickers=len(tickers)) as sp:
        results = scan(...)
        sp["results"] = len(results)   # attributes added to the log/span
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("talon.trace")

# Optional OpenTelemetry — resolved once at import.
try:  # pragma: no cover - depends on optional install
    from opentelemetry import trace as _otel_trace

    _TRACER = _otel_trace.get_tracer("talon")
except Exception:  # opentelemetry not installed
    _TRACER = None


@contextmanager
def stage(name: str, **attrs):
    """Time a pipeline stage. Yields a mutable dict for extra attributes.

    Logs `stage=<name> ms=<elapsed> <attrs>` at INFO, and records an OTel
    span when available. Never raises from instrumentation itself.
    """
    data: dict = dict(attrs)
    start = time.monotonic()
    span_cm = _TRACER.start_as_current_span(name) if _TRACER is not None else None
    span = span_cm.__enter__() if span_cm is not None else None
    try:
        yield data
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        try:
            if span is not None:
                for k, v in data.items():
                    try:
                        span.set_attribute(k, v)
                    except Exception:
                        pass
                span.set_attribute("elapsed_ms", round(elapsed_ms, 1))
        finally:
            if span_cm is not None:
                try:
                    span_cm.__exit__(None, None, None)
                except Exception:
                    pass
        extra = " ".join(f"{k}={v}" for k, v in data.items())
        logger.info("stage=%s ms=%.1f %s", name, elapsed_ms, extra)
