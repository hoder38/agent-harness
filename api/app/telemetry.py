import logging

from .settings import get_settings

log = logging.getLogger(__name__)
_done = False


def setup_tracing(service: str) -> None:
    """Send LangChain/LangGraph traces to Arize Phoenix via OpenInference (no-op if unset)."""
    global _done
    s = get_settings()
    if _done or not s.phoenix_collector_endpoint:
        return
    try:
        from phoenix.otel import register

        register(
            project_name=s.phoenix_project,
            endpoint=s.phoenix_collector_endpoint,
            auto_instrument=True,
            batch=True,
            set_global_tracer_provider=True,
        )
        _done = True
        log.info("tracing → %s (%s)", s.phoenix_collector_endpoint, service)
    except Exception:
        log.exception("tracing setup failed; continuing without traces")
