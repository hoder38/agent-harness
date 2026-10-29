"""Embeddings (Ollama qwen3-embedding) and the CPU reranker client (§10).

Embeddings run on the GPU normally and on the CPU (`num_gpu: 0`) while the GPU is GAMING (§11), so indexing and
recall keep working during games. Both are local services on the internal network: never through the proxy.
"""
import logging
from typing import Any

import httpx

from .. import config, state
from ..settings import get_settings

log = logging.getLogger(__name__)

# Qwen3-Embedding is instruction-tuned on the query side only; documents are embedded as-is.
QUERY_INSTRUCT = "Instruct: Given a question, retrieve notes and passages that answer it\nQuery: "

_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(trust_env=False, timeout=120)
    return _client


def rag_cfg() -> dict[str, Any]:
    return config.policy().get("rag") or {}


async def _gpu_ok() -> bool:
    try:
        return await state.get_gpu_state(state.get_redis()) != "GAMING"
    except Exception:
        return True


async def embed(texts: list[str], *, query: bool = False) -> list[list[float]]:
    if not texts:
        return []
    cfg = rag_cfg()
    opts: dict[str, Any] = {"num_ctx": int(cfg.get("embed_ctx", 2048))}
    if not await _gpu_ok():
        opts["num_gpu"] = 0
    inputs = [QUERY_INSTRUCT + t if query else t for t in texts]
    out: list[list[float]] = []
    batch = int(cfg.get("embed_batch", 16))
    for i in range(0, len(inputs), batch):
        r = await _http().post(f"{get_settings().ollama_url}/api/embed", json={
            "model": cfg.get("embed_model", "qwen3-embedding:0.6b"), "input": inputs[i:i + batch],
            "truncate": True, "options": opts, "keep_alive": cfg.get("embed_keep_alive", "10m")})
        r.raise_for_status()
        out += r.json()["embeddings"]
    return out


async def aembed_documents(texts: list[str]) -> list[list[float]]:
    """LangGraph PostgresStore index `embed` callable (memories are short; same space as queries)."""
    return await embed(texts)


async def rerank(query: str, docs: list[str]) -> list[float] | None:
    """Relevance 0..1 per doc (same order), or None when the reranker is unavailable (caller keeps RRF order)."""
    url = get_settings().reranker_url
    if not url or not docs:
        return None
    try:
        r = await _http().post(f"{url}/rerank", json={"query": query, "documents": docs},
                               timeout=float(rag_cfg().get("rerank_timeout_s", 15)))
        r.raise_for_status()
        scores = [0.0] * len(docs)
        for item in r.json()["results"]:
            scores[item["index"]] = float(item["score"])
        return scores
    except Exception as e:
        log.warning("reranker unavailable: %s", e)
        return None
