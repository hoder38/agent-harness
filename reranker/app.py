"""Cross-encoder reranker for RAG (ARCHITECTURE.md §10): POST /rerank {query, documents, top_n?} -> scores.

CPU only, so it never competes with games or Ollama for VRAM. The model is downloaded once through the
egress proxy into the `hf` volume (HF_HOME). The harness falls back to RRF order when this service is down.
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))
THREADS = int(os.environ.get("RERANK_THREADS", "8"))

log = logging.getLogger("reranker")
_model = None
_lock = asyncio.Lock()


def _load():
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(THREADS)
    return CrossEncoder(MODEL, max_length=MAX_LENGTH, device="cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logging.basicConfig(level=logging.INFO)
    t = time.monotonic()
    _model = await asyncio.to_thread(_load)
    log.info("loaded %s in %.1fs", MODEL, time.monotonic() - t)
    yield


app = FastAPI(lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    documents: list[str] = Field(max_length=100)
    top_n: int | None = None


@app.get("/healthz")
async def healthz():
    if _model is None:
        raise HTTPException(503, "loading")
    return {"ok": True, "model": MODEL}


@app.post("/rerank")
async def rerank(req: RerankRequest):
    if _model is None:
        raise HTTPException(503, "loading")
    if not req.documents:
        return {"results": [], "ms": 0}
    t = time.monotonic()
    async with _lock:  # one batch at a time; torch already uses THREADS cores
        scores = await asyncio.to_thread(
            lambda: _model.predict([(req.query, d) for d in req.documents], activation_fn=_sigmoid(),
                                   batch_size=16))
    order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
    if req.top_n:
        order = order[: req.top_n]
    return {"results": [{"index": i, "score": round(float(scores[i]), 5)} for i in order],
            "ms": int((time.monotonic() - t) * 1000)}


def _sigmoid():
    import torch

    return torch.nn.Sigmoid()
