"""Cross-encoder reranker for RAG (ARCHITECTURE.md §10): POST /rerank {query, documents, top_n?} -> scores.

Runs on the GPU in fp16 (~1.2 GB VRAM, stays loaded while GAMING; 2026-11-09, was CPU at ~4.2 s per query) and
falls back to the CPU when CUDA is unavailable (`RERANK_DEVICE=cpu` forces it). The model is downloaded once through
the egress proxy into the `hf` volume (HF_HOME). The harness falls back to RRF order when this service is down.
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
DEVICE = os.environ.get("RERANK_DEVICE", "auto")   # auto | cuda | cpu

log = logging.getLogger("reranker")
_model = None
_device = "cpu"
_lock = asyncio.Lock()


def _load():
    global _device
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(THREADS)
    if DEVICE != "cpu" and torch.cuda.is_available():
        try:
            m = CrossEncoder(MODEL, max_length=MAX_LENGTH, device="cuda",
                             model_kwargs={"torch_dtype": torch.float16})   # fp16 load: no fp32 copy left reserved
            torch.cuda.empty_cache()
            _device = "cuda"
            return m
        except Exception as e:  # noqa: BLE001  (out of VRAM, unsupported arch): the CPU still works
            log.warning("CUDA load failed, using CPU: %s", e)
            torch.cuda.empty_cache()
    _device = "cpu"
    return CrossEncoder(MODEL, max_length=MAX_LENGTH, device="cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logging.basicConfig(level=logging.INFO)
    t = time.monotonic()
    _model = await asyncio.to_thread(_load)
    log.info("loaded %s on %s in %.1fs", MODEL, _device, time.monotonic() - t)
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
    return {"ok": True, "model": MODEL, "device": _device}


@app.post("/rerank")
async def rerank(req: RerankRequest):
    if _model is None:
        raise HTTPException(503, "loading")
    if not req.documents:
        return {"results": [], "ms": 0}
    t = time.monotonic()
    async with _lock:  # one batch at a time (GPU, or THREADS CPU cores)
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
