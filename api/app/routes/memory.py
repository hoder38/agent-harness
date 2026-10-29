"""Memory review + knowledge-base REST (§10, §19 "Memory review UI"). Owner-only, under /api/agent."""
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import config, db, memory
from ..auth import require_owner
from ..rag import indexer, store

router = APIRouter(prefix="/api/agent")
Namespace = Literal["profile", "procedures", "episodes", "coding_episodes"]


def _out(m: dict) -> dict:
    return {**m, "created_at": m["created_at"].isoformat() if m.get("created_at") else None,
            "updated_at": m["updated_at"].isoformat() if m.get("updated_at") else None}


# ---------------------------------------------------------------- memories
@router.get("/memories")
async def list_memories(namespace: Namespace | None = None, q: str | None = None, limit: int = 100,
                        offset: int = 0, owner: str = Depends(require_owner)):
    items = await memory.search(namespace, q, private=True, limit=min(max(limit, 1), 500), offset=max(offset, 0))
    return {"memories": [_out(m) for m in items], "namespaces": list(memory.NAMESPACES)}


class MemoryIn(BaseModel):
    namespace: Namespace = "profile"
    text: str = Field(min_length=1, max_length=2000)
    private: bool = False
    pinned: bool = False


@router.post("/memories")
async def add_memory(body: MemoryIn, owner: str = Depends(require_owner)):
    key = await memory.remember(body.namespace, body.text, private=body.private, pinned=body.pinned,
                                kind="manual", source={"channel": "web"})
    return _out(await memory.get(body.namespace, key))


class MemoryPatch(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=2000)
    pinned: bool | None = None
    privacy: Literal["normal", "private"] | None = None


@router.put("/memories/{namespace}/{key}")
async def edit_memory(namespace: Namespace, key: str, body: MemoryPatch, owner: str = Depends(require_owner)):
    m = await memory.update(namespace, key, text=body.text, pinned=body.pinned, privacy=body.privacy)
    if not m:
        raise HTTPException(404, "memory not found")
    await db.audit("memory", "edit", {"namespace": namespace, "key": key,
                                      "fields": [k for k, v in body.model_dump().items() if v is not None]})
    return _out(m)


@router.delete("/memories/{namespace}/{key}")
async def delete_memory(namespace: Namespace, key: str, owner: str = Depends(require_owner)):
    await memory.forget(namespace, key)
    await db.audit("memory", "delete", {"namespace": namespace, "key": key})
    return {"ok": True}


# ---------------------------------------------------------------- knowledge base
@router.get("/kb/documents")
async def kb_documents(owner: str = Depends(require_owner)):
    return {"documents": [{**d, "doc_date": d["doc_date"].isoformat() if d["doc_date"] else None,
                           "indexed_at": d["indexed_at"].isoformat()} for d in await store.list_documents()]}


class UploadIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2_000_000)
    private: bool = False


@router.post("/kb/documents")
async def kb_upload(body: UploadIn, owner: str = Depends(require_owner)):
    """Add a note/document to the knowledge base (stored in Postgres; the docs folder itself is read-only)."""
    return await indexer.add_upload(body.title, body.text, private=body.private)


@router.delete("/kb/documents/{doc_id}")
async def kb_delete(doc_id: int, owner: str = Depends(require_owner)):
    """Uploads only: folder documents come back on the next scan, so delete the file instead."""
    if not await store.delete_documents("upload", doc_id=doc_id):
        raise HTTPException(404, "no uploaded document with this id")
    return {"ok": True}


@router.get("/kb/search")
async def kb_search(q: str, k: int = 5, rerank: bool = True, owner: str = Depends(require_owner)):
    hits = await store.search(q, private=True, k=min(max(k, 1), 20), use_rerank=rerank)
    return {"hits": [{"cite": store.cite_label(h), "title": h["title"], "heading": h["heading"],
                      "path": h["path"], "source": h["source"], "privacy_tag": h["privacy_tag"],
                      "score": h["score"], "vec_sim": h["vec_sim"], "text": h["text"]} for h in hits]}


@router.post("/kb/reindex")
async def kb_reindex(request: Request, owner: str = Depends(require_owner)):
    q = (config.policy().get("queues") or {}).get("background", {}).get("name", "arq:background")
    await request.app.state.enqueue("index_docs_job", {}, f"index_docs:{int(time.time()) // 60}", q)
    return {"queued": True}
