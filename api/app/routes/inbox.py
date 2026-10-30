"""Inbox REST (§12 L10), owner-only, under /api/agent."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import inbox, state
from ..auth import require_owner
from .web import _public_tid, submit_answer

router = APIRouter(prefix="/api/agent")


@router.get("/inbox")
async def list_inbox(status: Literal["open", "done", "dismissed", "all"] = "open", kind: inbox.Kind | None = None,
                     limit: int = 100, owner: str = Depends(require_owner)):
    items = await inbox.list_(status, kind, limit)
    return {"items": [{**inbox.out(i), "thread_id": _public_tid(i["thread_id"]) if i["thread_id"] else None}
                      for i in items], "counts": await inbox.counts()}


@router.get("/inbox/count")
async def inbox_count(owner: str = Depends(require_owner)):
    return await inbox.counts()


@router.post("/inbox/{item_id}/read")
async def read_item(item_id: int, owner: str = Depends(require_owner)):
    if not await inbox.mark_read(item_id):
        raise HTTPException(404, "unknown item")
    return {"ok": True}


@router.post("/inbox/{item_id}/dismiss")
async def dismiss_item(item_id: int, request: Request, owner: str = Depends(require_owner)):
    """Dismiss an item. A code-change approval is rejected (the graph discards the sandbox, nothing is applied);
    a question is dropped and its run cancelled."""
    item = await inbox.get(item_id)
    if not item:
        raise HTTPException(404, "unknown item")
    if item["status"] != "open":
        return {"ok": True}
    if item["kind"] == "approval" and item["run_id"]:
        if pending := await inbox.restore_pending(state.get_redis(), item):
            await submit_answer(request, owner, item["run_id"], item["thread_id"], pending, None)   # "/skip" = reject
            return {"ok": True, "rejected": True}
    if item["kind"] in inbox.ACTIONABLE and item["run_id"]:
        r = state.get_redis()
        pending = await state.get_pending(r, item["thread_id"])
        if pending and pending.get("run_id") == item["run_id"]:
            await state.clear_pending(r, item["thread_id"])
            await state.clear_clarify_ctx(r, item["thread_id"])
        await state.request_cancel(r, item["run_id"])
    await inbox.set_status(item_id, "dismissed")
    return {"ok": True}


class DismissBulk(BaseModel):
    kind: Literal["alert", "memory"]


@router.post("/inbox/dismiss")
async def dismiss_bulk(body: DismissBulk, owner: str = Depends(require_owner)):
    return {"dismissed": await inbox.dismiss_bulk(body.kind)}


class AnswerBody(BaseModel):
    value: str | None = Field(default=None, max_length=100_000)   # option text or a free reply; empty = skip/reject


@router.post("/inbox/{item_id}/answer")
async def answer_item(item_id: int, body: AnswerBody, request: Request, owner: str = Depends(require_owner)):
    """Approve/reject/answer a paused run from the Inbox, however long ago it paused."""
    item = await inbox.get(item_id)
    if not item:
        raise HTTPException(404, "unknown item")
    if (item.get("detail") or {}).get("calendar") and item["status"] == "open":   # approved calendar proposal
        from ..tools.calendar import execute
        from .. import gcal

        try:
            done = await execute(item["detail"])
        except gcal.CalendarError as e:
            raise HTTPException(502, str(e))
        await inbox.set_status(item_id, "done")
        return {"ok": True, "result": done}
    if item["kind"] not in inbox.ACTIONABLE or not item["run_id"]:
        raise HTTPException(409, "this item has nothing to answer")
    if item["status"] != "open":
        raise HTTPException(409, f"already {item['status']}")
    r = state.get_redis()
    pending = await inbox.restore_pending(r, item)
    if not pending:
        raise HTTPException(409, "the conversation has moved on to another question; answer that one first")
    res = await submit_answer(request, owner, item["run_id"], item["thread_id"], pending, body.value)
    return {"run_id": res.run_id, "thread_id": _public_tid(item["thread_id"])}
