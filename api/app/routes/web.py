"""nodeReact-facing REST + SSE (§4)."""
import json
import re
import time
from typing import Any, AsyncIterator, Literal

from arq.jobs import Job as ArqJob
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from .. import config, db, pipeline, state
from ..auth import require_owner
from ..graphs.main import build_graph
from ..models import Job, KillSwitch, Override

router = APIRouter(prefix="/api/agent")
_THREAD_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _thread_id(tid: str) -> str:
    if not _THREAD_RE.match(tid):
        raise HTTPException(400, "thread id must match [A-Za-z0-9_-]{1,64}")
    return f"web:{tid}"


class RunRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    mode: Literal["interactive", "background"] = "interactive"
    override: Override | None = None
    client_msg_id: str | None = Field(default=None, max_length=128)  # idempotency key from the browser


def sse(event_id: str, type_: str, data: Any) -> str:
    return f"id: {event_id}\nevent: {type_}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def _event_stream(run_id: str, last_id: str = "0") -> AsyncIterator[str]:
    r = state.get_redis()
    yield "retry: 3000\n\n"
    async for eid, type_, data in state.read_events(r, run_id, last_id):
        if type_ == "ping":
            yield ": ping\n\n"
        else:
            yield sse(eid, type_, data)


def _stream_response(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/threads/{tid}/runs")
async def create_run(tid: str, body: RunRequest, request: Request, owner: str = Depends(require_owner)):
    thread_id = _thread_id(tid)
    job = Job(channel="web", mode=body.mode, user_id=owner, thread_id=thread_id, text=body.text,
              override=body.override, dedupe_key=f"web:{body.client_msg_id or time.time_ns()}")
    res = await pipeline.submit(job, request.app.state.enqueue)
    if res.duplicate:
        raise HTTPException(409, "duplicate request")
    if res.command_reply:
        return JSONResponse({"command_reply": res.command_reply})
    if "text/event-stream" in request.headers.get("accept", ""):
        return _stream_response(_event_stream(res.run_id))
    return {"run_id": res.run_id, "thread_id": thread_id, "decision": res.decision.model_dump()}


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request, owner: str = Depends(require_owner)):
    if not await state.get_run_meta(state.get_redis(), run_id):
        raise HTTPException(404, "unknown run")
    last_id = request.headers.get("last-event-id") or request.query_params.get("last_id") or "0"
    return _stream_response(_event_stream(run_id, last_id))


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request, owner: str = Depends(require_owner)):
    r = state.get_redis()
    meta = await state.get_run_meta(r, run_id)
    if not meta:
        raise HTTPException(404, "unknown run")
    await state.request_cancel(r, run_id)
    if meta.get("status") == "queued":  # not started yet: also drop it from the queue
        try:
            await ArqJob(run_id, request.app.state.arq, _queue_name=meta["queue"]).abort(timeout=0.5)
        except Exception:
            pass
        await state.emit(r, run_id, "cancelled", {"reason": "user (queued)"})
    await db.audit("command", "cancel", {}, run_id)
    return {"ok": True}


@router.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, owner: str = Depends(require_owner)):
    raise HTTPException(501, "approvals/resume arrive with phase 7 (Inbox + background approvals)")


def _serialize(m: Any) -> dict[str, Any] | None:
    if isinstance(m, HumanMessage):
        return {"id": m.id, "role": "user", "content": m.content}
    if isinstance(m, AIMessage):
        return {"id": m.id, "role": "assistant", "content": m.content,
                "tool_calls": [t["name"] for t in m.tool_calls] or None}
    if isinstance(m, ToolMessage):
        return {"id": m.id, "role": "tool", "name": m.name, "content": str(m.content)[:500]}
    return None


@router.get("/threads/{tid}")
async def get_thread(tid: str, request: Request, owner: str = Depends(require_owner)):
    thread_id = _thread_id(tid)
    graph = build_graph(None, [], checkpointer=request.app.state.checkpointer)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    msgs = [s for m in (snap.values or {}).get("messages", []) if (s := _serialize(m))]
    return {"thread_id": thread_id, "messages": msgs,
            "private": await state.is_thread_private(state.get_redis(), thread_id)}


class KillSwitchBody(BaseModel):
    state: KillSwitch


@router.get("/status")
async def get_status(owner: str = Depends(require_owner)):
    r = state.get_redis()
    b = await state.get_budget(r, config.policy())
    return {"killswitch": await state.get_killswitch(r), "gpu_state": await state.get_gpu_state(r),
            "budget": b.model_dump(), "policy_version": config.policy().get("version")}


@router.put("/killswitch")
async def put_killswitch(body: KillSwitchBody, owner: str = Depends(require_owner)):
    await state.set_killswitch(state.get_redis(), body.state)
    await db.audit("killswitch", body.state, {"source": "web"})
    return {"killswitch": body.state}
