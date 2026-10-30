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

from .. import config, db, inbox, pipeline, state
from ..channels import line
from ..auth import require_owner
from ..graphs.coding import build_coding_graph
from ..graphs.grounded import build_grounded_graph
from ..graphs.main import build_graph
from ..models import Job, KillSwitch, Override
from ..settings import get_settings

router = APIRouter(prefix="/api/agent")
_THREAD_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


LINE_TID = "line"   # the owner's LINE thread, exposed without leaking the LINE user id
_OVERRIDE_PREFIX = re.compile(r"^/(local|cloud|code)\s+", re.I)


def _thread_id(tid: str) -> str:
    if tid == LINE_TID:
        uid = get_settings().line_owner_user_id
        if not uid:
            raise HTTPException(404, "LINE is not configured")
        return line.thread_id(uid)
    if not _THREAD_RE.match(tid):
        raise HTTPException(400, "thread id must match [A-Za-z0-9_-]{1,64}")
    return f"web:{tid}"


def _public_tid(thread_id: str) -> str:
    return LINE_TID if thread_id.startswith("line:") else thread_id.removeprefix("web:")


def _thread_title(thread_id: str, first_text: str | None) -> str:
    if thread_id.startswith("line:"):
        return "LINE"
    title = _OVERRIDE_PREFIX.sub("", (first_text or "").strip())
    return " ".join(title.split())[:30] or "(untitled)"


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
    pending = await state.get_pending(r, meta.get("thread_id", ""))
    if pending and pending.get("run_id") == run_id:  # waiting for an answer: drop the question
        await state.clear_pending(r, meta["thread_id"])
        await state.clear_clarify_ctx(r, meta["thread_id"])
        await state.set_run_meta(r, run_id, status="cancelled")
    if meta.get("status") == "queued":  # not started yet: also drop it from the queue
        try:
            await ArqJob(run_id, request.app.state.arq, _queue_name=meta["queue"]).abort(timeout=0.5)
        except Exception:
            pass
        await state.emit(r, run_id, "cancelled", {"reason": "user (queued)"})
    await inbox.resolve_run(run_id, "dismissed")
    await db.audit("command", "cancel", {}, run_id)
    return {"ok": True}


class ResumeRequest(BaseModel):
    value: str | None = Field(default=None, max_length=100_000)  # None/empty = /skip (answer with an assumption)


async def submit_answer(request: Request, owner: str, run_id: str, thread_id: str, pending: dict[str, Any],
                        value: str | None):
    """Answer a run's open question (also used by the Inbox); empty = /skip."""
    value = (value or "").strip()
    job = Job(channel="web", mode="interactive", user_id=owner, thread_id=thread_id, text=value or "/skip",
              dedupe_key=f"resume:{run_id}:{pending.get('round')}")
    res = await pipeline.submit(job, request.app.state.enqueue)
    if res.duplicate:
        raise HTTPException(409, "duplicate request")
    return res


@router.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, body: ResumeRequest, request: Request, owner: str = Depends(require_owner)):
    """Answer the open question of a run: a grounded clarify (§12a) or a coding approval (§9a)."""
    r = state.get_redis()
    meta = await state.get_run_meta(r, run_id)
    if not meta:
        raise HTTPException(404, "unknown run")
    thread_id = meta.get("thread_id", "")
    pending = await state.get_pending(r, thread_id)
    if not pending or pending.get("run_id") != run_id:
        raise HTTPException(409, "this run is not waiting for an answer")
    res = await submit_answer(request, owner, run_id, thread_id, pending, body.value)
    if res.command_reply:
        return JSONResponse({"command_reply": res.command_reply})
    if "text/event-stream" in request.headers.get("accept", ""):
        return _stream_response(_event_stream(res.run_id))
    return {"run_id": res.run_id, "thread_id": thread_id, "decision": res.decision.model_dump()}


def _serialize(m: Any) -> dict[str, Any] | None:
    if isinstance(m, HumanMessage):
        return {"id": m.id, "role": "user", "content": m.content}
    if isinstance(m, AIMessage):
        return {"id": m.id, "role": "assistant", "content": m.content,
                "tool_calls": [t["name"] for t in m.tool_calls] or None}
    if isinstance(m, ToolMessage):
        return {"id": m.id, "role": "tool", "name": m.name, "content": str(m.content)[:500]}
    return None


def _fill_empty_replies(msgs: list[dict[str, Any]]) -> None:
    """A local model may end a tool run with an empty message; show the last tool result instead of "…"
    (the live run already does this via _tool_fallback)."""
    last_tool = ""
    for m in msgs:
        if m["role"] == "user":
            last_tool = ""
        elif m["role"] == "tool":
            last_tool = m["content"]
        elif m["role"] == "assistant" and not m["content"] and not m.get("tool_calls") and last_tool:
            m["content"] = last_tool


@router.get("/threads")
async def list_threads(owner: str = Depends(require_owner)):
    prefixes = ("web:",)
    if uid := get_settings().line_owner_user_id:
        prefixes += (f"line:{uid}",)
    rows = await db.list_threads(prefixes)
    r = state.get_redis()
    private = {t["thread_id"] for t in rows if await state.is_thread_private(r, t["thread_id"])}
    return {"threads": [{"id": _public_tid(r["thread_id"]),
                         "title": _thread_title(r["thread_id"], r["first_text"]),
                         "channel": r["thread_id"].split(":", 1)[0],
                         "private": r["thread_id"] in private,
                         "updated_at": r["updated_at"].isoformat()} for r in rows]}


@router.delete("/threads/{tid}")
async def hide_thread(tid: str, owner: str = Depends(require_owner)):
    """Remove from the list only; the history stays and any new message brings it back."""
    await db.set_thread_hidden(_thread_id(tid), True)
    return {"ok": True}


@router.get("/threads/{tid}")
async def get_thread(tid: str, request: Request, owner: str = Depends(require_owner)):
    thread_id = _thread_id(tid)
    r = state.get_redis()
    pol = config.policy()
    pending = await state.get_pending(r, thread_id)
    if pending and pending.get("kind") == "coding":   # the coding graph's interrupt is only visible to it
        graph = build_coding_graph(None, checkpointer=request.app.state.checkpointer)
    elif (pol.get("grounding") or {}).get("enabled"):
        graph = build_grounded_graph(None, policy=pol, checkpointer=request.app.state.checkpointer)
    else:
        graph = build_graph(None, [], checkpointer=request.app.state.checkpointer)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    msgs = [s for m in (snap.values or {}).get("messages", []) if (s := _serialize(m))]
    _fill_empty_replies(msgs)
    question = None
    if pending and snap.interrupts:
        v = snap.interrupts[0].value or {}
        question = {"run_id": pending["run_id"], "round": v.get("round"), "question": v.get("question"),
                    "options": v.get("options") or []}
    return {"thread_id": thread_id, "messages": msgs, "pending": question,
            "private": await state.is_thread_private(r, thread_id)}


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
