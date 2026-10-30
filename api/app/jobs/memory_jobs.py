"""Background memory/RAG jobs (§10): per-turn memory extraction, episode summaries, docs indexing, TTL sweep.

All LLM work here uses the local model (memories may hold PRIVATE text) on the background queue, waits while the
GPU is GAMING or the kill switch is `all`, and shares the GPU semaphore with interactive runs.
"""
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage

from .. import config, db, inbox, memory, state
from ..llm.client import make_llm
from ..rag import indexer

log = logging.getLogger(__name__)

ACTIVE = "memory:active_threads"  # zset thread_id -> last turn time; drained by the episode cron


def _bg_queue(pol: dict[str, Any]) -> str:
    return (pol.get("queues") or {}).get("background", {}).get("name", "arq:background")


async def after_turn(redis, *, run_id: str, thread_id: str, user_text: str, answer: str, private: bool,
                     channel: str) -> None:
    """Called by run_agent after a finished turn: queue fact extraction and mark the thread for an episode."""
    pol = config.policy()
    mc = pol.get("memory") or {}
    if not mc.get("enabled"):
        return
    await redis.zadd(ACTIVE, {thread_id: time.time()})
    if mc.get("auto_extract", True) and memory.worth_extracting(user_text):
        await redis.enqueue_job("extract_memory", {
            "run_id": run_id, "thread_id": thread_id, "text": user_text[:4000], "answer": answer[:2000],
            "private": private, "channel": channel}, _job_id=f"mem:{run_id}", _queue_name=_bg_queue(pol))


async def _wait_reason(r) -> str | None:
    if await state.get_killswitch(r) == "all":
        return "killswitch=all"
    if await state.get_gpu_state(r) == "GAMING":
        return "GPU GAMING"
    return None


async def _later(ctx: dict[str, Any], fn: str, payload: dict[str, Any], job_id: str, why: str) -> dict[str, Any]:
    pol = config.policy()
    await ctx["redis"].enqueue_job(fn, payload, _job_id=f"{job_id}:{int(time.time())}", _queue_name=_bg_queue(pol),
                                   _defer_by=float(pol.get("defer_recheck_s", 300)))
    return {"status": "deferred", "reason": why}


async def extract_memory(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    r = state.get_redis()
    if why := await _wait_reason(r):
        return await _later(ctx, "extract_memory", payload, f"mem:{payload['run_id']}", why)
    pol = config.policy()
    llm = make_llm("local", await state.get_killswitch(r), timeout_s=120)
    async with state.GpuSemaphore(r, int(pol.get("gpu_concurrency", 2))):
        ops = await memory.extract(llm, "local", user_text=payload["text"], answer=payload.get("answer", ""),
                                   private=bool(payload.get("private")),
                                   source={"thread_id": payload.get("thread_id"), "run_id": payload["run_id"],
                                           "channel": payload.get("channel")})
    if ops:
        brief = [{k: o.get(k) for k in ("op", "namespace", "key")} for o in ops]  # no text: it may be private
        log.info("memory ops for %s: %s", payload["run_id"], brief)
        await db.audit("memory", "extract", {"ops": brief, "private": bool(payload.get("private"))},
                       run_id=payload["run_id"])
        for o in ops:
            await inbox.memory_written(o["op"], o["namespace"], o["key"], o.get("text", ""),
                                       private=bool(payload.get("private")), source="extraction")
    return {"status": "done", "ops": brief if ops else []}


async def thread_messages(checkpointer, thread_id: str) -> list[Any]:
    tup = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    return list(((tup.checkpoint or {}).get("channel_values") or {}).get("messages") or []) if tup else []


async def summarize_episode(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    r = state.get_redis()
    tid = payload["thread_id"]
    if why := await _wait_reason(r):
        return await _later(ctx, "summarize_episode", payload, f"episode:{tid}", why)
    msgs = await thread_messages(ctx["checkpointer"], tid)
    if sum(isinstance(m, HumanMessage) for m in msgs) < int((config.policy().get("memory") or {}).get(
            "episode_min_turns", 2)):
        return {"status": "skipped", "reason": "too short"}
    pol = config.policy()
    llm = make_llm("local", await state.get_killswitch(r), timeout_s=180)
    async with state.GpuSemaphore(r, int(pol.get("gpu_concurrency", 2))):
        text = await memory.summarize_episode(llm, "local", msgs, thread_id=tid,
                                              private=await state.is_thread_private(r, tid))
    return {"status": "done" if text else "empty"}


async def index_docs_job(ctx: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return await indexer.index_docs()


# ---------------------------------------------------------------- scheduler crons
async def index_docs_cron(ctx: dict[str, Any]) -> None:
    rc = config.policy().get("rag") or {}
    if not rc.get("enabled"):
        return
    every = max(1, int(rc.get("index_interval_min", 5)))
    if time.gmtime().tm_min % every:
        return
    await indexer.index_docs()


async def episodes_cron(ctx: dict[str, Any]) -> None:
    """Threads idle for `memory.episode_idle_s` since their last turn get (re)summarised once."""
    pol = config.policy()
    mc = pol.get("memory") or {}
    if not mc.get("enabled"):
        return
    r = ctx["redis"]
    due = await r.zrangebyscore(ACTIVE, "-inf", time.time() - float(mc.get("episode_idle_s", 1800)))
    for raw in due:
        tid = raw.decode() if isinstance(raw, bytes) else raw
        await r.zrem(ACTIVE, tid)
        await r.enqueue_job("summarize_episode", {"thread_id": tid}, _job_id=f"episode:{tid}:{int(time.time())}",
                            _queue_name=_bg_queue(pol))


async def sweep_memory_ttl(ctx: dict[str, Any]) -> None:
    if (n := await memory.get_store().sweep_ttl()):
        log.info("memory TTL sweep removed %s items", n)


async def inbox_cleanup_cron(ctx: dict[str, Any]) -> None:
    if n := await inbox.cleanup():
        log.info("inbox cleanup removed %s items", n)
