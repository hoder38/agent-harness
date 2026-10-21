"""L7/L8: arq tasks — execute one run of the ReAct graph and stream events (§8)."""
import asyncio
import contextlib
import logging
import time
from typing import Any

import openai
from arq import Retry
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from .. import config, db, state
from ..channels import line
from ..graphs.main import build_graph, system_prompt
from ..llm.client import CloudBlocked, make_llm
from ..models import LOCAL_MODELS, Decision, Job
from ..telemetry import setup_tracing
from ..tools import tools_for
from .alerts import check_budget, send_alert

log = logging.getLogger(__name__)

TRANSIENT = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)
CANCEL_POLL_S = 0.25


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return ""


async def startup(ctx: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.INFO)
    setup_tracing("agent-worker")
    pool = await db.open_db()
    saver = AsyncPostgresSaver(conn=pool)
    await saver.setup()
    ctx["checkpointer"] = saver


async def shutdown(ctx: dict[str, Any]) -> None:
    await db.close_db()


async def _requeue(ctx: dict[str, Any], payload: dict[str, Any], queue: str, defer_s: float) -> None:
    await ctx["redis"].enqueue_job(
        "run_agent", payload, _job_id=f"{payload['run_id']}:{int(time.time())}",
        _queue_name=queue, _defer_by=defer_s,
    )


async def _deliver_line(job: Job, decision: Decision, delivery: dict[str, Any], text: str, pol: dict) -> None:
    to = delivery.get("line_user_id")
    if not to:
        return
    ex = pol.get("executor", {})
    if decision.private:
        text = ex.get("private_line_notice", "🔒 Private answer: open the web app.")
    text = text[: int(ex.get("line_max_chars", 4500))]
    try:
        await line.deliver(to, text, delivery.get("reply_token"), delivery.get("reply_ts"),
                           float(ex.get("line_reply_token_ttl_s", 50)))
    except Exception:
        log.exception("LINE delivery failed")


async def run_agent(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    r = state.get_redis()
    pol = config.policy()
    run_id: str = payload["run_id"]
    job = Job(**payload["job"])
    decision = Decision(**payload["decision"])
    delivery: dict[str, Any] = payload.get("delivery") or {}
    meta = await state.get_run_meta(r, run_id) or {}
    queue = meta.get("queue", "arq:interactive")

    if await state.is_cancelled(r, run_id):
        await state.emit(r, run_id, "cancelled", {"reason": "cancelled before start"})
        return {"status": "cancelled"}

    ks = await state.get_killswitch(r)
    if ks == "all" and decision.queue != "interactive":
        await state.emit(r, run_id, "status", {"status": "paused", "reason": "killswitch=all"})
        await _requeue(ctx, payload, queue, float(pol.get("defer_recheck_s", 300)))
        return {"status": "paused"}

    route = decision.route
    if route == "defer":
        if await state.get_gpu_state(r) == "GAMING":
            await state.emit(r, run_id, "status", {"status": "deferred", "reason": "PRIVATE job waits for GPU FREE"})
            await _requeue(ctx, payload, queue, float(pol.get("defer_recheck_s", 300)))
            return {"status": "deferred"}
        route = payload.get("intent") or "chat"
        decision.reasons.append("defer: GPU free again → running locally")

    model = decision.model
    if model not in LOCAL_MODELS and ks in ("cloud", "all"):
        model = "local_cpu" if await state.get_gpu_state(r) == "GAMING" else "local"
        decision.reasons.append(f"worker: killswitch={ks} → {model}")
        await state.emit(r, run_id, "decision", {**decision.model_dump(), "model": model})

    await state.set_run_meta(r, run_id, status="running", model=model)
    ex = pol.get("executor", {})
    usage = {"input": 0, "output": 0}
    streamed = False
    cancelled = False
    started = time.monotonic()
    try:
        llm = make_llm(model, ks)
        graph = build_graph(
            llm, tools_for(route),
            system=system_prompt(route, job.channel, model),
            max_turns=int(ex.get("max_history_turns", 20)),
            checkpointer=ctx["checkpointer"],
        )
        cfg = {
            "configurable": {"thread_id": job.thread_id},
            "recursion_limit": int(ex.get("recursion_limit", 25)),
            "run_name": f"run:{route}",
            "metadata": {"run_id": run_id, "route": route, "model": model, "channel": job.channel},
            "tags": [route, model, job.channel],
        }
        sem = (state.GpuSemaphore(r, int(pol.get("gpu_concurrency", 2)))
               if model == "local" else contextlib.nullcontext())
        last_check = 0.0
        async with sem:
            async for chunk, _meta in graph.astream(
                {"messages": [HumanMessage(payload["text"])]}, cfg, stream_mode="messages"
            ):
                now = time.monotonic()
                if now - last_check > CANCEL_POLL_S:
                    last_check = now
                    if await state.is_cancelled(r, run_id):
                        cancelled = True
                        break
                if isinstance(chunk, AIMessageChunk):
                    if piece := _text(chunk.content):
                        streamed = True
                        await state.emit(r, run_id, "token", piece)
                    if chunk.usage_metadata:
                        usage["input"] += chunk.usage_metadata.get("input_tokens", 0)
                        usage["output"] += chunk.usage_metadata.get("output_tokens", 0)
                elif isinstance(chunk, ToolMessage):
                    await state.emit(r, run_id, "tool", {"name": chunk.name, "content": _text(chunk.content)[:500]})
                    await db.audit("tool_call", chunk.name, {"content": _text(chunk.content)[:2000]}, run_id)
    except asyncio.CancelledError:  # arq abort / job timeout
        await state.emit(r, run_id, "cancelled", {"reason": "aborted"})
        await db.update_outcome(run_id, {"status": "cancelled"})
        raise
    except CloudBlocked as e:
        await state.emit(r, run_id, "error", {"message": str(e)})
        return {"status": "error"}
    except TRANSIENT as e:
        tries = ctx.get("job_try", 1)
        if not streamed and tries < 3:
            await state.emit(r, run_id, "status", {"status": "retrying", "attempt": tries + 1, "error": str(e)[:200]})
            raise Retry(defer=2 ** tries) from e
        return await _fail(r, run_id, job, decision, delivery, pol, e)
    except Exception as e:
        log.exception("run %s failed", run_id)
        return await _fail(r, run_id, job, decision, delivery, pol, e)

    cost = await _account(r, run_id, model, usage, pol)
    if cancelled:
        await state.emit(r, run_id, "cancelled", {"reason": "user"})
        await state.set_run_meta(r, run_id, status="cancelled")
        await db.update_outcome(run_id, {"status": "cancelled", "usage": usage}, cost)
        return {"status": "cancelled"}

    final = ""
    snap = await graph.aget_state(cfg)
    for m in reversed(snap.values.get("messages", [])):
        if isinstance(m, AIMessage):
            final = _text(m.content)
            break
    elapsed = int((time.monotonic() - started) * 1000)
    await state.emit(r, run_id, "done", {"text": final, "model": model, "usage": usage, "cost_usd": cost, "ms": elapsed})
    await state.set_run_meta(r, run_id, status="done")
    await db.update_outcome(run_id, {"status": "done", "usage": usage, "ms": elapsed, "model": model}, cost)
    if job.channel == "line":
        await _deliver_line(job, decision, delivery, final, pol)
    return {"status": "done"}


async def _fail(r, run_id, job, decision, delivery, pol, e: Exception) -> dict[str, Any]:
    msg = f"{type(e).__name__}: {e}"[:500]
    await state.emit(r, run_id, "error", {"message": msg})
    await state.set_run_meta(r, run_id, status="error")
    await db.update_outcome(run_id, {"status": "error", "error": msg})
    if job.channel == "line":
        await _deliver_line(job, decision, delivery, "Sorry, that failed. Check the web app for details.", pol)
    return {"status": "error"}


async def _account(r, run_id: str, model: str, usage: dict[str, int], pol: dict[str, Any]) -> float:
    price = (pol.get("prices_usd_per_mtok") or {}).get(model, {"input": 0, "output": 0})
    cost = (usage["input"] * price.get("input", 0) + usage["output"] * price.get("output", 0)) / 1e6
    provider = (pol.get("providers") or {}).get(model, "local")
    if usage["input"] or usage["output"]:
        await db.log_usage(run_id, provider, model, usage["input"], usage["output"], cost)
    if cost > 0:
        await state.add_spend(r, provider, cost)
        for key, msg in await check_budget(pol):
            await send_alert(key, msg)
    return cost
