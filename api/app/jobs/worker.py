"""L7/L8: arq tasks — execute one run of the grounded graph (§12a; ReAct when disabled) and stream events (§8)."""
import asyncio
import contextlib
import logging
import re
import time
from typing import Any

import openai
from arq import Retry
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from .. import config, db, inbox, state
from ..channels import line
from ..graphs.coding import build_coding_graph
from ..graphs.grounded import SKIP, build_grounded_graph
from ..graphs.main import build_graph, system_prompt
from ..llm.client import CloudBlocked, make_llm
from ..models import LOCAL_MODELS, Decision, Job
from ..telemetry import setup_tracing
from ..tools import tools_for
from ..tools.sandbox import SandboxClient, coding_config
from . import memory_jobs
from .alerts import check_budget, send_alert

log = logging.getLogger(__name__)

TRANSIENT = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)
CANCEL_POLL_S = 0.25
_CJK = re.compile(r"[\u3400-\u9fff]")


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


def _use_grounded(pol: dict[str, Any]) -> bool:
    return bool((pol.get("grounding") or {}).get("enabled"))


def _tool_fallback(messages: list[Any]) -> str:
    """Local models sometimes end a tool run with an empty message. Our tools return readable one-liners
    ("Added todo #6: ..."), so reply with the last tool result of this turn instead of nothing."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage) and (t := _text(m.content).strip()):
            return t
    return ""


def _gaming_model(decision: Decision, ks: str) -> str:
    """Model for a run that would use the GPU while GAMING (§11); "defer" = wait for the GPU."""
    if decision.private:
        return "local_cpu" if decision.queue == "interactive" else "defer"
    return "cloud_fast" if ks == "off" else "local_cpu"


async def _defer(r, run_id: str, payload: dict[str, Any], queue: str, reason: str) -> dict[str, Any]:
    await state.defer_job(r, run_id, payload, queue)
    await state.set_run_meta(r, run_id, status="deferred")
    await state.emit(r, run_id, "status", {"status": "deferred", "reason": reason})
    return {"status": "deferred"}


async def _graph_input(graph, cfg: dict, payload: dict[str, Any], job: Job, decision: Decision, model: str,
                       grounded: bool) -> Any:
    """New turn, resume of a pending clarify, or continuation of this run after a retry/requeue."""
    run_id = payload["run_id"]
    snap = await graph.aget_state(cfg)
    vals = snap.values or {}
    mine = vals.get("run_id") == run_id or any(getattr(m, "id", None) == run_id for m in vals.get("messages", []))
    if snap.next and not snap.interrupts and mine:
        return None
    resume = payload.get("resume")
    if grounded and resume and snap.interrupts:
        return Command(resume=resume.get("value") or SKIP,
                       update={"run_id": run_id, "model": model, "private": decision.private, "channel": job.channel,
                               "answer_task_type": decision.task_type})
    msg = HumanMessage(payload["text"], id=run_id)
    if not grounded:
        return {"messages": [msg]}
    return {"messages": [msg], "run_id": run_id, "thread_id": job.thread_id, "channel": job.channel,
            "private": decision.private, "model": model, "force_clarify": decision.route == "clarify",
            "task_type": decision.task_type, "confirm_task": decision.confirm_task,
            "correction": payload.get("correction")}


def _line_question(q: dict[str, Any], zh: bool) -> str:
    text = str(q.get("question") or "")
    text += "".join(f"\n{i}. {o}" for i, o in enumerate(q.get("options") or [], 1))
    return text + ("\n\n（回覆答案，或傳 /skip 讓我自行假設）" if zh else "\n\n(Reply with your answer, or /skip to let me assume.)")


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

    gpu = await state.get_gpu_state(r)
    route = decision.route
    if route == "defer":
        if gpu == "GAMING":
            return await _defer(r, run_id, payload, queue, "PRIVATE job waits for GPU FREE")
        route = payload.get("intent") or "chat"
        decision.reasons.append("defer: GPU free again → running locally")

    model = payload.get("model") or decision.model
    if model not in LOCAL_MODELS and ks in ("cloud", "all"):
        model = "local_cpu" if gpu == "GAMING" else "local"
        decision.reasons.append(f"worker: killswitch={ks} → {model}")
        await state.emit(r, run_id, "decision", {**decision.model_dump(), "model": model})
    if model == "local" and gpu == "GAMING":  # queued before the game started
        alt = _gaming_model(decision, ks)
        if alt == "defer":
            return await _defer(r, run_id, payload, queue, "GPU is GAMING: PRIVATE background job waits")
        model = alt
        decision.reasons.append(f"worker: GPU GAMING → {model}")
        await state.emit(r, run_id, "decision", {**decision.model_dump(), "model": model})

    await state.set_run_meta(r, run_id, status="running", model=model)
    ex = pol.get("executor", {})
    gov = pol.get("gpu_governor") or {}
    coding = route == "coding"
    # the assistant route needs its own tools (todos), which the grounded graph does not have
    grounded = _use_grounded(pol) and not coding and route != "assistant"
    usage = {"input": 0, "output": 0}
    streamed = False
    cancelled = False
    gaming_stop = False
    asked: dict[str, Any] | None = None
    started = time.monotonic()
    try:
        llm = make_llm(model, ks)
        max_turns = int(ex.get("max_history_turns", 20))
        if coding:
            async def notify(text: str) -> None:
                await state.emit(r, run_id, "token", text)

            async def is_cancelled() -> bool:
                return await state.is_cancelled(r, run_id)

            graph = build_coding_graph(SandboxClient(), config=coding_config(), notify=notify,
                                       cancelled=is_cancelled, checkpointer=ctx["checkpointer"])
            answer_nodes: set[str] = set()
        elif grounded:
            graph = build_grounded_graph(llm, policy=pol, checkpointer=ctx["checkpointer"], max_turns=max_turns)
            answer_nodes = {"compose"}
        else:
            tools = tools_for(route, private=decision.private, local=model in LOCAL_MODELS)
            graph = build_graph(
                llm, tools,
                system=system_prompt(route, job.channel, model, web=any(t.name == "fetch_page" for t in tools)),
                max_turns=max_turns,
                tool_limits={k: int(v) for k, v in (pol.get("tool_limits") or {}).items()},
                checkpointer=ctx["checkpointer"],
            )
            answer_nodes = {"agent"}
        cfg = {
            "configurable": {"thread_id": job.thread_id},
            "recursion_limit": int(ex.get("recursion_limit", 25)),
            "run_name": f"run:{route}",
            "metadata": {"run_id": run_id, "route": route, "model": model, "channel": job.channel},
            "tags": [route, model, job.channel],
        }
        inp = await _graph_input(graph, cfg, payload, job, decision, model, grounded or coding)
        if coding and isinstance(inp, dict):
            inp["killswitch"] = ks
        sem = (state.GpuSemaphore(r, int(pol.get("gpu_concurrency", 2)))
               if model == "local" and not coding else contextlib.nullcontext())
        short_s = float(gov.get("inflight_short_s", 20))
        last_check = 0.0
        stream_id = None
        echo = ""  # "我理解為：…" line from the grounded compose node, sent before each (re)written answer
        async with sem:
            async for mode, data in graph.astream(inp, cfg, stream_mode=["messages", "updates", "custom"]):
                now = time.monotonic()
                if now - last_check > CANCEL_POLL_S:
                    last_check = now
                    if await state.is_cancelled(r, run_id):
                        cancelled = True
                        break
                    if (model == "local" and not coding and now - started > short_s
                            and await state.get_gpu_state(r) == "GAMING"):
                        gaming_stop = True  # long local run: hand the GPU back (§11)
                        break
                if mode == "updates":
                    if isinstance(data, dict) and data.get("__interrupt__"):
                        asked = dict(data["__interrupt__"][0].value or {})
                    continue
                if mode == "custom":
                    if isinstance(data, dict) and data.get("echo"):
                        echo = str(data["echo"])
                    continue
                chunk, cmeta = data
                if isinstance(chunk, AIMessageChunk):
                    if chunk.usage_metadata:
                        usage["input"] += chunk.usage_metadata.get("input_tokens", 0)
                        usage["output"] += chunk.usage_metadata.get("output_tokens", 0)
                    if cmeta.get("langgraph_node") in answer_nodes and (piece := _text(chunk.content)):
                        if grounded and stream_id is not None and chunk.id != stream_id:
                            await state.emit(r, run_id, "reset", {"reason": "answer rewritten"})
                        if chunk.id != stream_id and echo:
                            await state.emit(r, run_id, "token", echo + "\n\n")
                        stream_id = chunk.id
                        streamed = True
                        await state.emit(r, run_id, "token", piece)
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

    if gaming_stop:
        alt = _gaming_model(decision, ks)
        if streamed:
            await state.emit(r, run_id, "reset", {"reason": f"GPU GAMING: continuing on {alt}"})
        if alt == "defer":
            return await _defer(r, run_id, payload, queue, "GPU is GAMING: PRIVATE run paused")
        await state.emit(r, run_id, "status", {"status": "requeued", "reason": "GPU GAMING", "model": alt})
        await state.set_run_meta(r, run_id, status="queued")
        await _requeue(ctx, {**payload, "model": alt}, queue, 0)
        return {"status": "requeued"}

    elapsed = int((time.monotonic() - started) * 1000)
    snap = await graph.aget_state(cfg)
    vals = snap.values or {}

    if asked is not None or snap.interrupts:
        q = asked or dict(snap.interrupts[0].value or {})
        await state.emit(r, run_id, "interrupt", {**q, "run_id": run_id, "model": model, "usage": usage,
                                                  "cost_usd": cost, "ms": elapsed})
        await state.set_run_meta(r, run_id, status="awaiting_input")
        g_cfg = pol.get("grounding") or {}
        ttl = (int((coding_config().get("coding") or {}).get("approval_ttl_s", 259200)) if coding
               else int(g_cfg.get("pending_ttl_s", 86400)))
        await state.set_pending(r, job.thread_id, run_id=run_id, round_=int(q.get("round") or 1), ttl_s=ttl,
                                kind="coding" if coding else "grounded")
        await inbox.notify_question(
            run_id=run_id, thread_id=job.thread_id, channel=job.channel, q=q, round_=int(q.get("round") or 1),
            ttl_s=ttl, private=await state.is_thread_private(r, job.thread_id), coding=bool(coding),
            background=job.mode == "background", repo=(vals.get("coding") or {}).get("repo"))
        if not coding:
            await state.clarify_ctx_from_question(r, job.thread_id, str(q.get("question") or "")[:1000],
                                                  int((pol.get("clarify") or {}).get("context_ttl_s", 86400)))
        await db.update_outcome(run_id, {"status": "awaiting_input", "usage": usage, "ms": elapsed, "model": model,
                                         "clarify_round": q.get("round")}, cost)
        if job.channel == "line":
            await _deliver_line(job, decision, delivery,
                                _line_question(q, bool(_CJK.search(str(q.get("question") or "")))), pol)
        return {"status": "awaiting_input"}

    final = ""
    for m in reversed(vals.get("messages", [])):
        if isinstance(m, AIMessage):
            final = _text(m.content)
            break
    if not final.strip():
        final = _tool_fallback(vals.get("messages", []))
    done: dict[str, Any] = {"text": final, "model": model, "usage": usage, "cost_usd": cost, "ms": elapsed}
    outcome: dict[str, Any] = {"status": "done", "usage": usage, "ms": elapsed, "model": model}
    if grounded and vals.get("answer"):
        ans, b = vals["answer"], vals.get("budget") or {}
        summary = {"verdict": ans.get("verdict"), "citations": ans.get("citations"),
                   "unverified": len(ans.get("unverified") or []), "budget": b}
        done["grounding"] = summary
        outcome["grounding"] = summary
    if coding and vals.get("coding"):
        c = vals["coding"]
        summary = {k: c.get(k) for k in ("repo", "task_id", "status", "branch", "usage", "attempt", "revisions")}
        summary["tests_passed"] = (c.get("tests") or {}).get("passed")
        done["coding"] = summary
        outcome["coding"] = summary
    await state.emit(r, run_id, "done", done)
    await state.set_run_meta(r, run_id, status="done")
    await db.update_outcome(run_id, outcome, cost)
    if grounded or coding:
        await state.clear_clarify_ctx(r, job.thread_id)
    elif decision.route == "clarify" and final:
        await state.set_clarify_question(r, job.thread_id, final[:1000])
    if job.channel == "line":
        await _deliver_line(job, decision, delivery, final, pol)
    if not coding and ctx.get("redis") is not None:
        try:
            await memory_jobs.after_turn(ctx["redis"], run_id=run_id, thread_id=job.thread_id,
                                         user_text=payload.get("text") or "", answer=final,
                                         private=decision.private, channel=job.channel)
        except Exception:
            log.exception("memory after_turn failed")
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
