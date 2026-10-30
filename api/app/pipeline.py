"""L1→L7: dedupe, owner commands, Gate 0/1/2, decide, log, enqueue."""
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import config, db, inbox, memory, state
from .decision import decide
from .gates.gate0 import gate0
from .gates.gate1_qwen import gate1
from .gates.jev_router import route
from .models import Decision, Job

log = logging.getLogger(__name__)

Enqueue = Callable[[str, dict[str, Any], str, str], Awaitable[None]]  # (fn, payload, job_id, queue_name)

_COMMAND_RE = re.compile(r"^\s*/(stop)\s+(cloud|all)\s*$|^\s*/(resume)\s*$|^\s*/(status)\s*$", re.IGNORECASE)
# "/word" at the start (not a path like /etc/x); routing overrides are handled by Gate 0.
_SLASH_RE = re.compile(r"^\s*/([A-Za-z][\w-]*)(?=\s|$)")
_OVERRIDES = {"local", "cloud", "code"}
_SKIP_RE = re.compile(r"^\s*/skip\s*$", re.IGNORECASE)

HELP_TEXT = """Commands
/help - show this list

Routing (put at the start of a message):
/local <message> - answer with the local model only
/cloud <message> - answer with a cloud model
/code <message> - treat as a coding task

Clarifying questions:
/skip - answer without clarifying (I state my assumption)

Memory:
/remember <fact> - save a fact or preference about you (review and edit them on the Memory page)

Owner controls:
/stop cloud - block all cloud model calls
/stop all - block cloud calls and pause background jobs
/resume - turn the kill switch off
/status - show kill switch, GPU state and today's spend"""


@dataclass
class SubmitResult:
    duplicate: bool = False
    command_reply: str | None = None
    run_id: str | None = None
    decision: Decision | None = None
    reasons: list[str] = field(default_factory=list)


async def handle_command(text: str, *, source: str) -> str | None:
    """Owner commands (/stop cloud|all, /resume, /status, /help); any other /word gets the help text."""
    m = _COMMAND_RE.match(text or "")
    if not m:
        word = _SLASH_RE.match(text or "")
        if word and word.group(1).lower() not in _OVERRIDES:
            name = word.group(1).lower()
            return HELP_TEXT if name == "help" else f"Unknown command /{name}.\n\n{HELP_TEXT}"
        return None
    r = state.get_redis()
    if m.group(1):
        new = m.group(2).lower()
        await state.set_killswitch(r, new)  # type: ignore[arg-type]
        await db.audit("killswitch", new, {"source": source})
        return f"🛑 Kill switch = {new}. " + ("Cloud model calls blocked." if new == "cloud" else "Cloud blocked and background queues paused.")
    if m.group(3):
        await state.set_killswitch(r, "off")
        await db.audit("killswitch", "off", {"source": source})
        return "▶️ Kill switch off. Normal operation resumed."
    ks = await state.get_killswitch(r)
    gpu = await state.get_gpu_state(r)
    b = await state.get_budget(r, config.policy())
    spend = ", ".join(f"{p} ${b.spend_usd.get(p, 0):.2f}/${lim:.2f}" for p, lim in b.limits_usd.items())
    return f"killswitch={ks} gpu={gpu} spend today: {spend}"


async def _remember(job: Job, fact: str) -> str:
    """/remember <fact>: private when the thread or the fact is (or the fact holds any secret/PII), so it is only
    recalled on PRIVATE runs (local models)."""
    if not (config.policy().get("memory") or {}).get("enabled"):
        return "Memory is turned off (policy memory.enabled)."
    if not fact:
        return "Usage: /remember <fact>, e.g. /remember I prefer answers in bullet points"
    r = state.get_redis()
    g0 = gate0(Job(**{**job.model_dump(), "text": fact}), config.privacy_rules(),
               thread_private=await state.is_thread_private(r, job.thread_id))
    private = g0.privacy_tag == "PRIVATE" or bool(g0.findings)
    await memory.remember("profile", fact, private=private, kind="explicit",
                          source={"thread_id": job.thread_id, "channel": job.channel})
    await db.audit("memory", "remember", {"private": private, "channel": job.channel})
    return "🧠 Remembered" + (" (private)" if private else "") + f": {fact}"


def _with_clarification(request: str, question: str | None, answer: str) -> str:
    """Routing text for an answer to a clarifying question: the router judges the whole exchange."""
    parts = [request]
    if question:
        parts.append(f"[Assistant asked] {question}")
    parts.append(f"[User answered] {answer}")
    return "\n".join(parts)


def _line_write_request(text: str) -> bool:
    """Clear add-a-todo / add-an-event phrasings (the same cheap rules the assistant route uses), checked in code."""
    from types import SimpleNamespace

    from .tools import required_tool

    t = re.sub(r"^\s*/\w+\s+", "", text)
    return required_tool([SimpleNamespace(name="add_todo"), SimpleNamespace(name="add_event")], t) is not None


async def submit(job: Job, enqueue: Enqueue, *, delivery: dict[str, Any] | None = None) -> SubmitResult:
    r = state.get_redis()
    if not await state.dedupe(r, job.dedupe_key):
        return SubmitResult(duplicate=True)

    pending = await state.get_pending(r, job.thread_id)
    skip = bool(_SKIP_RE.match(job.text or ""))
    if skip and not pending:
        return SubmitResult(command_reply="Nothing to skip: no question is waiting for an answer.")
    if not skip and (fact := memory.parse_remember(job.text or "")) is not None:
        return SubmitResult(command_reply=await _remember(job, fact))
    if not skip and (reply := await handle_command(job.text or "", source=job.channel)):
        return SubmitResult(command_reply=reply)

    if not skip and job.channel == "line" and not pending and _line_write_request(job.text or ""):
        return SubmitResult(command_reply="LINE 只能聊天和收通知,不能新增待辦或行事曆。請到網頁用同樣的句子操作。")

    t0 = time.monotonic()
    pol = config.policy()
    gpu, ks = await state.get_gpu_state(r), await state.get_killswitch(r)
    budget = await state.get_budget(r, pol)

    g0 = gate0(job, config.privacy_rules(), thread_private=await state.is_thread_private(r, job.thread_id))
    if g0.privacy_tag is None:
        g1 = await gate1(g0.text)  # deferred stub → None
        g0.privacy_tag = "PRIVATE" if g1 else "NORMAL"
    private = g0.privacy_tag == "PRIVATE"
    if private:
        await state.mark_thread_private(r, job.thread_id)

    router_text, router_redacted = g0.text, g0.redacted_text
    clarify_cfg = pol.get("clarify") or {}
    clarify_ctx = await state.get_clarify_ctx(r, job.thread_id)
    clarify_rounds = clarify_ctx["rounds"] if clarify_ctx else 0
    prev = None if clarify_ctx else await state.get_prev_msg(r, job.thread_id)
    if prev:
        router_text = f"[Previous message] {prev['text']}\n[Current message] {g0.text}"
        router_redacted = f"[Previous message] {prev['redacted_text']}\n[Current message] {g0.redacted_text}"
    if clarify_ctx:
        router_text = _with_clarification(clarify_ctx["text"], clarify_ctx.get("question"), g0.text)
        # On PRIVATE threads the model saw unredacted text, so its question stays out of the redacted copy.
        router_redacted = _with_clarification(clarify_ctx["redacted_text"],
                                              None if private else clarify_ctx.get("question"),
                                              g0.redacted_text)

    router = await route(
        text=router_text, redacted_text=router_redacted, private=private,
        gpu_state=gpu, killswitch=ks, policy=pol,
    )
    corrected = None
    corr_th = (pol.get("thresholds") or {}).get("correction", 0.6)
    if prev and not pending and router.correction > corr_th:
        # The owner says the last reply misread them: classify both messages as one request, from scratch.
        corrected = {"text": f"{prev['text']}\n{g0.text}", "redacted_text": f"{prev['redacted_text']}\n{g0.redacted_text}"}
        first = router
        router = await route(
            text=corrected["text"], redacted_text=corrected["redacted_text"], private=private,
            gpu_state=gpu, killswitch=ks, policy=pol,
        )
        router.correction = first.correction
    decision = decide(job, g0, router, gpu, budget, pol, killswitch=ks, clarify_rounds=clarify_rounds,
                      resuming=bool(pending))
    decision.reasons = g0.reasons + decision.reasons
    if corrected:
        decision.reasons.append(f"correction {router.correction:.2f} > {corr_th}: re-routed previous + current message")
    if pending:
        await inbox.resolve_run(pending["run_id"], "done")
        # Answer to the grounded graph's question: keep the exchange for the next round's routing;
        # the worker clears it when the turn is done.
        await state.clear_pending(r, job.thread_id)
        if clarify_ctx:
            await state.set_clarify_ctx(r, job.thread_id, text=router_text, redacted_text=router_redacted,
                                        rounds=clarify_rounds, ttl_s=int(clarify_cfg.get("context_ttl_s", 86400)))
        decision.reasons.append(f"resume: answer to the question of run {pending['run_id']}"
                                + (" (skipped)" if skip else ""))
        if pending.get("kind") == "coding":
            # Repo choice / approval for the coding graph: the answer goes back to it, whatever the router said.
            decision.route = "coding"
            decision.queue = "background" if decision.queue == "background" or job.channel == "line" else "coding"
            decision.reasons.append("resume: coding graph question → coding")
    elif decision.route == "clarify":
        # A first clarify keeps just this message (not the previous-message prefix); later rounds keep the exchange.
        await state.set_clarify_ctx(r, job.thread_id,
                                    text=router_text if clarify_ctx else g0.text,
                                    redacted_text=router_redacted if clarify_ctx else g0.redacted_text,
                                    rounds=clarify_rounds + 1,
                                    ttl_s=int(clarify_cfg.get("context_ttl_s", 86400)))
    elif clarify_ctx:
        await state.clear_clarify_ctx(r, job.thread_id)
    if not pending:
        latest = corrected or {"text": g0.text, "redacted_text": g0.redacted_text}
        await state.set_prev_msg(r, job.thread_id, text=latest["text"], redacted_text=latest["redacted_text"],
                                 ttl_s=int(clarify_cfg.get("previous_ttl_s", 3600)))
    latency = int((time.monotonic() - t0) * 1000)

    run_id = uuid.uuid4().hex
    await db.log_decision(
        job_id=job.id, run_id=run_id, thread_id=job.thread_id, channel=job.channel, mode=job.mode,
        redacted_text=g0.redacted_text,
        inputs={
            "job": job.model_dump(mode="json", exclude={"text"}),
            "gate0": g0.model_dump(mode="json", exclude={"text"}),
            "gpu_state": gpu, "budget": budget.model_dump(), "killswitch": ks,
            "clarify_rounds": clarify_rounds,
            **({"resume_of": pending["run_id"]} if pending else {}),
            **({"router_text": router_redacted} if router_redacted != g0.redacted_text else {}),
            **({"correction_text": corrected["redacted_text"]} if corrected else {}),
        },
        router=router.model_dump(mode="json"), policy_version=str(pol.get("version")),
        decision=decision.model_dump(), latency_ms=latency,
    )

    # Non-private threads may later be continued on a cloud model, so they only ever store redacted
    # text. PRIVATE threads are pinned to local models (sticky), so they keep the original.
    model_text = g0.text if private else g0.redacted_text
    if delivery and decision.queue != "interactive":
        # The channel acks immediately with the reply token; the result is pushed later.
        delivery = {**delivery, "reply_token": None}
    payload = {
        "run_id": run_id,
        "job": job.model_dump(mode="json"),
        "text": model_text,
        "decision": decision.model_dump(),
        "intent": router.intent,
        "delivery": delivery or {},
    }
    if corrected:
        payload["correction"] = corrected["text"] if private else corrected["redacted_text"]
    if pending:
        payload["resume"] = {"value": None if skip else model_text, "parent_run_id": pending["run_id"]}
        await state.set_run_meta(r, pending["run_id"], status="answered", answered_by=run_id)
    queue_cfg = pol.get("queues", {}).get(decision.queue, {})
    queue_name = queue_cfg.get("name", f"arq:{decision.queue}")
    await state.set_run_meta(r, run_id, thread_id=job.thread_id, queue=queue_name, status="queued",
                             channel=job.channel)
    await state.emit(r, run_id, "decision", {**decision.model_dump(), "router": router.source, "latency_ms": latency})
    await enqueue("run_agent", payload, run_id, queue_name)
    return SubmitResult(run_id=run_id, decision=decision)
