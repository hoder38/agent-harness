"""Copilot session ingest (§9b): hook-triggered job, debounce helper and the nightly reconciler.

Local model only (sessions can hold anything the owner typed); waits while the GPU is GAMING or the kill switch is
`all`, like the other background memory jobs.
"""
import logging
import time
from pathlib import Path
from typing import Any

from .. import config, copilot_ingest, db, inbox, memory, state
from ..llm.client import make_llm
from ..settings import get_settings
from .memory_jobs import _bg_queue, _later, _wait_reason

log = logging.getLogger(__name__)

STATE = "copilot:ingest:"      # hash per session: last_event_id, ended, facts_done
SCHEDULED = "copilot:scheduled:"


def cfg() -> dict[str, Any]:
    return config.policy().get("copilot_ingest") or {}


def _ingest_queue(pol: dict[str, Any]) -> str:
    return (pol.get("queues") or {}).get("ingest", {}).get("name") or _bg_queue(pol)


async def schedule(redis, session_id: str, *, final: bool) -> str:
    """Queue an ingest for a hook event. agentStop fires after every agent turn, so those share one deferred job
    (the job re-reads the whole file); sessionEnd always runs now and also marks the session as ended."""
    pol = config.policy()
    c = cfg()
    q = _ingest_queue(pol)
    if final:
        await redis.hset(STATE + session_id, "ended", "1")
        await redis.enqueue_job("ingest_copilot", {"session_id": session_id, "final": True},
                                _job_id=f"copilot:{session_id}:end:{int(time.time())}", _queue_name=q)
        return "queued"
    debounce = max(1, int(c.get("debounce_s", 60)))
    if not await redis.set(SCHEDULED + session_id, "1", nx=True, ex=debounce):
        return "debounced"
    await redis.enqueue_job("ingest_copilot", {"session_id": session_id, "final": False},
                            _job_id=f"copilot:{session_id}:{int(time.time())}", _queue_name=q,
                            _defer_by=float(debounce))
    return "queued"


async def ingest_copilot(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    sid = str(payload.get("session_id") or "")
    r = state.get_redis()
    if why := await _wait_reason(r):
        return await _later(ctx, "ingest_copilot", payload, f"copilot:{sid}", why)
    c = cfg()
    d = copilot_ingest.read_session(sid)
    if d is None:
        return {"status": "skipped", "reason": "session not found"}
    st = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
          for k, v in (await r.hgetall(STATE + sid)).items()}
    final = bool(payload.get("final")) or d.ended or st.get("ended") == "1"
    want_facts = final and bool(c.get("extract_facts", True)) and st.get("facts_done") != "1"
    if d.last_event_id == st.get("last_event_id") and not want_facts:
        return {"status": "skipped", "reason": "no new events"}
    if len(d.turns) < int(c.get("min_turns", 1)):
        await r.hset(STATE + sid, mapping={"last_event_id": d.last_event_id})
        return {"status": "skipped", "reason": "too short"}

    pol = config.policy()
    llm = make_llm("local", await state.get_killswitch(r), timeout_s=180)
    ops: list[dict[str, Any]] = []
    async with state.GpuSemaphore(r, int(pol.get("gpu_concurrency", 2))):
        summary = await copilot_ingest.summarize(llm, "local", d)
        if summary:
            await copilot_ingest.write_episode(d, summary)
        if want_facts:
            prompts = "\n\n".join(copilot_ingest.redact(p)[:1000] for p, _ in d.turns if p.strip())[:3000]
            if memory.worth_extracting(prompts):
                ops = await memory.extract(llm, "local", user_text=prompts, answer="", private=False,
                                           source={"thread_id": d.thread_id, "session_id": sid,
                                                   "channel": "copilot_ingest"})
    await copilot_ingest.write_thread(ctx["checkpointer"], d)
    done = {"last_event_id": d.last_event_id}
    if want_facts:
        done["facts_done"] = "1"
    await r.hset(STATE + sid, mapping=done)
    brief = [{k: o.get(k) for k in ("op", "namespace", "key")} for o in ops]   # no text: keys only
    await db.audit("memory", "copilot_ingest", {"session": sid, "turns": len(d.turns), "final": final,
                                                "episode": bool(summary), "ops": brief})
    for o in ops:
        await inbox.memory_written(o["op"], o["namespace"], o["key"], o.get("text", ""), source="copilot")
    return {"status": "done", "turns": len(d.turns), "episode": bool(summary), "ops": brief}


def _recent_sessions(root: Path, now: float, idle_s: float, days: float) -> list[str]:
    out = []
    if not root.is_dir():
        return out
    for p in root.iterdir():
        ev = p / "events.jsonl"
        try:
            age = now - ev.stat().st_mtime
        except OSError:
            continue
        if idle_s <= age <= days * 86400 and copilot_ingest.SESSION_RE.match(p.name):
            out.append(p.name)
    return out


async def reconcile_cron(ctx: dict[str, Any]) -> None:
    """Nightly: ingest idle sessions whose sessionEnd hook never arrived (crash, SIGKILL, hooks not installed)."""
    c = cfg()
    if not c.get("enabled"):
        return
    r = ctx["redis"]
    q = _ingest_queue(config.policy())
    now = time.time()
    n = 0
    for sid in _recent_sessions(Path(get_settings().copilot_sessions_dir), now,
                                float(c.get("reconcile_idle_s", 1800)), float(c.get("reconcile_days", 3))):
        if await r.hget(STATE + sid, "ended") in ("1", b"1"):
            continue
        await r.enqueue_job("ingest_copilot", {"session_id": sid, "final": True},
                            _job_id=f"copilot:{sid}:reconcile:{int(now)}", _queue_name=q)
        n += 1
    if n:
        log.info("copilot reconciler queued %s sessions", n)
