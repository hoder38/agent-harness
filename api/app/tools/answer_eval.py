"""Nightly eval (phase 11): end-to-end answers (config/answer_eval.yaml) + retrieval (config/rag_eval.yaml).

  docker compose exec agent-worker-bg python -m app.tools.answer_eval [-v] [--no-rag]

Each run is stored in `eval_runs`; a drop against the previous run (answer passes down by `eval.max_answer_drop`
or more, or KB recall@5 down by more than `eval.max_recall_drop`) sends a LINE alert (`eval:regression`).
"""
import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from .. import config, db, pipeline, state
from ..models import Job
from ..settings import get_settings
from ..todos import TZ
from . import rag_eval

FINAL = {"done", "error", "cancelled", "awaiting_input"}


def cfg() -> dict[str, Any]:
    return config.policy().get("eval") or {}


def _file(name: str) -> str:
    return str(Path(get_settings().policy_path).with_name(name))


def passes(text: str, expect: list[str]) -> bool:
    low = text.lower()
    return any(e.lower() in low for e in expect)


async def _wait(r, run_id: str, timeout_s: float) -> str:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        status = ((await state.get_run_meta(r, run_id)) or {}).get("status")
        if status in FINAL:
            return status
        await asyncio.sleep(2)
    await state.request_cancel(r, run_id)
    return "timeout"


async def _done_event(r, run_id: str) -> dict[str, Any]:
    for _, f in reversed(await r.xrange(f"run:{run_id}")):
        if f.get("type") == "done":
            return json.loads(f["data"])
    return {}


async def run_case(redis, i: int, case: dict[str, Any], stamp: str, timeout_s: float) -> dict[str, Any]:
    r = state.get_redis()
    expect = case["expect"] if isinstance(case["expect"], list) else [case["expect"]]

    async def enqueue(fn: str, payload: dict[str, Any], job_id: str, queue: str) -> None:
        await redis.enqueue_job(fn, payload, _job_id=job_id, _queue_name=queue)

    tid = f"eval:{stamp}-{i}"
    t0 = time.monotonic()
    res = await pipeline.submit(Job(channel="cron", mode="interactive", user_id=get_settings().owner_id,
                                    thread_id=tid, text=case["q"], source="eval",
                                    dedupe_key=f"eval:{stamp}:{i}"), enqueue)
    out: dict[str, Any] = {"q": case["q"], "expect": expect, "run_id": res.run_id,
                           "route": res.decision.route if res.decision else None}
    if not res.run_id:
        return {**out, "status": "not run", "passed": False, "text": res.command_reply or ""}
    status = await _wait(r, res.run_id, timeout_s)
    if status == "awaiting_input":
        await state.clear_pending(r, tid)
    done = await _done_event(r, res.run_id)
    text = done.get("text") or ""
    v = done.get("verify") or {}
    return {**out, "status": status, "passed": status == "done" and passes(text, expect), "text": text[:500],
            "model": done.get("model"), "cost_usd": done.get("cost_usd", 0), "ms": int((time.monotonic() - t0) * 1000),
            "verify_passed": v.get("passed"), "verify_score": v.get("score"), "escalations": v.get("escalations", 0)}


async def _previous() -> dict[str, Any] | None:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("SELECT n, passed, metrics FROM eval_runs ORDER BY id DESC LIMIT 1")
        return await cur.fetchone()


def regressions(prev: dict[str, Any] | None, passed: int, recall: float | None, c: dict[str, Any]) -> list[str]:
    if not prev:
        return []
    out = []
    if prev["passed"] - passed >= int(c.get("max_answer_drop", 2)):
        out.append(f"answers passed {prev['passed']} → {passed}")
    old = (prev.get("metrics") or {}).get("recall@5")
    if recall is not None and old is not None and old - recall > float(c.get("max_recall_drop", 0.05)):
        out.append(f"KB recall@5 {old:.2f} → {recall:.2f}")
    return out


async def run(redis, *, rag: bool = True, verbose: bool = False) -> dict[str, Any]:
    c = cfg()
    if await state.get_killswitch(state.get_redis()) == "all":
        return {"skipped": "killswitch=all"}
    cases = config.load_yaml(_file("answer_eval.yaml")).get("cases") or []
    stamp = datetime.now(TZ).strftime("%Y%m%d%H%M")
    detail = []
    for i, case in enumerate(cases, 1):
        d = await run_case(redis, i, case, stamp, float(c.get("case_timeout_s", 240)))
        detail.append(d)
        if verbose or not d["passed"]:
            print(f"{'✓' if d['passed'] else '✗'} {d['q']} [{d.get('status')}, {d.get('model')}] {d['text'][:120]!r}")
    passed = sum(d["passed"] for d in detail)
    metrics: dict[str, Any] = {
        "answer_pass": round(passed / max(len(detail), 1), 3),
        "verify_fail": sum(1 for d in detail if d.get("verify_passed") is False),
        "escalated": sum(1 for d in detail if d.get("escalations")),
        "cost_usd": round(sum(float(d.get("cost_usd") or 0) for d in detail), 4),
        "ms_avg": int(sum(d.get("ms", 0) for d in detail) / max(len(detail), 1)),
    }
    if rag and (config.policy().get("rag") or {}).get("enabled"):
        rcases = config.load_yaml(_file("rag_eval.yaml")).get("cases") or []
        rr = await rag_eval.run(rcases, 5, True, False)
        metrics["recall@5"], metrics["mrr"] = rr["recall@5"], rr["mrr"]
    prev = await _previous()
    bad = regressions(prev, passed, metrics.get("recall@5"), c)
    metrics["regressions"] = bad
    async with db.get_pool().connection() as conn:
        await conn.execute("INSERT INTO eval_runs (n, passed, metrics, detail) VALUES (%s,%s,%s,%s)",
                           (len(detail), passed, Jsonb(metrics), Jsonb(detail)))
    if bad:
        from ..jobs.alerts import send_alert
        await send_alert("eval:regression", "Nightly eval regression: " + "; ".join(bad) +
                         f" ({passed}/{len(detail)} answers passed)")
    return {"n": len(detail), "passed": passed, **metrics}


async def answer_eval_job(ctx: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return await run(ctx["redis"], rag=(payload or {}).get("rag", True))


async def main() -> None:
    from arq import create_pool
    from arq.connections import RedisSettings

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--no-rag", action="store_true")
    a = ap.parse_args()
    await db.open_db()
    redis = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    try:
        print(json.dumps(await run(redis, rag=not a.no_rag, verbose=a.verbose), ensure_ascii=False))
    finally:
        await redis.aclose()
        await db.close_db()


if __name__ == "__main__":
    asyncio.run(main())
