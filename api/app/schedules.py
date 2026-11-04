"""Schedules (phase 11, §4 Cron): rows in the `schedules` table, checked every minute by the scheduler.

kind `brief` = morning brief (app/brief.py), `eval` = nightly answer + retrieval eval (app/tools/answer_eval.py),
`prompt` = run `text` as a background turn in the web thread `web:sched-<name>`, `vault_daily` = the day's summary in
the Obsidian vault's AI/Daily note (app/vault_daily.py, §10a). Built-in rows come from
policy.yaml `schedules.builtin` and are only inserted when missing, so edits made through the API stay.
A schedule whose minute passes while the scheduler is down is skipped, not caught up.
"""
import logging
import re
from datetime import datetime, timezone
from typing import Any

from . import config, db
from .models import Job
from .todos import TZ

log = logging.getLogger(__name__)

KINDS = ("brief", "eval", "prompt", "vault_daily")
_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
_COLS = "id, name, kind, cron, text, enabled, builtin, last_run_at, last_status, created_at"


def _field(spec: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        m = re.fullmatch(r"(\*|\d+(?:-\d+)?)(?:/(\d+))?", part.strip())
        if not m:
            raise ValueError(f"bad cron field {spec!r}")
        rng, step = m.group(1), int(m.group(2) or 1)
        a, b = (lo, hi) if rng == "*" else (int(rng.split("-")[0]), int(rng.split("-")[-1]))
        if m.group(2) and rng.isdigit():  # "5/15" = from 5 to the end, every 15
            b = hi
        if not (lo <= a <= b <= hi) or step < 1:
            raise ValueError(f"cron field {spec!r} out of range {lo}-{hi}")
        out.update(range(a, b + 1, step))
    return out


def parse(expr: str) -> list[set[int]]:
    """5-field cron "m h dom mon dow" (dow 0 or 7 = Sunday), Asia/Taipei."""
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError("cron needs 5 fields: minute hour day-of-month month day-of-week")
    return [_field(p, lo, hi) for p, (lo, hi) in zip(parts, _RANGES)]


def matches(expr: str, when: datetime) -> bool:
    m, h, dom, mon, dow = parse(expr)
    t = when.astimezone(TZ)
    if 7 in dow:
        dow = dow | {0}
    day_ok = t.day in dom
    dow_ok = (t.isoweekday() % 7) in dow
    star_dom, star_dow = expr.split()[2] == "*", expr.split()[4] == "*"
    # standard cron: when both day fields are restricted, either one may match
    day = day_ok and dow_ok if star_dom or star_dow else day_ok or dow_ok
    return t.minute in m and t.hour in h and t.month in mon and day


async def seed() -> None:
    builtin = (config.policy().get("schedules") or {}).get("builtin") or {}
    async with db.get_pool().connection() as conn:
        for name, s in builtin.items():
            await conn.execute("INSERT INTO schedules (name, kind, cron, enabled, builtin) VALUES (%s,%s,%s,%s,true) "
                               "ON CONFLICT (name) DO NOTHING", (name, s["kind"], s["cron"], s.get("enabled", True)))


async def list_() -> list[dict[str, Any]]:
    async with db.get_pool().connection() as conn:
        return await (await conn.execute(f"SELECT {_COLS} FROM schedules ORDER BY id")).fetchall()


async def get(sid: int) -> dict[str, Any] | None:
    async with db.get_pool().connection() as conn:
        return await (await conn.execute(f"SELECT {_COLS} FROM schedules WHERE id = %s", (sid,))).fetchone()


async def create(name: str, cron: str, text: str, enabled: bool = True) -> dict[str, Any]:
    parse(cron)
    async with db.get_pool().connection() as conn:
        return await (await conn.execute(
            f"INSERT INTO schedules (name, kind, cron, text, enabled) VALUES (%s,'prompt',%s,%s,%s) RETURNING {_COLS}",
            (name, cron, text, enabled))).fetchone()


async def update(sid: int, **f: Any) -> dict[str, Any] | None:
    f = {k: v for k, v in f.items() if v is not None and k in ("cron", "text", "enabled")}
    if "cron" in f:
        parse(f["cron"])
    if not f:
        return await get(sid)
    sets = ", ".join(f"{k} = %s" for k in f)
    async with db.get_pool().connection() as conn:
        return await (await conn.execute(f"UPDATE schedules SET {sets} WHERE id = %s RETURNING {_COLS}",
                                         (*f.values(), sid))).fetchone()


async def delete(sid: int) -> bool:
    async with db.get_pool().connection() as conn:
        return (await conn.execute("DELETE FROM schedules WHERE id = %s AND NOT builtin", (sid,))).rowcount > 0


async def _claim(sid: int, minute: datetime) -> bool:
    """One run per schedule per minute, even if two schedulers tick (unique cron + this row lock)."""
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("UPDATE schedules SET last_run_at = %s WHERE id = %s "
                                 "AND (last_run_at IS NULL OR last_run_at < %s)", (minute, sid, minute))
        return cur.rowcount > 0


async def _status(sid: int, status: str) -> None:
    async with db.get_pool().connection() as conn:
        await conn.execute("UPDATE schedules SET last_status = %s WHERE id = %s", (status[:300], sid))


async def run_one(redis, s: dict[str, Any], minute: datetime) -> str:
    pol = config.policy()
    bg = (pol.get("queues") or {}).get("background", {}).get("name", "arq:background")
    if s["kind"] == "brief":
        from . import brief
        return f"ok {await brief.run()}"
    if s["kind"] == "eval":
        await redis.enqueue_job("answer_eval_job", {}, _job_id=f"eval:{minute:%Y%m%d%H%M}", _queue_name=bg)
        return "queued"
    if s["kind"] == "vault_daily":
        if not (pol.get("vault") or {}).get("enabled"):
            return "skipped (vault off)"
        from . import vault_daily
        return f"ok {await vault_daily.run()}"
    if s["kind"] == "prompt":
        from . import pipeline
        from .settings import get_settings

        async def enqueue(fn: str, payload: dict[str, Any], job_id: str, queue: str) -> None:
            await redis.enqueue_job(fn, payload, _job_id=job_id, _queue_name=queue)

        job = Job(channel="cron", mode="background", user_id=get_settings().owner_id, source="schedule",
                  thread_id=f"web:sched-{re.sub(r'[^A-Za-z0-9_-]', '-', s['name'])[:50]}", text=s["text"],
                  dedupe_key=f"sched:{s['id']}:{minute:%Y%m%d%H%M}")
        res = await pipeline.submit(job, enqueue)
        return f"run {res.run_id}" if res.run_id else (res.command_reply or "not run")[:200]
    return f"unknown kind {s['kind']}"


async def schedules_cron(ctx: dict[str, Any]) -> int:
    await seed()  # cheap; picks up builtins added to the hot-reloaded policy
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    n = 0
    for s in await list_():
        try:
            if not s["enabled"] or not matches(s["cron"], now) or not await _claim(s["id"], now):
                continue
            n += 1
            status = await run_one(ctx["redis"], s, now)
        except Exception as e:  # noqa: BLE001 — one broken schedule must not stop the others
            log.exception("schedule %s failed", s["name"])
            status = f"error {type(e).__name__}: {e}"
        else:
            log.info("schedule %s: %s", s["name"], status)
        await _status(s["id"], status)
    return n
