"""Today page (§19): one read-only aggregate for nodeReact `/Today`. Cheap SQL + Redis only; weather and calendar
are cached briefly, container health comes from the alert watcher's last check (no Docker call per request)."""
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from . import brief, config, db, gcal, inbox, state, todos, weather
from .jobs.alerts import HEALTH_KEY

log = logging.getLogger(__name__)
SEVERITY = {"high": 0, "medium": 1, "low": 2}
WATCHER_STALE_S = 180   # the watcher runs every minute


async def _cached(key: str, ttl_s: int, load) -> Any:
    r = state.get_redis()
    if (raw := await r.get(key)) is not None:
        return json.loads(raw)
    value, ok = await load()
    if ok:
        await r.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_s)
    return value


async def _weather() -> tuple[list[str], bool]:
    lines = await weather.today(brief.cfg().get("weather") or [], timeout_s=5)
    return lines, not any("無法取得" in w for w in lines)


async def _calendar(day: datetime) -> tuple[list[str] | None, bool]:
    if not gcal.available():
        return None, False
    try:
        return [brief._event_line(e) for e in await gcal.list_events(day, day + timedelta(days=1))], True
    except Exception:
        log.exception("today: calendar failed")
        return None, False


def _card(severity: str, title: str, link: str | None = None, key: str | None = None) -> dict[str, Any]:
    return {"severity": severity, "title": title, "link": link, "key": key}


def _issue_severity(key: str) -> str:
    return "high" if key.startswith(("container:", "watcher:", "governor:", "budget:100")) else "medium"


async def status_cards(now: datetime) -> list[dict[str, Any]]:
    """Harness issues, most severe first; empty = all OK. Thresholds come from the alert watcher / policy."""
    r = state.get_redis()
    cards: list[dict[str, Any]] = []
    if (ks := await state.get_killswitch(r)) != "off":
        cards.append(_card("high", f"Kill switch is on: {ks}", key="killswitch"))
    if ov := await r.get("gpu_state_override"):
        cards.append(_card("medium", f"GPU state pinned to {ov} (host/gpu-state.sh auto clears it)", key="gpu:override"))
    health = json.loads(raw) if (raw := await r.get(HEALTH_KEY)) else None
    live: set[str] = set()
    if not health or time.time() - float(health.get("ts") or 0) > WATCHER_STALE_S:
        cards.append(_card("high", "Health watcher has not reported recently (agent-scheduler down?)", key="watcher:stale"))
    else:
        for i in health.get("issues") or []:
            live.add(i["key"])
            cards.append(_card(_issue_severity(i["key"]), i["message"], key=i["key"]))
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("SELECT metrics FROM eval_runs ORDER BY id DESC LIMIT 1")
        last = await cur.fetchone()
        if last and (bad := (last["metrics"] or {}).get("regressions")):
            cards.append(_card("medium", "Nightly eval regression: " + "; ".join(bad), key="eval:regression"))
        cur = await conn.execute("SELECT name, last_status FROM schedules WHERE enabled AND last_status LIKE %s",
                                 ("error%",))
        for s in await cur.fetchall():
            cards.append(_card("medium", f"Schedule {s['name']} failed: {s['last_status']}", key=f"schedule:{s['name']}"))
        # alerts already shown live above are not counted twice
        cur = await conn.execute(
            """SELECT count(*) FILTER (WHERE kind = 'alert' AND created_at >= %s
                                         AND NOT (coalesce(detail->>'key', '') = ANY(%s::text[]))) AS alerts,
                      count(*) FILTER (WHERE kind = 'verify') AS verify
               FROM inbox WHERE status = 'open'""", (now - timedelta(hours=24), list(live)))
        n = await cur.fetchone()
    if n["alerts"]:
        cards.append(_card("low", f"{n['alerts']} unresolved alert(s) in the last 24 h", "/Inbox", "inbox:alert"))
    if n["verify"]:
        cards.append(_card("low", f"{n['verify']} answer(s) failed verification", "/Inbox", "inbox:verify"))
    return sorted(cards, key=lambda c: SEVERITY[c["severity"]])


async def today(now: datetime | None = None) -> dict[str, Any]:
    t0 = time.monotonic()
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(todos.TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    overdue, due = brief.due_todos(await todos.list_("open"), day + timedelta(days=1), now)
    waiting = [i for i in await inbox.list_("open", limit=200) if i["kind"] in ("approval", "question")]
    return {
        "date": day.date().isoformat(),
        "weather": await _cached("today:weather", 1800, _weather),
        "calendar": await _cached(f"today:calendar:{day:%Y%m%d}", 120, lambda: _calendar(day)),
        "todos": {"overdue": [todos.out(t, now) for t in overdue], "due_today": [todos.out(t, now) for t in due]},
        "inbox": [{**inbox.out(i), "thread_id": None} for i in waiting],
        "status": await status_cards(now),
        "ms": int((time.monotonic() - t0) * 1000),
    }
