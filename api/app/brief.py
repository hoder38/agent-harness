"""Morning brief (phase 11): weather, today's calendar, due todos, Inbox and yesterday's spend — built in code.

No model writes the brief. Calendar and todo titles are not PRIVATE (§6, 2026-11-05), so:
- the full brief goes to the normal web thread `web:brief` (follow-up questions may use cloud models) and the Inbox;
- LINE only gets the weather, the counts and the spend (same rule as todo reminders).
"""
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.messages import AIMessage

from . import config, db, gcal, inbox, todos, weather
from .channels.notify import notify

log = logging.getLogger(__name__)

THREAD = "web:brief"
WEEKDAY = "一二三四五六日"


def cfg() -> dict[str, Any]:
    return config.policy().get("brief") or {}


async def _events(start: datetime, end: datetime) -> list[dict[str, Any]] | None:
    if not gcal.available():
        return None
    try:
        return await gcal.list_events(start, end)
    except Exception:
        log.exception("brief: calendar failed")
        return None


def _event_line(ev: dict[str, Any]) -> str:
    s, e = ev.get("start", {}), ev.get("end", {})
    if "date" in s:
        when = "整天"
    else:
        a = datetime.fromisoformat(s["dateTime"]).astimezone(todos.TZ)
        b = datetime.fromisoformat(e["dateTime"]).astimezone(todos.TZ)
        when = f"{a:%H:%M}–{b:%H:%M}"
    where = f" @ {ev['location']}" if ev.get("location") else ""
    return f"{when} {' '.join((ev.get('summary') or '(無標題)').split())}{where}"


def due_todos(rows: list[dict[str, Any]], day_end: datetime, now: datetime) -> tuple[list[dict], list[dict]]:
    """(overdue, due later today) among open todos."""
    dated = [t for t in rows if t.get("status") == "open" and t.get("due_at")]
    return ([t for t in dated if t["due_at"] < now], [t for t in dated if now <= t["due_at"] < day_end])


async def _spend(start: datetime, end: datetime) -> float:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("SELECT coalesce(sum(cost_usd), 0) AS usd FROM usage "
                                 "WHERE created_at >= %s AND created_at < %s", (start, end))
        return float((await cur.fetchone())["usd"])


async def _inbox_counts() -> dict[str, int]:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            """SELECT count(*) FILTER (WHERE kind IN ('approval', 'question')) AS waiting,
                      count(*) FILTER (WHERE kind = 'verify') AS verify
               FROM inbox WHERE status = 'open'""")
        return dict(await cur.fetchone())


async def _quiet_alerts(since: datetime) -> list[dict[str, Any]]:
    """Alerts that were only logged (policy `alerts.quiet_prefixes`, e.g. a long job waiting in a queue)."""
    from .jobs.alerts import is_quiet
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("SELECT created_at, key, message FROM alerts WHERE created_at >= %s AND NOT delivered "
                                 "ORDER BY created_at", (since,))
        return [a for a in await cur.fetchall() if is_quiet(a["key"])]


def compose(*, day: datetime, weather_lines: list[str], events: list[dict] | None, overdue: list[dict],
            due: list[dict], counts: dict[str, int], spend: float | None,
            quiet: list[dict] | None = None) -> tuple[str, str]:
    """(full brief for the web, LINE notice without any calendar/todo text)."""
    head = f"☀️ 早安！{day:%m/%d}（{WEEKDAY[day.weekday()]}）"
    full, notice = [head, "", "🌤 天氣"], [head]
    full += [f"- {w}" for w in weather_lines]
    notice += weather_lines
    if events is None:
        full += ["", "📅 今天行程：行事曆未連線"]
    else:
        full += ["", f"📅 今天行程（{len(events)}）"] + [f"- {_event_line(e)}" for e in events]
        full += [] if events else ["- 沒有行程"]
    full += ["", f"✅ 待辦：{len(due)} 個今天到期、{len(overdue)} 個逾期"]
    full += [f"- ⚠️ {t['title']}（逾期，{t['due_at'].astimezone(todos.TZ):%m/%d %H:%M}）" for t in overdue]
    full += [f"- {t['title']}（{t['due_at'].astimezone(todos.TZ):%H:%M}）" for t in due]
    full += ["", f"📥 Inbox：{counts.get('waiting', 0)} 個等你回覆，{counts.get('verify', 0)} 個回答未通過檢查"]
    parts = [f"{len(events)} 個行程" if events is not None else None, f"{len(due)} 個待辦到期",
             f"{len(overdue)} 個逾期" if overdue else None,
             f"{counts.get('waiting', 0)} 個等你回覆" if counts.get("waiting") else None,
             f"{counts.get('verify', 0)} 個回答未通過檢查" if counts.get("verify") else None]
    notice += ["今天：" + "、".join(p for p in parts if p)]
    if quiet:
        full += ["", f"🔕 過去 24 小時記錄的提醒（{len(quiet)}，未推播）"]
        full += [f"- {a['created_at'].astimezone(todos.TZ):%m/%d %H:%M} {a['message']}" for a in quiet[-10:]]
        notice += [f"🔕 {len(quiet)} 個未推播的系統提醒（如任務排隊過久）"]
    if spend is not None:
        full += ["", f"💰 昨日花費 US${spend:.2f}"]
        notice += [f"昨日花費 US${spend:.2f}"]
    notice += ["詳細內容請到網頁「晨報」查看。"]
    return "\n".join(full), "\n".join(notice)


async def _write_thread(text: str, msg_id: str) -> None:
    """Append the brief to the web thread as an assistant message, so it shows in the chat list and can be asked
    about. `as_node="final_check"` leaves the thread at END, like a finished grounded turn."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from .graphs.grounded import build_grounded_graph
    pol = config.policy()
    graph = build_grounded_graph(None, policy=pol, checkpointer=AsyncPostgresSaver(conn=db.get_pool()))
    await graph.aupdate_state({"configurable": {"thread_id": THREAD}},
                              {"messages": [AIMessage(text, id=msg_id)]}, as_node="final_check")
    await db.log_decision(job_id=str(uuid.uuid4()), run_id=msg_id, thread_id=THREAD, channel="cron",
                          mode="background", redacted_text="☀️ 每日晨報", inputs={}, router={},
                          policy_version=str(pol.get("version")), decision={"route": "brief"}, latency_ms=0)


async def run(now: datetime | None = None) -> dict[str, Any]:
    c = cfg()
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(todos.TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day + timedelta(days=1)
    weather_lines = await weather.today(c.get("weather") or [])
    events = await _events(day, day_end) if "calendar" in c.get("sections", ["calendar"]) else None
    overdue, due = due_todos(await todos.list_("open"), day_end, now)
    counts = await _inbox_counts()
    spend = await _spend(day - timedelta(days=1), day) if c.get("spend", True) else None
    quiet = await _quiet_alerts(now - timedelta(hours=24))
    full, notice = compose(day=day, weather_lines=weather_lines, events=events, overdue=overdue, due=due,
                           counts=counts, spend=spend, quiet=quiet)
    msg_id = f"brief-{day:%Y%m%d}"  # the same id replaces a brief re-run on the same day
    await _write_thread(full, msg_id)
    await inbox.dismiss_bulk("brief")
    await inbox.add("brief", f"☀️ 晨報 {day:%m/%d}", full, priority="low", thread_id=THREAD)
    if c.get("line", True):
        await notify(notice)
    return {"events": None if events is None else len(events), "due": len(due), "overdue": len(overdue)}
