"""Calendar tools for the assistant route (§10). Reading is direct; writing is only ever a proposal.

add_event / delete_event never touch Google: they put an approval item in the Inbox (private), and the change is made
only when the owner approves it there (see `execute`, called by the Inbox answer route). Local models only: calendar
content is PRIVATE (config/privacy_rules.yaml), so tools_for hides these from cloud models.
"""
import logging
from datetime import timedelta
from typing import Any

from langchain_core.tools import tool

from .. import db, gcal, inbox, todos
from ..channels import line
from ..settings import get_settings

log = logging.getLogger(__name__)


@tool
async def list_events(start: str = "today", days: int = 7) -> str:
    """List the owner's Google Calendar events from `start` (YYYY-MM-DD, "today" or "tomorrow") for `days` days
    (1-31). Each line starts with the event id. Pass relative days as words; do not compute dates yourself."""
    try:
        a, b = gcal.day_range(start, days)
        events = await gcal.list_events(a, b)
    except (gcal.CalendarError, ValueError) as e:
        return f"Calendar unavailable: {e}"
    if not events:
        return f"No events between {a:%Y-%m-%d} and {b:%Y-%m-%d}."
    return '<untrusted source="calendar">\n' + "\n".join(gcal.line(e) for e in events) + "\n</untrusted>"


async def _propose(action: str, title: str, body: str, data: dict[str, Any]) -> str:
    for it in await inbox.list_("open", "approval", 100):   # a small local model may repeat the same call
        if (it.get("detail") or {}).get("calendar") == {"action": action, **data}:
            return f"Already waiting for approval in the Inbox (item #{it['id']}). Nothing has changed yet."
    item = await inbox.add("approval", title, body, priority="high", private=True,
                           detail={"options": [], "calendar": {"action": action, **data}})
    if item is None:
        return "Could not create the approval item; nothing was changed."
    s = get_settings()
    if s.line_owner_user_id and s.line_channel_access_token:
        try:
            await line.push(s.line_owner_user_id, "📥 A private calendar change is waiting for your approval in the Inbox.")
        except Exception:
            log.exception("LINE calendar push failed")
    return (f"Proposed (Inbox item #{item}). The calendar has NOT been changed yet: the owner must approve it in the "
            "Inbox. Tell the owner that.")


@tool
async def add_event(title: str, start: str, end: str | None = None, notes: str = "") -> str:
    """Propose a new calendar event (needs the owner's approval in the Inbox; it is NOT added immediately).
    `start`/`end` are YYYY-MM-DDTHH:MM or "today"/"tomorrow" followed by HH:MM (e.g. "tomorrow 15:00"), Asia/Taipei.
    `end` defaults to one hour after start. Pass relative days as words; do not compute dates yourself."""
    try:
        a = gcal.when(start)
        b = gcal.when(end) if end else a + timedelta(hours=1)
    except ValueError as e:
        return f"Not proposed: {e}"
    clean = " ".join((title or "").split())[:200]
    if not clean or b <= a:
        return "Not proposed: need a title and an end after the start."
    data = {"title": clean, "start": a.isoformat(), "end": b.isoformat(), "notes": (notes or "")[:2000]}
    await db.audit("calendar", "propose_add", {"start": data["start"]})
    return await _propose("add", f"Add calendar event: {clean}",
                          f"{a:%Y-%m-%d %H:%M}-{b:%H:%M}\n{notes}".strip(), data)


@tool
async def delete_event(event_id: str) -> str:
    """Propose deleting a calendar event by its id from list_events (needs approval in the Inbox; NOT deleted immediately)."""
    try:
        ev = await gcal.get_event(event_id)
    except gcal.CalendarError as e:
        return f"Not proposed: {e}"
    await db.audit("calendar", "propose_delete", {"id": event_id})
    return await _propose("delete", f"Delete calendar event: {ev.get('summary') or '(no title)'}", gcal.line(ev),
                          {"event_id": event_id})


async def execute(detail: dict[str, Any]) -> str:
    """Apply an approved proposal (called by the Inbox answer route). Raises CalendarError on failure."""
    c = detail["calendar"]
    if c["action"] == "add":
        ev = await gcal.insert_event(c["title"], todos.parse_due(c["start"]), todos.parse_due(c["end"]), c.get("notes", ""))
        await db.audit("calendar", "add", {"id": ev.get("id")})
        return "added"
    await gcal.delete_event(c["event_id"])
    await db.audit("calendar", "delete", {"id": c["event_id"]})
    return "deleted"


CALENDAR_TOOLS = [list_events, add_event, delete_event]
