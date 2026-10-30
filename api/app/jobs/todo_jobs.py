"""Todo due reminders (§10, phase 7): LINE push without todo text + a full-title Inbox alert."""
import logging
from datetime import datetime, timezone
from typing import Any

from .. import config, inbox, todos
from ..channels import line
from ..settings import get_settings

log = logging.getLogger(__name__)


def cfg() -> dict[str, Any]:
    return (config.policy().get("todos") or {}).get("reminders") or {}


async def todo_reminders_cron(ctx: dict[str, Any]) -> int:
    """Returns the number of reminders sent. Skipped entirely in quiet hours (they go out right after)."""
    c = cfg()
    if not c.get("enabled", True):
        return 0
    now = datetime.now(timezone.utc)
    if todos.in_quiet_hours(now, c.get("quiet") or {}):
        return 0
    before_s, overdue_s = int(c.get("before_s", 3600)), int(c.get("overdue_s", 3600))
    sent = []
    for t in await todos.reminder_candidates(now, before_s):
        stage = todos.reminder_stage(t, now, before_s, overdue_s)
        if stage and await todos.claim_reminder(t["id"], int(t["reminders_sent"]), stage):
            when = t["due_at"].astimezone(todos.TZ).strftime("%m-%d %H:%M")
            await inbox.add("alert", f"Todo {'overdue' if stage == 2 else 'due soon'}: {t['title']}",
                            f"#{t['id']} due {when}", priority="high" if stage == 2 else "normal",
                            detail={"todo_id": t["id"], "stage": stage})
            sent.append(stage)
    if sent:
        await _push(len(sent), sum(1 for s in sent if s == 2))
    return len(sent)


async def _push(n: int, overdue: int) -> None:
    """Todo titles can hold private details, so LINE only gets a generic line; the titles are in the Inbox."""
    s = get_settings()
    if not (s.line_owner_user_id and s.line_channel_access_token):
        return
    text = f"📌 {n} todo reminder{'s' if n > 1 else ''}" + (f" ({overdue} overdue)" if overdue else "") + \
        ". Open the Inbox on the web to see them."
    try:
        await line.push(s.line_owner_user_id, text)
    except Exception:
        log.exception("LINE todo reminder push failed")
