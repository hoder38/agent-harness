"""Inbox (§12 L10, §19): what needs the owner's attention, in one list.

- `approval` / `question`: a run paused on an `interrupt()` (coding approval, repo choice, background clarify). The
  LangGraph checkpoint lives in Postgres, so the item stays answerable for days; answering it restores the Redis
  `pending` marker if that expired and resumes the run.
- `alert`: health alerts (also pushed to LINE by `alerts.send_alert`).
- `memory`: every automatic or agent-made memory write, low priority.
Everything here is best effort: a failure to write an inbox row never fails the run that caused it.
"""
import json
import logging
from typing import Any, Literal

from . import config, db, state
from .channels import line
from .settings import get_settings

log = logging.getLogger(__name__)

Kind = Literal["approval", "question", "alert", "memory"]
ACTIONABLE = ("approval", "question")
_COLS = "id, created_at, updated_at, kind, priority, title, body, status, read, private, run_id, thread_id, detail"


def cfg() -> dict[str, Any]:
    return config.policy().get("inbox") or {}


def out(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "created_at": row["created_at"].isoformat(), "updated_at": row["updated_at"].isoformat()}


async def add(kind: str, title: str, body: str = "", *, priority: str = "normal", private: bool = False,
              run_id: str | None = None, thread_id: str | None = None,
              detail: dict[str, Any] | None = None) -> int | None:
    if not cfg().get("enabled", True):
        return None
    try:
        async with db.get_pool().connection() as conn:
            cur = await conn.execute(
                """INSERT INTO inbox (kind, priority, title, body, private, run_id, thread_id, detail)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (run_id) WHERE status = 'open' AND kind IN ('approval', 'question') DO NOTHING
                   RETURNING id""",
                (kind, priority, title[:300], body[:8000], private, run_id, thread_id,
                 json.dumps(detail or {}, ensure_ascii=False, default=str)))
            row = await cur.fetchone()
            return row["id"] if row else None
    except Exception:
        log.exception("failed to write inbox item")
        return None


async def get(item_id: int) -> dict[str, Any] | None:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(f"SELECT {_COLS} FROM inbox WHERE id = %s", (item_id,))
        return await cur.fetchone()


async def list_(status: str = "open", kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    where, args = [], []
    if status != "all":
        where.append("status = %s"); args.append(status)
    if kind:
        where.append("kind = %s"); args.append(kind)
    sql = (f"SELECT {_COLS} FROM inbox {'WHERE ' + ' AND '.join(where) if where else ''} "
           "ORDER BY (priority = 'high') DESC, (priority = 'low'), created_at DESC LIMIT %s")
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(sql, (*args, min(max(limit, 1), 500)))
        return await cur.fetchall()


async def counts() -> dict[str, int]:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            """SELECT count(*) FILTER (WHERE status = 'open') AS open,
                      count(*) FILTER (WHERE status = 'open' AND NOT read) AS unread,
                      count(*) FILTER (WHERE status = 'open' AND kind IN ('approval', 'question')) AS actionable
               FROM inbox""")
        return dict(await cur.fetchone())


async def mark_read(item_id: int) -> bool:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("UPDATE inbox SET read = true, updated_at = now() WHERE id = %s", (item_id,))
        return cur.rowcount > 0


async def set_status(item_id: int, status: str) -> bool:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            "UPDATE inbox SET status = %s, read = true, updated_at = now() WHERE id = %s AND status = 'open'",
            (status, item_id))
        return cur.rowcount > 0


async def dismiss_bulk(kind: str) -> int:
    """Dismiss all open items of a non-actionable kind (approvals and questions are never bulk-dismissed)."""
    if kind in ACTIONABLE:
        raise ValueError("approvals and questions are resolved one by one")
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            "UPDATE inbox SET status = 'dismissed', read = true, updated_at = now() WHERE status = 'open' AND kind = %s",
            (kind,))
        return cur.rowcount


async def resolve_run(run_id: str, status: str = "done") -> None:
    """The run's question was answered (anywhere: web chat, LINE, inbox) or the run was cancelled."""
    try:
        async with db.get_pool().connection() as conn:
            await conn.execute(
                """UPDATE inbox SET status = %s, read = true, updated_at = now()
                   WHERE run_id = %s AND status = 'open' AND kind IN ('approval', 'question')""", (status, run_id))
    except Exception:
        log.exception("failed to resolve inbox item")


async def cleanup() -> int:
    c = cfg()
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            """DELETE FROM inbox WHERE
                 (status <> 'open' AND updated_at < now() - make_interval(days => %s))
                 OR (status = 'open' AND priority = 'low' AND created_at < now() - make_interval(days => %s))""",
            (int(c.get("keep_done_days", 30)), int(c.get("keep_low_days", 14))))
        return cur.rowcount


# ---------------------------------------------------------------- producers
def _brief(text: str, n: int = 160) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


async def notify_question(*, run_id: str, thread_id: str, channel: str, q: dict[str, Any], round_: int,
                          ttl_s: int, private: bool, coding: bool, background: bool,
                          repo: str | None = None) -> None:
    """A run paused on a question. Coding questions always get an item (they wait days); other clarifies only for
    background jobs, because a live chat shows the question inline. LINE push unless the question came via LINE."""
    if not (coding or background):
        return
    approval = q.get("kind") == "approval"
    question = str(q.get("question") or "")
    title = (f"Approve code change in {repo}" if approval and repo else
             f"Question about {repo}" if coding and repo else _brief(question, 80) or "Question")
    item = await add("approval" if approval else "question", title, question, priority="high", private=private,
                     run_id=run_id, thread_id=thread_id,
                     detail={"options": q.get("options") or [], "round": round_, "pending_kind":
                             "coding" if coding else "grounded", "ttl_s": ttl_s, "channel": channel})
    if item is None or channel == "line":
        return
    s = get_settings()
    if s.line_owner_user_id and s.line_channel_access_token:
        text = ("📥 A private task is waiting for you in the Inbox." if private
                else f"📥 {title}\nOpen the Inbox on the web to answer.")
        try:
            await line.push(s.line_owner_user_id, text)
        except Exception:
            log.exception("LINE inbox push failed")


async def memory_written(op: str, namespace: str, key: str, text: str = "", *, private: bool = False,
                         source: str = "agent") -> None:
    if not cfg().get("memory_items", True):
        return
    title = {"add": "Remembered", "update": "Updated memory", "delete": "Forgot"}.get(op, "Memory")
    await add("memory", f"{title} ({namespace})", text, priority="low", private=private,
              detail={"op": op, "namespace": namespace, "key": key, "source": source})


async def restore_pending(r, item: dict[str, Any]) -> dict[str, Any] | None:
    """Make the item's question answerable again if the Redis marker expired. Returns the pending record."""
    pending = await state.get_pending(r, item["thread_id"])
    if pending and pending.get("run_id") == item["run_id"]:
        return pending
    if pending:   # the thread has moved on to another question
        return None
    d = item.get("detail") or {}
    await state.set_pending(r, item["thread_id"], run_id=item["run_id"], round_=int(d.get("round") or 1),
                            ttl_s=3600, kind=d.get("pending_kind") or "coding")
    return await state.get_pending(r, item["thread_id"])
