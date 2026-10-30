"""Todos (§10): one table used by the web UI (REST), agent tools and Copilot (MCP)."""
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from . import db

TZ = ZoneInfo("Asia/Taipei")
STATUSES = ("open", "done")
_COLS = "id, title, notes, status, due_at, source_channel, created_at, updated_at"


def parse_due(value: str | None) -> datetime | None:
    """ISO date or datetime; naive values are Asia/Taipei. A bare date means end of that day."""
    v = (value or "").strip()
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"bad due date {value!r}: use YYYY-MM-DD or YYYY-MM-DDTHH:MM") from e
    if len(v) <= 10:
        dt = dt.replace(hour=23, minute=59)
    return dt if dt.tzinfo else dt.replace(tzinfo=TZ)


def out(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, **{k: row[k].isoformat() if row.get(k) else None for k in ("due_at", "created_at", "updated_at")}}


async def add(title: str, *, notes: str = "", due_at: datetime | None = None, channel: str = "web") -> dict[str, Any]:
    title = " ".join((title or "").split())[:300]
    if not title:
        raise ValueError("title is required")
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            f"INSERT INTO todos (title, notes, due_at, source_channel) VALUES (%s,%s,%s,%s) RETURNING {_COLS}",
            (title, (notes or "")[:5000], due_at, channel[:30]))
        return await cur.fetchone()


async def get(todo_id: int) -> dict[str, Any] | None:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(f"SELECT {_COLS} FROM todos WHERE id = %s", (todo_id,))
        return await cur.fetchone()


async def list_(status: str = "open", limit: int = 200) -> list[dict[str, Any]]:
    """Open todos: due soonest first (undated last). Done: most recently updated first."""
    where = "" if status == "all" else "WHERE status = %s"
    order = ("ORDER BY (status = 'done'), CASE WHEN status = 'open' THEN due_at END NULLS LAST, "
             "CASE WHEN status = 'done' THEN updated_at END DESC, id")
    args: tuple[Any, ...] = (() if status == "all" else (status,)) + (min(max(limit, 1), 500),)
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(f"SELECT {_COLS} FROM todos {where} {order} LIMIT %s", args)
        return await cur.fetchall()


async def update(todo_id: int, *, title: str | None = None, notes: str | None = None, status: str | None = None,
                 due_at: datetime | None = None, clear_due: bool = False) -> dict[str, Any] | None:
    sets, args = [], []
    if title is not None:
        t = " ".join(title.split())[:300]
        if not t:
            raise ValueError("title is required")
        sets.append("title = %s"); args.append(t)
    if notes is not None:
        sets.append("notes = %s"); args.append(notes[:5000])
    if status is not None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        sets.append("status = %s"); args.append(status)
    if clear_due:
        sets.append("due_at = NULL")
    elif due_at is not None:
        sets.append("due_at = %s"); args.append(due_at)
    if not sets:
        return await get(todo_id)
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            f"UPDATE todos SET {', '.join(sets)}, updated_at = now() WHERE id = %s RETURNING {_COLS}",
            (*args, todo_id))
        return await cur.fetchone()


async def delete(todo_id: int) -> bool:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("DELETE FROM todos WHERE id = %s", (todo_id,))
        return cur.rowcount > 0


def line(t: dict[str, Any]) -> str:
    due = f" (due {t['due_at'].astimezone(TZ).strftime('%Y-%m-%d %H:%M')})" if t.get("due_at") else ""
    mark = "x" if t["status"] == "done" else " "
    return f"[{mark}] #{t['id']} {t['title']}{due}"
