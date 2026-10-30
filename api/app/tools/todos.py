"""Todo tools for the agent (§10). Local database only, so PRIVATE threads get them too. No delete tool."""
from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

from .. import db, todos


@tool
async def list_todos(status: str = "open") -> str:
    """List the owner's todos (status: open | done | all), due soonest first. Each line starts with its #id."""
    rows = await todos.list_(status if status in ("open", "done", "all") else "open", 50)
    return "\n".join(todos.line(t) for t in rows) or "No todos."


@tool
async def add_todo(title: str, due: str | None = None, notes: str = "") -> str:
    """Add a todo. `due` is optional: YYYY-MM-DD, MM/DD (e.g. "11/5"; the next such date), YYYY-MM-DDTHH:MM, or "today"/"tomorrow" optionally followed by
    HH:MM (e.g. "tomorrow 07:30"), all in Asia/Taipei. Pass relative days as words; do not compute the date yourself."""
    try:
        due_at = todos.parse_due(due)
        clean = " ".join(title.split())
        # a small local model sometimes repeats the same call within one run
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
        for old in await todos.list_("open", 50):
            if old["title"] == clean and old["due_at"] == due_at and old["created_at"] >= cutoff:
                return f"Already added: todo #{old['id']}: {old['title']}"
        t = await todos.add(title, notes=notes, due_at=due_at, channel="agent")
    except ValueError as e:
        return f"Not added: {e}"
    await db.audit("todo", "add", {"id": t["id"]})
    when = f" (due {t['due_at'].astimezone(todos.TZ):%Y-%m-%d %H:%M})" if t.get("due_at") else " (no due date)"
    return f"Added todo #{t['id']}: {t['title']}{when}"


@tool
async def complete_todo(todo_id: int) -> str:
    """Mark a todo done by its #id (use list_todos to find it)."""
    t = await todos.update(todo_id, status="done")
    if not t:
        return f"No todo #{todo_id}."
    await db.audit("todo", "complete", {"id": todo_id})
    return f"Done: #{t['id']} {t['title']}"


TODO_TOOLS = [list_todos, add_todo, complete_todo]
