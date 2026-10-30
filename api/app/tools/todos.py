"""Todo tools for the agent (§10). Local database only, so PRIVATE threads get them too. No delete tool."""
from langchain_core.tools import tool

from .. import db, todos


@tool
async def list_todos(status: str = "open") -> str:
    """List the owner's todos (status: open | done | all), due soonest first. Each line starts with its #id."""
    rows = await todos.list_(status if status in ("open", "done", "all") else "open", 50)
    return "\n".join(todos.line(t) for t in rows) or "No todos."


@tool
async def add_todo(title: str, due: str | None = None, notes: str = "") -> str:
    """Add a todo. `due` is optional: YYYY-MM-DD or YYYY-MM-DDTHH:MM in Asia/Taipei (resolve relative dates such
    as "tomorrow" with get_current_time first)."""
    try:
        t = await todos.add(title, notes=notes, due_at=todos.parse_due(due), channel="agent")
    except ValueError as e:
        return f"Not added: {e}"
    await db.audit("todo", "add", {"id": t["id"]})
    return f"Added todo #{t['id']}: {t['title']}"


@tool
async def complete_todo(todo_id: int) -> str:
    """Mark a todo done by its #id (use list_todos to find it)."""
    t = await todos.update(todo_id, status="done")
    if not t:
        return f"No todo #{todo_id}."
    await db.audit("todo", "complete", {"id": todo_id})
    return f"Done: #{t['id']} {t['title']}"


TODO_TOOLS = [list_todos, add_todo, complete_todo]
