"""Todo REST (§10), owner-only, under /api/agent."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .. import todos
from ..auth import require_owner

router = APIRouter(prefix="/api/agent")


class TodoIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    notes: str = Field(default="", max_length=5000)
    due_at: str | None = Field(default=None, max_length=40)


class TodoPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    notes: str | None = Field(default=None, max_length=5000)
    status: Literal["open", "done"] | None = None
    due_at: str | None = Field(default=None, max_length=40)   # "" clears it


def _due(value: str | None):
    try:
        return todos.parse_due(value)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.get("/todos")
async def list_todos(status: Literal["open", "done", "all"] = "open", limit: int = 200,
                     owner: str = Depends(require_owner)):
    return {"todos": [todos.out(t) for t in await todos.list_(status, limit)]}


@router.post("/todos")
async def add_todo(body: TodoIn, owner: str = Depends(require_owner)):
    return todos.out(await todos.add(body.title, notes=body.notes, due_at=_due(body.due_at), channel="web"))


@router.put("/todos/{todo_id}")
async def update_todo(todo_id: int, body: TodoPatch, owner: str = Depends(require_owner)):
    t = await todos.update(todo_id, title=body.title, notes=body.notes, status=body.status,
                           due_at=_due(body.due_at), clear_due=body.due_at == "")
    if not t:
        raise HTTPException(404, "unknown todo")
    return todos.out(t)


@router.delete("/todos/{todo_id}")
async def delete_todo(todo_id: int, owner: str = Depends(require_owner)):
    if not await todos.delete(todo_id):
        raise HTTPException(404, "unknown todo")
    return {"ok": True}
