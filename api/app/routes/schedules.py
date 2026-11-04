"""Schedules REST (phase 11), owner-only, under /api/agent. Built-in schedules can be edited but not deleted."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from psycopg.errors import UniqueViolation

from .. import schedules
from ..auth import require_owner

router = APIRouter(prefix="/api/agent")


class ScheduleIn(BaseModel):
    name: str = Field(min_length=1, max_length=60, pattern=r"^[A-Za-z0-9_-]+$")
    cron: str = Field(min_length=9, max_length=100)
    text: str = Field(min_length=1, max_length=4000)
    enabled: bool = True


class SchedulePatch(BaseModel):
    cron: str | None = Field(default=None, min_length=9, max_length=100)
    text: str | None = Field(default=None, min_length=1, max_length=4000)
    enabled: bool | None = None


@router.get("/schedules")
async def list_schedules(owner: str = Depends(require_owner)):
    return {"schedules": await schedules.list_()}


@router.post("/schedules")
async def add_schedule(body: ScheduleIn, owner: str = Depends(require_owner)):
    try:
        return await schedules.create(body.name, body.cron, body.text, body.enabled)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except UniqueViolation as e:
        raise HTTPException(409, "a schedule with that name exists") from e


@router.patch("/schedules/{sid}")
async def update_schedule(sid: int, body: SchedulePatch, owner: str = Depends(require_owner)):
    try:
        row = await schedules.update(sid, **body.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if not row:
        raise HTTPException(404, "no such schedule")
    return row


@router.delete("/schedules/{sid}")
async def delete_schedule(sid: int, owner: str = Depends(require_owner)):
    if not await schedules.delete(sid):
        raise HTTPException(404, "no such schedule (built-in ones can only be disabled)")
    return {"ok": True}


@router.post("/schedules/{sid}/run")
async def run_schedule(sid: int, request: Request, owner: str = Depends(require_owner)):
    """Run now, without touching last_run_at."""
    s = await schedules.get(sid)
    if not s:
        raise HTTPException(404, "no such schedule")
    status = await schedules.run_one(request.app.state.arq, s, datetime.now(timezone.utc))
    return {"status": status}
