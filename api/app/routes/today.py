"""Today page REST (§19), owner-only, read-only."""
from fastapi import APIRouter, Depends

from .. import today as today_mod
from ..auth import require_owner

router = APIRouter(prefix="/api/agent")


@router.get("/today")
async def get_today(owner: str = Depends(require_owner)):
    return await today_mod.today()
