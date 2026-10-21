"""Host-only endpoints (127.0.0.1:8100), authenticated with the harness token."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..auth import require_harness_token
from ..jobs.alerts import send_alert

router = APIRouter(prefix="/internal", dependencies=[Depends(require_harness_token)])


class AlertBody(BaseModel):
    key: str = Field(max_length=200)
    message: str = Field(max_length=2000)


@router.post("/alert")
async def post_alert(body: AlertBody):
    """Used by host/backup.sh and other host scripts."""
    return {"sent": await send_alert(body.key, body.message)}
