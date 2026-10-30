"""Copilot hook ingest (§9b): `POST /ingest/copilot` on the host-only port, harness token."""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .. import config, copilot_ingest
from ..auth import require_harness_token
from ..jobs import copilot_jobs

router = APIRouter(prefix="/ingest", dependencies=[Depends(require_harness_token)])


class CopilotHook(BaseModel):
    """Hook payload (camelCase events send `sessionId`, PascalCase ones `session_id`; `reason` only exists on sessionEnd)."""
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    session_id: str = Field(validation_alias=AliasChoices("sessionId", "session_id"), max_length=64)
    cwd: str | None = Field(default=None, max_length=500)
    reason: str | None = Field(default=None, max_length=100)
    event: str | None = Field(default=None, max_length=40)   # harness.sh passes the hook name


@router.post("/copilot")
async def ingest_copilot(body: CopilotHook, request: Request):
    if not (config.policy().get("copilot_ingest") or {}).get("enabled"):
        return {"status": "disabled"}
    if not copilot_ingest.SESSION_RE.match(body.session_id):
        raise HTTPException(422, "bad sessionId")
    final = body.event == "sessionEnd" or (body.event is None and body.reason is not None)
    status = await copilot_jobs.schedule(request.app.state.arq, body.session_id, final=final)
    return {"status": status, "thread_id": copilot_ingest.THREAD_PREFIX + body.session_id}
