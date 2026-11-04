"""`POST /events/{source}` (phase 11): see app/events.py."""
import hashlib
import json

from fastapi import APIRouter, HTTPException, Request

from .. import events

router = APIRouter()
MAX_BODY = 256 * 1024


@router.post("/events/{source}")
async def post_event(source: str, request: Request):
    c, key = events.source_cfg(source), events.secret(source)
    if c is None or not key:
        raise HTTPException(404, "unknown event source")
    body = await request.body()
    if len(body) > MAX_BODY:
        raise HTTPException(413, "event too large")
    sig = request.headers.get("x-hub-signature-256") or request.headers.get("x-signature-256")
    if not events.valid_signature(key, body, sig):
        raise HTTPException(401, "bad signature")
    try:
        payload = json.loads(body or b"{}")
    except ValueError as e:
        raise HTTPException(400, "body must be JSON") from e
    event_id = (request.headers.get("x-event-id") or request.headers.get("x-github-delivery")
                or hashlib.sha256(body).hexdigest())[:200]
    return {"ok": True, "duplicate": not await events.record(source, event_id, payload, c)}
