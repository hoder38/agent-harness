"""LINE webhook (§4, phase 8): signature check, owner allowlist, quick reply, push for long jobs."""
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from .. import pipeline
from ..channels import line
from ..models import Job
from ..settings import get_settings

log = logging.getLogger(__name__)
router = APIRouter()


async def handle_event(ev: dict[str, Any], enqueue: pipeline.Enqueue) -> None:
    s = get_settings()
    user_id = (ev.get("source") or {}).get("userId")
    if not s.line_owner_user_id or user_id != s.line_owner_user_id:
        log.warning("LINE event from non-owner %s ignored", user_id)
        return
    if ev.get("type") != "message":
        return
    reply_token = ev.get("replyToken")
    msg = ev.get("message") or {}
    if msg.get("type") != "text":
        if reply_token:
            await line.reply(reply_token, "Only text is supported for now (voice/image/video arrive in phase 9).")
        return

    job = Job(channel="line", mode="interactive", user_id=s.owner_id, thread_id=line.thread_id(user_id),
              text=msg.get("text", ""), dedupe_key=f"line:{ev.get('webhookEventId') or msg.get('id')}")
    delivery = {"line_user_id": user_id, "reply_token": reply_token, "reply_ts": time.time()}
    res = await pipeline.submit(job, enqueue, delivery=delivery)
    if res.duplicate:
        return
    if res.command_reply:
        await line.deliver(user_id, res.command_reply, reply_token, time.time(), 50)
        return
    if res.decision and res.decision.queue != "interactive" and reply_token:
        # Long job: acknowledge now with the (free) reply token, deliver the result later via push.
        await line.reply(reply_token, "⏳ Working on it — I'll message you when it's done.")


@router.post("/line/webhook")
async def webhook(request: Request, background: BackgroundTasks,
                  x_line_signature: str | None = Header(default=None)):
    body = await request.body()
    if not line.verify_signature(body, x_line_signature, get_settings().line_channel_secret):
        raise HTTPException(401, "bad signature")
    data = json.loads(body or b"{}")
    for ev in data.get("events", []):
        background.add_task(_safe_handle, ev, request.app.state.enqueue)
    return {"ok": True}


async def _safe_handle(ev: dict[str, Any], enqueue: pipeline.Enqueue) -> None:
    try:
        await handle_event(ev, enqueue)
    except Exception:
        log.exception("LINE event handling failed")
