"""LINE webhook (§4, phase 8): signature check, owner allowlist, quick reply, push for long jobs."""
import json
import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from .. import media, pipeline, state
from ..channels import line
from ..models import Job
from ..settings import get_settings

log = logging.getLogger(__name__)
router = APIRouter()


async def handle_event(ev: dict[str, Any], enqueue: pipeline.Enqueue, arq: Any = None) -> None:
    s = get_settings()
    user_id = (ev.get("source") or {}).get("userId")
    if not s.line_owner_user_id or user_id != s.line_owner_user_id:
        log.warning("LINE event from non-owner %s ignored", user_id)
        return
    if ev.get("type") != "message":
        return
    reply_token = ev.get("replyToken")
    msg = ev.get("message") or {}
    job = Job(channel="line", mode="interactive", user_id=s.owner_id, thread_id=line.thread_id(user_id),
              text=msg.get("text", ""), dedupe_key=f"line:{ev.get('webhookEventId') or msg.get('id')}")
    delivery = {"line_user_id": user_id, "reply_token": reply_token, "reply_ts": time.time()}
    if msg.get("type") in MEDIA_TYPES:
        await _handle_media(msg, job, delivery, arq)
        return
    if msg.get("type") != "text":
        if reply_token:
            await line.reply(reply_token, "目前只支援文字、語音、圖片、影片和檔案。")
        return
    res = await pipeline.submit(job, enqueue, delivery=delivery)
    if res.duplicate:
        return
    if res.command_reply:
        await line.deliver(user_id, res.command_reply, reply_token, time.time(), 50)
        return
    if res.decision and res.decision.queue != "interactive" and reply_token:
        # Long job: acknowledge now with the (free) reply token, deliver the result later via push.
        await line.reply(reply_token, "⏳ Working on it — I'll message you when it's done.")


MEDIA_TYPES = {"image", "audio", "video", "file"}


async def _handle_media(msg: dict[str, Any], job: Job, delivery: dict[str, Any], arq: Any) -> None:
    """Phase 9 (§5): store the file, then the media-worker extracts it and hands the run to the pipeline."""
    reply_token = delivery.get("reply_token")
    if arq is None:
        raise RuntimeError("LINE media needs the arq pool")
    if not await state.dedupe(state.get_redis(), job.dedupe_key):   # webhook redelivery: don't download again
        return
    try:
        att = await media.store(line.content(msg["id"], media.max_bytes("line")), channel="line",
                                thread_id=job.thread_id, filename=msg.get("fileName") or "")
    except (media.MediaError, ValueError) as e:
        if reply_token:
            await line.reply(reply_token, f"無法處理這個檔案:{e}")
        return
    except httpx.HTTPError as e:   # LINE content API (expired message, outage)
        log.warning("LINE content download failed for %s: %s", msg.get("id"), e)
        if reply_token:
            await line.reply(reply_token, "無法下載這個檔案,請稍後再傳一次。")
        return
    job.text = ""
    job.attachments = await media.refs([att["id"]], job.thread_id)
    run_id = await media.start_run(job, arq, delivery=delivery, deduped=True)
    if run_id and media.is_heavy(att["kind"]) and reply_token:
        # Video and large files take a while: ack now with the free reply token, push the answer later.
        await line.reply(reply_token, "⏳ 處理中,完成後會通知你。")


@router.post("/line/webhook")
async def webhook(request: Request, background: BackgroundTasks,
                  x_line_signature: str | None = Header(default=None)):
    body = await request.body()
    if not line.verify_signature(body, x_line_signature, get_settings().line_channel_secret):
        raise HTTPException(401, "bad signature")
    data = json.loads(body or b"{}")
    for ev in data.get("events", []):
        background.add_task(_safe_handle, ev, request.app.state.enqueue, request.app.state.arq)
    return {"ok": True}


async def _safe_handle(ev: dict[str, Any], enqueue: pipeline.Enqueue, arq: Any = None) -> None:
    try:
        await handle_event(ev, enqueue, arq)
    except Exception:
        log.exception("LINE event handling failed")
