"""Phase 9 (§5): hand a run to the pipeline once the media-worker has extracted its attachments."""
import logging
import time
from typing import Any

from .. import config, media, pipeline, state
from ..channels import line
from ..models import AttachmentRef, Job

log = logging.getLogger(__name__)


async def _line(delivery: dict[str, Any], text: str, pol: dict[str, Any], *, ack: bool = False) -> None:
    to = delivery.get("line_user_id")
    if not to:
        return
    ttl = float((pol.get("executor") or {}).get("line_reply_token_ttl_s", 50))
    try:
        if ack:   # only spend a fresh reply token; never push an ack
            if delivery.get("reply_token") and time.time() - float(delivery.get("reply_ts") or 0) < ttl:
                await line.reply(delivery["reply_token"], text)
            return
        await line.deliver(to, text, delivery.get("reply_token"), delivery.get("reply_ts"), ttl)
    except Exception:
        log.exception("LINE delivery failed")


async def _index_large_docs(atts: list[dict[str, Any]]) -> None:
    """A large document goes into the owner's knowledge base (§10), so later turns can search all of it."""
    if not (config.policy().get("rag") or {}).get("enabled"):
        return
    from ..rag import indexer

    for a in atts:
        if a["kind"] != "large_doc" or a["status"] != "done" or (a.get("meta") or {}).get("kb_path"):
            continue
        try:
            res = await indexer.add_upload(a["filename"], a.get("text") or "", private=bool(a.get("private")))
            path = f"upload/{res.get('path') or indexer.upload_path(a['filename'], a.get('text') or '')}"
            await media.set_meta(a["id"], kb_path=path)
            a["meta"] = {**(a.get("meta") or {}), "kb_path": path}
        except Exception:
            log.exception("indexing attachment %s failed", a["id"])


async def submit_media_run(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    r = state.get_redis()
    pol = config.policy()
    run_id = payload["run_id"]
    job = Job(**payload["job"])
    delivery: dict[str, Any] = payload.get("delivery") or {}
    if await state.is_cancelled(r, run_id) or (await state.get_run_meta(r, run_id) or {}).get("status") == "cancelled":
        return {"status": "cancelled"}

    atts = await media.get([a.id for a in job.attachments])
    await _index_large_docs(atts)
    user_text = (job.text or "").strip()
    if not user_text and atts and all(a["status"] != "done" for a in atts):
        why = "; ".join(f"{a['filename']}: {a.get('error') or a['status']}" for a in atts)[:300]
        await state.emit(r, run_id, "error", {"message": f"Could not read the attachment ({why})"})
        await state.set_run_meta(r, run_id, status="error")
        await _line(delivery, "抱歉,無法讀取這個檔案。", pol)
        return {"status": "error"}

    job.attachments = [AttachmentRef(id=a["id"], kind=a["kind"], mime=a["mime"]) for a in atts]
    if any(media.is_heavy(a["kind"]) for a in atts):
        job.mode = "background"   # §5.6: video and large documents always run in the background
    full = media.compose(user_text, atts)
    short = media.router_view(user_text, atts)
    job.text = full

    async def enqueue(fn: str, p: dict[str, Any], job_id: str, queue: str) -> None:
        await ctx["redis"].enqueue_job(fn, p, _job_id=job_id, _queue_name=queue)

    res = await pipeline.submit(job, enqueue, delivery=delivery, run_id=run_id, router_text=short)
    if res.command_reply:
        await state.emit(r, run_id, "done", {"text": res.command_reply})
        await state.set_run_meta(r, run_id, status="done")
        await _line(delivery, res.command_reply, pol)
        return {"status": "done"}
    if res.decision and res.decision.queue != "interactive":
        await _line(delivery, "⏳ 檔案已讀取,正在處理,完成後會通知你。", pol, ack=True)
    return {"status": "submitted", "queue": res.decision.queue if res.decision else None}
