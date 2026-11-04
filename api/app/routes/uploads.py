"""Uploads (§4, §5 phase 9): POST the raw file body; the media-worker starts extracting right away."""
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import media, state
from ..auth import require_owner
from .web import _thread_id

router = APIRouter(prefix="/api/agent")


@router.post("/uploads")
async def upload(request: Request, thread: str | None = None, owner: str = Depends(require_owner)):
    """Body = the file (any Content-Type; the real type is sniffed). Optional `X-Filename` (URI-encoded) and
    `?thread=<id>`; a file for a PRIVATE chat is extracted locally only."""
    thread_id = _thread_id(thread) if thread else None
    private = bool(thread_id) and await state.is_thread_private(state.get_redis(), thread_id)
    if (n := request.headers.get("content-length")) and n.isdigit() and int(n) > media.max_bytes("web"):
        raise HTTPException(413, f"file too large (limit {media.max_bytes('web') // (1024 * 1024)} MB)")
    try:
        att = await media.store(request.stream(), channel="web", thread_id=thread_id,
                                filename=unquote(request.headers.get("x-filename") or ""), private=private)
    except media.MediaError as e:
        raise HTTPException(415 if "type" in str(e) else 413 if "large" in str(e) else 400, str(e))
    await request.app.state.arq.enqueue_job("extract_attachment", att["id"], _job_id=f"extract:{att['id']}",
                                            _queue_name=media.media_queue())
    return media.public(att)


@router.get("/uploads/{att_id}")
async def get_upload(att_id: str, owner: str = Depends(require_owner)):
    rows = await media.get([att_id])
    if not rows:
        raise HTTPException(404, "unknown attachment")
    return media.public(rows[0])
