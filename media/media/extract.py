"""One attachment → text, at most once: a Redis lock per attachment, and a finished extraction of the same bytes is
reused (unless it ran on a fallback path)."""
import asyncio
import logging
import os
import tempfile
import time
from typing import Any, Awaitable, Callable

from . import asr, common, doc, image, video

log = logging.getLogger(__name__)
Progress = Callable[[str], Awaitable[None]] | None


async def _voice(att: dict[str, Any], progress: Progress) -> dict[str, Any]:
    src = str(common.file_path(att["sha256"]))
    info = await video.probe(src)
    if info["has_video"]:
        return await video.extract(att, progress=progress)
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "a.wav")
        await video.to_wav(src, wav)
        res = await asr.transcribe(wav)
    dur = res.get("duration_s") or info["duration_s"]
    long = dur > float(common.cfg().get("voice_as_message_s", 120))
    text = asr.as_text(res, timestamps=long)
    return {"kind": "voice", "text": text, "summary": f"語音 {common.mmss(dur)} · {text[:80]}",
            "meta": {"duration_s": dur, "language": res.get("language"), "path": res["path"], "degraded": False}}


async def run_extractor(att: dict[str, Any], progress: Progress = None) -> dict[str, Any]:
    kind = att["kind"]
    if kind == "image":
        return await image.extract(att)
    if kind == "voice":
        return await _voice(att, progress)
    if kind == "video":
        return await video.extract(att, progress=progress)
    if kind in ("doc", "large_doc"):
        return await doc.extract(att)
    raise ValueError(f"unknown kind {kind}")


async def extract(att_id: str, progress: Progress = None, *, wait_s: float = 3600) -> dict[str, Any]:
    """Returns the finished row (status done or error)."""
    r = common.get_redis()
    lock = f"media:lock:{att_id}"
    deadline = time.monotonic() + wait_s
    while True:
        att = await common.get_att(att_id)
        if att is None:
            raise ValueError(f"unknown attachment {att_id}")
        if att["status"] in ("done", "error"):
            return att
        if await r.set(lock, "1", nx=True, ex=int(wait_s)):
            break
        if time.monotonic() > deadline:   # the other extraction is stuck
            raise TimeoutError(f"attachment {att_id} still being extracted")
        await asyncio.sleep(2)            # someone else is extracting it (the eager upload job)
    try:
        cached = await common.find_cached(att["sha256"], att_id)
        if cached:
            fields = {k: cached[k] for k in ("kind", "text", "summary", "meta")}
            fields["meta"] = {**(cached["meta"] or {}), "cached_from": cached["id"]}
            fields["meta"].pop("kb_path", None)
            await common.update_att(att_id, status="done", error=None, **fields)
            return {**att, **fields, "status": "done"}
        await common.update_att(att_id, status="processing")
        t = time.monotonic()
        try:
            res = await run_extractor(att, progress)
        except Exception as e:
            log.exception("extraction of %s (%s) failed", att_id, att["kind"])
            msg = f"{type(e).__name__}: {e}"[:300]
            await common.update_att(att_id, status="error", error=msg)
            return {**att, "status": "error", "error": msg}
        res["meta"] = {**res.get("meta", {}), "seconds": round(time.monotonic() - t, 1)}
        await common.update_att(att_id, status="done", error=None, **res)
        log.info("extracted %s %s → %s in %.1fs (%s)", att_id, att["kind"], res["kind"], res["meta"]["seconds"],
                 res["meta"].get("path"))
        return {**att, **res, "status": "done"}
    finally:
        await r.delete(lock)
