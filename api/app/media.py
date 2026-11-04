"""L2 input normalization, API side (§5, phase 9): store uploads, attachment rows, compose run text.

Extraction itself runs in the media-worker (`media/`, queue `arq:media`). A run with attachments goes
api → media-worker `media_run` (extract) → agent worker `submit_media_run` (compose + pipeline.submit), all under
one run_id, so the browser follows a single SSE stream.
"""
import hashlib
import json
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import filetype

from . import config, db, state
from .models import AttachmentRef, Job
from .settings import get_settings

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS attachments (
    id          TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    sha256      TEXT NOT NULL,
    mime        TEXT NOT NULL,
    kind        TEXT NOT NULL,                  -- voice | image | video | doc | large_doc
    size        BIGINT NOT NULL,
    filename    TEXT NOT NULL DEFAULT '',
    channel     TEXT NOT NULL,
    thread_id   TEXT,
    private     BOOLEAN NOT NULL DEFAULT false,
    status      TEXT NOT NULL DEFAULT 'pending', -- pending | processing | done | error
    text        TEXT,                           -- extracted text (sections: transcript, OCR, description, ...)
    summary     TEXT,                           -- one line for lists and LINE
    meta        JSONB NOT NULL DEFAULT '{}',    -- duration, pages, path (gpu|cpu|cloud), degraded, timings
    error       TEXT
);
CREATE INDEX IF NOT EXISTS attachments_sha ON attachments (sha256);
"""

HEAVY_DEFAULT = ("video", "large_doc")
_DOC_MIMES = {   # what the media-worker can convert (Docling backends + PDF); keep in step with media/media/doc.py
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
_BLOCK_RE = re.compile(r'\n*<untrusted source="attachment" kind="(\w+)" name="([^"]*)"[^>]*>.*?</untrusted>', re.S)


class MediaError(ValueError):
    """Rejected upload; the message is shown to the owner."""


def cfg() -> dict[str, Any]:
    return config.policy().get("media") or {}


def files_root() -> Path:
    return Path(get_settings().files_root)


def file_path(sha: str) -> Path:
    return files_root() / sha[:2] / sha


def sniff(head: bytes, filename: str = "") -> tuple[str, str]:
    """(mime, kind) from the first bytes, not the extension. Office files are zips: only their content says which."""
    ft = filetype.guess(head)
    mime = ft.mime if ft else ""
    if mime.startswith("image/"):
        return mime, "image"
    if mime.startswith("audio/"):
        return mime, "voice"
    if mime.startswith("video/"):
        return mime, "video"   # the media-worker turns a video without a picture stream into voice
    if mime in _DOC_MIMES:
        return mime, "doc"
    if not ft and b"\x00" not in head[:8192]:
        try:
            head[:8192].decode("utf-8")
        except UnicodeDecodeError as e:
            if e.start < len(head[:8192]) - 4:   # a multi-byte char cut at the end is fine
                raise MediaError("unsupported file type") from e
        return "text/plain", "doc"
    raise MediaError(f"unsupported file type ({mime or 'unknown'})")


def max_bytes(channel: str) -> int:
    lim = (cfg().get("max_mb") or {})
    return int(float(lim.get(channel, lim.get("web", 50))) * 1024 * 1024)


async def store(chunks: AsyncIterator[bytes], *, channel: str, thread_id: str | None, filename: str = "",
                private: bool = False) -> dict[str, Any]:
    """Stream an upload to the files volume (sha256 name, deduplicated) and insert its `attachments` row."""
    limit = max_bytes(channel)
    root = files_root()
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    h, size, head = hashlib.sha256(), 0, b""
    fd, tmp = tempfile.mkstemp(dir=root / "tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            async for chunk in chunks:
                size += len(chunk)
                if size > limit:
                    raise MediaError(f"file too large (limit {limit // (1024 * 1024)} MB)")
                if len(head) < 8192:
                    head += chunk[:8192 - len(head)]
                h.update(chunk)
                f.write(chunk)
        if not size:
            raise MediaError("empty file")
        mime, kind = sniff(head, filename)
        sha = h.hexdigest()
        dest = file_path(sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            os.unlink(tmp)
        else:
            os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    att = {"id": uuid.uuid4().hex, "sha256": sha, "mime": mime, "kind": kind, "size": size,
           "filename": _clean_name(filename) or f"{kind}.{mime.split('/')[-1][:8]}", "channel": channel,
           "thread_id": thread_id, "private": private, "status": "pending"}
    async with db.get_pool().connection() as conn:
        await conn.execute(
            """INSERT INTO attachments (id, sha256, mime, kind, size, filename, channel, thread_id, private)
               VALUES (%(id)s, %(sha256)s, %(mime)s, %(kind)s, %(size)s, %(filename)s, %(channel)s, %(thread_id)s,
                       %(private)s)""", att)
    await db.audit("upload", kind, {"id": att["id"], "mime": mime, "size": size, "channel": channel})
    return att


def _clean_name(name: str) -> str:
    name = os.path.basename((name or "").replace("\\", "/"))
    return re.sub(r'[\x00-\x1f"<>]', "", name)[:120]


async def get(ids: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("SELECT * FROM attachments WHERE id = ANY(%s)", (ids,))
        rows = {r["id"]: r for r in await cur.fetchall()}
    return [rows[i] for i in ids if i in rows]


def is_heavy(kind: str) -> bool:
    return kind in set(config.policy().get("heavy_media_kinds") or HEAVY_DEFAULT)


# ---------------------------------------------------------------- run text
def _block(att: dict[str, Any], limit: int) -> str:
    body = re.sub(r"</?untrusted", lambda m: m.group(0).replace("<", "&lt;"), (att.get("text") or "").strip(),
                  flags=re.I)   # the file must not close its own block
    if att.get("status") != "done":
        body = f"(could not be processed: {att.get('error') or att.get('status')})"
    elif not body:
        body = "(no text or speech found)"
    elif len(body) > limit:
        note = (" The full text was added to the owner's knowledge base." if (att.get("meta") or {}).get("kb_path")
                else "")
        body = body[:limit].rstrip() + f"\n… (truncated, {len(body)} chars in total.{note})"
    name = (att.get("filename") or "").replace('"', "'")
    return f'<untrusted source="attachment" kind="{att["kind"]}" name="{name}">\n{body}\n</untrusted>'


def voice_is_message(att: dict[str, Any], user_text: str) -> bool:
    """A short voice note with no typed text is the owner speaking the request (LINE voice, short recordings);
    anything else (meetings, long recordings) is content to work on and stays untrusted."""
    if user_text.strip() or att.get("kind") != "voice" or att.get("status") != "done":
        return False
    dur = float((att.get("meta") or {}).get("duration_s") or 0)
    return att.get("channel") == "line" or dur <= float(cfg().get("voice_as_message_s", 120))


def compose(user_text: str, atts: list[dict[str, Any]], *, limit: int | None = None) -> str:
    """Run text: the owner's words, then one <untrusted> block per attachment (data, never instructions)."""
    c = cfg()
    user_text = (user_text or "").strip()
    if len(atts) == 1 and voice_is_message(atts[0], user_text):
        return (atts[0].get("text") or "").strip()
    total = int(limit or c.get("max_chars", 8000))
    per = max(200, total // max(1, len(atts)))
    if not user_text:
        prompts = c.get("default_prompts") or {}
        kinds = [a["kind"] for a in atts]
        kind = kinds[0] if len(set(kinds)) == 1 else "mixed"
        user_text = prompts.get(kind) or prompts.get("mixed") or "請說明附件的內容。"
    return "\n\n".join([user_text, *(_block(a, per) for a in atts)])


_KIND_LABEL = {"image": "圖片", "voice": "錄音", "video": "影片", "doc": "文件", "large_doc": "大型文件"}


def router_view(user_text: str, atts: list[dict[str, Any]]) -> str:
    """What the router / decision log / thread context see: the instruction plus 📎 labels, never file content
    (a long timeline or OCR drowns the instruction, and file text must not steer routing)."""
    content = compose(user_text, atts)
    if "<untrusted" not in content:   # a voice note that *is* the message
        return content[:int(cfg().get("router_chars", 1200))]
    labels = [f"📎 {_KIND_LABEL.get(a['kind'], a['kind'])}: {a.get('filename') or a['id']}" for a in atts]
    return "\n".join([strip_blocks(content), *labels])


def display_text(content: str) -> str:
    """Thread history: the owner's words plus one 📎 line per attachment, not the extracted text."""
    if "<untrusted source=\"attachment\"" not in content:
        return content
    names = [f"📎 {m.group(2)}" for m in _BLOCK_RE.finditer(content)]
    rest = _BLOCK_RE.sub("", content).strip()
    return "\n".join([rest, *names]) if rest else "\n".join(names)


def strip_blocks(content: str) -> str:
    """The owner's own words only (memory extraction must not learn "facts" from documents or OCR)."""
    return _BLOCK_RE.sub("", content or "").strip()


# ---------------------------------------------------------------- run submission
def media_queue(pol: dict[str, Any] | None = None) -> str:
    pol = pol or config.policy()
    return (pol.get("queues") or {}).get("media", {}).get("name", "arq:media")


async def start_run(job: Job, arq, *, delivery: dict[str, Any] | None = None, deduped: bool = False) -> str | None:
    """Accept a run with attachments: dedupe now (unless the caller did), then the media-worker extracts and
    hands it to the pipeline. Returns the run_id, or None for a duplicate."""
    r = state.get_redis()
    if not deduped and not await state.dedupe(r, job.dedupe_key):
        return None
    run_id = uuid.uuid4().hex
    q = media_queue()
    await state.set_run_meta(r, run_id, thread_id=job.thread_id, queue=q, status="processing", channel=job.channel)
    await state.emit(r, run_id, "status", {"status": "processing",
                                           "attachments": [a.model_dump() for a in job.attachments]})
    await arq.enqueue_job("media_run", {"run_id": run_id, "job": job.model_dump(mode="json"),
                                        "delivery": delivery or {}},
                          _job_id=f"media:{run_id}", _queue_name=q)
    return run_id


# ---------------------------------------------------------------- LINE: photo, then the question as a separate message
# LINE delivers "photo" and "what is this?" as two events. A file-only run waits `line_text_wait_s` for a text;
# a text in that window becomes its instruction, a later one (within `line_follow_s`) gets the file attached again.
_CLOSED = "\x00"


async def note_line_media(thread_id: str, run_id: str, att_ids: list[str]) -> None:
    await state.get_redis().set(f"line:media:{thread_id}", json.dumps({"run_id": run_id, "att_ids": att_ids}),
                                ex=int(cfg().get("line_follow_s", 60)))


async def line_followup(thread_id: str) -> dict[str, Any] | None:
    """The file just sent in this LINE thread (taken once: only the next text is about it)."""
    raw = await state.get_redis().getdel(f"line:media:{thread_id}")
    return json.loads(raw) if raw else None


async def offer_text(run_id: str, text: str, delivery: dict[str, Any]) -> bool:
    """Give a waiting file-only run its instruction. False if that run has already been submitted."""
    val = json.dumps({"text": text, "delivery": delivery}, ensure_ascii=False)
    return bool(await state.get_redis().set(f"media:text:{run_id}", val, nx=True, ex=3600))


async def take_text(run_id: str) -> dict[str, Any] | None:
    """Called once when the run is submitted: closes the slot, returns a text offered before that."""
    r = state.get_redis()
    if await r.set(f"media:text:{run_id}", _CLOSED, nx=True, ex=3600):
        return None
    raw = await r.get(f"media:text:{run_id}")
    return json.loads(raw) if raw and raw != _CLOSED else None


async def refs(ids: list[str], thread_id: str) -> list[AttachmentRef]:
    """Validate attachment ids for a run: they must exist and belong to this thread (or to none yet)."""
    rows = await get(ids)
    if len(rows) != len(set(ids)):
        raise MediaError("unknown attachment")
    out = []
    for a in rows:
        if a["thread_id"] not in (None, thread_id):
            raise MediaError("attachment belongs to another chat")
        out.append(AttachmentRef(id=a["id"], kind=a["kind"], mime=a["mime"]))
    return out


async def claim_for_thread(ids: list[str], thread_id: str, private: bool) -> None:
    async with db.get_pool().connection() as conn:
        await conn.execute(
            "UPDATE attachments SET thread_id = %s, private = private OR %s, updated_at = now() WHERE id = ANY(%s)",
            (thread_id, private, ids))


async def set_meta(att_id: str, **meta: Any) -> None:
    async with db.get_pool().connection() as conn:
        await conn.execute("UPDATE attachments SET meta = meta || %s::jsonb, updated_at = now() WHERE id = %s",
                           (json.dumps(meta, ensure_ascii=False, default=str), att_id))


def public(att: dict[str, Any]) -> dict[str, Any]:
    keep = ("id", "kind", "mime", "size", "filename", "status", "summary", "error")
    out = {k: att.get(k) for k in keep}
    m = att.get("meta") or {}
    out["meta"] = {k: m[k] for k in ("duration_s", "pages", "frames", "path", "degraded", "kb_path") if k in m}
    return out
