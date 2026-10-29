"""Docs-folder indexer (§10): scan DOCS_ROOT, re-index changed files (hash), drop deleted ones.

Runs from the scheduler cron every `rag.index_interval_min` and on demand (`python -m app.tools.index_docs`,
POST /api/agent/kb/reindex). A Redis lock keeps runs from overlapping. Phase 10a adds the vault watcher.
"""
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import config, state
from ..settings import get_settings
from . import chunking, store
from .embed import rag_cfg

log = logging.getLogger(__name__)
LOCK = "rag:index_lock"


def scan(root: str, exts: set[str], max_bytes: int) -> dict[str, Path]:
    """relative posix path → file; hidden files/dirs (.git, .obsidian, .trash) are skipped."""
    base = Path(root)
    out: dict[str, Path] = {}
    if not base.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if f.startswith(".") or p.suffix.lower() not in exts or p.is_symlink():
                continue
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            out[p.relative_to(base).as_posix()] = p
    return out


async def index_docs(root: str | None = None) -> dict[str, Any]:
    cfg = rag_cfg()
    r = state.get_redis()
    if not await r.set(LOCK, "1", nx=True, ex=int(cfg.get("index_lock_s", 1800))):
        return {"status": "busy"}
    t0 = time.monotonic()
    stats: dict[str, Any] = {"added": 0, "updated": 0, "unchanged": 0, "deleted": 0, "errors": [], "private": 0}
    try:
        root = root or get_settings().docs_root
        files = scan(root, {e.lower() for e in cfg.get("extensions") or [".md", ".markdown", ".txt"]},
                     int(cfg.get("max_file_kb", 2048)) * 1024)
        if not Path(root).is_dir():
            stats["errors"].append(f"{root} is not a directory")
            return stats  # never delete the index because the mount is missing
        rules = config.privacy_rules()
        for rel, p in files.items():
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
                mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                res = await store.upsert_document(source="docs", path=rel, raw=raw, rules=rules, mtime=mtime)
                stats[res["status"]] += 1
                if res.get("privacy") == "PRIVATE":
                    stats["private"] += 1
            except Exception as e:
                log.exception("index %s failed", rel)
                stats["errors"].append(f"{rel}: {type(e).__name__}: {e}"[:300])
        stats["deleted"] = await store.delete_documents("docs", set(files))
    finally:
        await r.delete(LOCK)
    stats["files"] = len(files)
    stats["ms"] = int((time.monotonic() - t0) * 1000)
    if stats["added"] or stats["updated"] or stats["deleted"] or stats["errors"]:
        log.info("docs index: %s", stats)
    return stats


def upload_path(title: str, text: str) -> str:
    slug = re.sub(r"[^\w\u3400-\u9fff-]+", "-", title.strip().lower()).strip("-")[:60] or "note"
    return f"{slug}-{chunking.sha256(text)[:8]}.md"


async def add_upload(title: str, text: str, *, private: bool = False) -> dict[str, Any]:
    raw = text if text.lstrip().startswith(("#", "---")) else f"# {title}\n\n{text}"
    return await store.upsert_document(source="upload", path=upload_path(title, text), raw=raw,
                                       rules=config.privacy_rules(), store_content=True, title=title,
                                       mtime=datetime.now(timezone.utc), force_private=private)
