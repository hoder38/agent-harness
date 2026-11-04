"""Obsidian vault (§10a): parse notes and index them as RAG source `vault`.

- Notes: frontmatter (tags, aliases, privacy), `[[wikilinks]]`, `![[embeds]]`, markdown links, inline `#tags`.
  Links are stored in `note_links` so retrieval can expand to notes one link away (store.search).
- Embedded attachments (PDF, images, Office) are copied into the files store and extracted by the media-worker
  (always as PRIVATE, so locally); the text is indexed with the parent note once it is ready.
- Full scans: at vault-indexer start and from the scheduler cron (fallback for missed events). The vault-indexer
  watcher (app/rag/watcher.py) re-indexes only changed files. Both hold the same Redis lock.

The vault is mounted read-only here; nothing in this module writes to it (writes: app/vault_writer.py).
"""
import asyncio
import hashlib
import logging
import os
import posixpath
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

from .. import config, db, state
from ..settings import get_settings
from . import chunking, store

log = logging.getLogger(__name__)

SOURCE = "vault"
LOCK = "rag:index_lock:vault"
NOTE_EXTS = (".md", ".markdown")
# Link targets with these extensions are files, anything else is a note ("v1.2 plan" is a note).
FILE_EXTS = frozenset(
    "pdf png jpg jpeg gif webp svg bmp heic heif avif tif tiff mp3 m4a wav ogg flac webm mp4 mov mkv avi docx doc "
    "xlsx xls pptx ppt csv tsv txt json html htm zip base canvas excalidraw epub".split())

_FENCE = re.compile(r"^\s*(```|~~~)")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_WIKI = re.compile(r"(!?)\[\[([^\[\]\n]+?)\]\]")
_MDLINK = re.compile(r"(!?)\[([^\]\n]*)\]\(\s*(<[^>\n]+>|[^)\s]+)(?:\s+\"[^\"\n]*\")?\s*\)")
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)
_TAG = re.compile(r"(?:(?<=\s)|(?<=^))#([^\s#\[\](){}<>,.;:!?'\"`，。、；：！？「」]+)", re.M)


def cfg() -> dict[str, Any]:
    return config.policy().get("vault") or {}


def root() -> Path:
    return Path(get_settings().vault_root)


# ---------------------------------------------------------------- parsing (pure)
@dataclass
class Link:
    kind: str           # link | embed
    target: str         # as written, without #heading / |alias
    frag: str = ""      # heading or ^block
    alias: str = ""
    md: bool = False    # markdown link [text](target) rather than [[wikilink]]
    span: tuple[int, int] = (0, 0)  # position in the text it was parsed from

    @property
    def is_file(self) -> bool:
        return is_file_target(self.target)


@dataclass
class Note:
    meta: dict[str, Any]
    frontmatter: str    # the raw frontmatter block ('' if none)
    body: str
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)


def is_note_path(rel: str) -> bool:
    return rel.lower().endswith(NOTE_EXTS)


def is_file_target(target: str) -> bool:
    ext = posixpath.splitext(target.rsplit("/", 1)[-1])[1].lower().lstrip(".")
    return ext in FILE_EXTS


def strip_code(text: str) -> str:
    """Code blocks and inline code blanked out (same length per line), so links/tags inside code are ignored."""
    out, fence = [], None
    for line in text.split("\n"):
        m = _FENCE.match(line)
        if m:
            fence = m.group(1) if fence is None else (None if m.group(1) == fence else fence)
            out.append(" " * len(line))
            continue
        out.append(" " * len(line) if fence else _INLINE_CODE.sub(lambda x: " " * len(x.group(0)), line))
    return "\n".join(out)


def _str_list(v: Any) -> list[str]:
    if isinstance(v, str):
        return [t for t in re.split(r"[,\s]+", v) if t]
    if isinstance(v, (list, tuple)):
        return [str(t).strip() for t in v if t is not None and str(t).strip()]
    return []


def parse_links(text: str) -> list[Link]:
    """Wikilinks/embeds and relative markdown links in `text` (code already stripped or not: caller decides)."""
    out: list[Link] = []
    for m in _WIKI.finditer(text):
        inner = m.group(2).replace("\\|", "|")
        target, _, alias = inner.partition("|")
        target, _, frag = target.partition("#")
        target = target.strip()
        if not target:          # [[#Heading]]: same note
            continue
        out.append(Link("embed" if m.group(1) else "link", target, frag.strip(), alias.strip(), False, m.span()))
    for m in _MDLINK.finditer(text):
        t = m.group(3)
        t = t[1:-1] if t.startswith("<") else t
        if _SCHEME.match(t) or t.startswith("#"):
            continue
        t, _, frag = t.partition("#")
        t = unquote(t).strip()
        if t:
            out.append(Link("embed" if m.group(1) else "link", t, unquote(frag), m.group(2).strip(), True, m.span()))
    return sorted(out, key=lambda l: l.span)


def parse(raw: str) -> Note:
    meta, body = chunking.split_frontmatter(raw)
    fm = raw[: len(raw) - len(body)]
    plain = strip_code(body)
    tags = [t.lstrip("#") for t in _str_list(meta.get("tags") or meta.get("tag"))]
    tags += [t for t in _TAG.findall(plain) if not t.isdigit()]
    aliases = _str_list(meta.get("aliases") or meta.get("alias"))
    return Note(meta=meta, frontmatter=fm, body=body, tags=list(dict.fromkeys(tags)), aliases=aliases,
                links=parse_links(plain))


def link_key(target: str, *, note: bool) -> str:
    """Lower-case vault-relative key: notes without `.md` (matches SQL vault_path_key/vault_name_key)."""
    t = posixpath.normpath(target.replace("\\", "/").strip().lstrip("/")).lower()
    t = "" if t == "." else t
    return re.sub(r"\.(md|markdown)$", "", t) if note else t


class Resolver:
    """Obsidian-style link resolution over the vault's file list: exact path, then path suffix, then unique name
    (the shallowest path wins when several notes share a name); `./`, `../` and markdown links try the linking
    note's folder first."""

    def __init__(self, paths: Iterable[str]):
        self.by_key: dict[str, str] = {}
        self.by_name: dict[str, list[str]] = {}
        for p in sorted(paths, key=lambda x: (x.count("/"), x)):
            note = is_note_path(p)
            k = link_key(p, note=note)
            self.by_key.setdefault(k, p)
            self.by_name.setdefault(k.rsplit("/", 1)[-1], []).append(p)

    def resolve(self, link: Link, src: str) -> str | None:
        note = not link.is_file
        cands = []
        if link.md or link.target.startswith(("./", "../")):
            cands.append(link_key(posixpath.join(posixpath.dirname(src), link.target), note=note))
        k = link_key(link.target, note=note)
        cands.append(k)
        for c in cands:
            if c in self.by_key:
                return self.by_key[c]
        if "/" in k:
            for kk, p in self.by_key.items():
                if kk.endswith("/" + k):
                    return p
            return None
        names = self.by_name.get(k) or []
        return names[0] if names else None

    def key(self, link: Link, src: str) -> str:
        """note_links.dst: the resolved target's key when it exists, else the key as written."""
        p = self.resolve(link, src)
        return link_key(p if p else link.target, note=not link.is_file)


def display(body: str, links: list[Link]) -> str:
    """Body text for chunking: links become their visible text; file embeds become `📎 name`."""
    out, pos = [], 0
    for l in links:
        a, b = l.span
        if a < pos:
            continue
        name = l.target.rsplit("/", 1)[-1]
        if l.kind == "embed" and l.is_file:
            text = f"📎 {name}"
        elif l.md:
            text = l.alias or name
        else:
            text = l.alias or (re.sub(r"\.(md|markdown)$", "", name, flags=re.I)
                               + (f" › {l.frag.lstrip('^')}" if l.frag else ""))
        out += [body[pos:a], text]
        pos = b
    out.append(body[pos:])
    return "".join(out)


def render(note: Note, attachments: list[tuple[str, str]]) -> str:
    """What gets indexed: frontmatter (privacy, dates) + display body + one section per extracted attachment."""
    text = note.frontmatter + display(note.body, note.links)
    for name, att_text in attachments:
        text += f"\n\n## 📎 {name}\n\n{att_text.strip()}"
    return text


def safe_rel(path: str) -> str | None:
    """A vault-relative note path, or None if it escapes the vault or touches hidden files (.git, .obsidian)."""
    p = posixpath.normpath((path or "").replace("\\", "/").strip().lstrip("/"))
    if not p or p == "." or p.startswith("../") or p == ".." or any(s.startswith(".") for s in p.split("/")):
        return None
    return p


# ---------------------------------------------------------------- scanning
def scan_all(base: Path) -> dict[str, Path]:
    """Every non-hidden regular file (notes and attachments), relative posix path → file."""
    out: dict[str, Path] = {}
    if not base.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            if f.startswith("."):
                continue
            p = Path(dirpath) / f
            if not p.is_symlink():
                out[p.relative_to(base).as_posix()] = p
    return out


# ---------------------------------------------------------------- attachments
_arq = None


async def _arq_pool():
    global _arq
    if _arq is None:
        from arq import create_pool
        from arq.connections import RedisSettings

        _arq = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _arq


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


async def _file_chunks(p: Path):
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            yield b


async def attachment_text(rel: str, p: Path) -> str | None:
    """Extracted text of an embedded file, or None (not supported / too large / extraction still pending).
    The first time a file (by content) is seen it is stored and queued on the media-worker."""
    from .. import media

    c = cfg()
    if not c.get("attachments", True):
        return None
    st = p.stat()
    if st.st_size == 0 or st.st_size > float(c.get("attachment_max_mb", 50)) * 1024 * 1024:
        return None
    async with db.get_pool().connection() as conn:
        row = await (await conn.execute("SELECT * FROM vault_files WHERE path=%s", (rel,))).fetchone()
    if row and row["size"] == st.st_size and row["mtime"] == st.st_mtime:
        sha, att_id = row["sha256"], row["att_id"]
    else:
        sha, att_id = await asyncio.to_thread(_sha256_file, p), None
    att = None
    async with db.get_pool().connection() as conn:
        if att_id:
            att = await (await conn.execute("SELECT id, status, text, kind FROM attachments WHERE id=%s",
                                            (att_id,))).fetchone()
        if not att or att["status"] == "error":   # same bytes extracted before (another note, an upload)?
            att = await (await conn.execute(
                "SELECT id, status, text, kind FROM attachments WHERE sha256=%s AND status <> 'error' "
                "ORDER BY (status = 'done') DESC, created_at DESC LIMIT 1", (sha,))).fetchone() or att
    if not att:
        with open(p, "rb") as f:
            head = f.read(8192)
        try:
            _, kind = media.sniff(head, p.name)
        except media.MediaError:
            kind = None
        if kind not in set(c.get("attachment_kinds") or ["image", "doc"]):
            att = None
        else:
            # PRIVATE: extracted on local models only, whatever the parent note's privacy
            att = await media.store(_file_chunks(p), channel="vault", thread_id=None, filename=p.name, private=True)
            await (await _arq_pool()).enqueue_job("extract_attachment", att["id"], _job_id=f"extract:{att['id']}",
                                                  _queue_name=media.media_queue())
            log.info("vault attachment %s queued for extraction (%s)", rel, att["id"])
    async with db.get_pool().connection() as conn:
        await conn.execute(
            """INSERT INTO vault_files (path, size, mtime, sha256, att_id) VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (path) DO UPDATE SET size=EXCLUDED.size, mtime=EXCLUDED.mtime, sha256=EXCLUDED.sha256,
                 att_id=EXCLUDED.att_id""", (rel, st.st_size, st.st_mtime, sha, att["id"] if att else None))
    if att and att.get("status") == "done" and (att.get("text") or "").strip():
        return att["text"][: int(c.get("attachment_max_chars", 100_000))]
    return None


# ---------------------------------------------------------------- indexing
async def _save_links(src: str, rows: list[tuple[str, str, str]]) -> None:
    async with db.get_pool().connection() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM note_links WHERE src=%s", (src,))
            for dst, kind, raw in dict.fromkeys(rows):
                await conn.execute("INSERT INTO note_links (src, dst, kind, raw) VALUES (%s,%s,%s,%s) "
                                   "ON CONFLICT DO NOTHING", (src, dst, kind, raw[:500]))


async def index_note(rel: str, p: Path, resolver: Resolver, files: dict[str, Path], rules: dict[str, Any],
                     *, force_links: bool = False) -> dict[str, Any]:
    raw = p.read_text(encoding="utf-8", errors="replace")
    note = parse(raw)
    atts: list[tuple[str, str]] = []
    for l in note.links:
        if l.kind == "embed" and l.is_file and (target := resolver.resolve(l, rel)) and target in files:
            if (t := await attachment_text(target, files[target])) and (target, t) not in atts:
                atts.append((target.rsplit("/", 1)[-1], t))
    title = str(note.meta.get("title") or "").strip() or re.sub(r"\.(md|markdown)$", "", p.name, flags=re.I)
    mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
    res = await store.upsert_document(source=SOURCE, path=rel, raw=render(note, atts), rules=rules, mtime=mtime,
                                      title=title, kw_extra=" ".join(note.tags + note.aliases),
                                      tags=note.tags, aliases=note.aliases)
    if res["status"] != "unchanged" or force_links:
        await _save_links(rel, [(resolver.key(l, rel), l.kind, l.target) for l in note.links])
    return res


async def _lock(wait_s: float) -> bool:
    r = state.get_redis()
    deadline = time.monotonic() + wait_s
    while not await r.set(LOCK, "1", nx=True, ex=int(cfg().get("index_lock_s", 1800))):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(1)
    return True


def _new_stats() -> dict[str, Any]:
    return {"added": 0, "updated": 0, "unchanged": 0, "deleted": 0, "errors": [], "private": 0}


async def _index_many(rels: Iterable[str], files: dict[str, Path], stats: dict[str, Any], *,
                      force_links: bool = False) -> None:
    resolver = Resolver(files)
    rules = config.privacy_rules()
    for rel in rels:
        try:
            res = await index_note(rel, files[rel], resolver, files, rules, force_links=force_links)
            stats[res["status"]] += 1
            stats["private"] += res.get("privacy") == "PRIVATE"
        except Exception as e:
            log.exception("vault index %s failed", rel)
            stats["errors"].append(f"{rel}: {type(e).__name__}: {e}"[:300])


async def index_vault(base: str | Path | None = None, *, force_links: bool = False,
                      wait_s: float = 0) -> dict[str, Any]:
    """Full scan: (re)index changed notes, drop deleted ones. Never deletes anything when the mount is missing."""
    base = Path(base or root())
    if not await _lock(wait_s):
        return {"status": "busy"}
    t0 = time.monotonic()
    stats = _new_stats()
    try:
        if not base.is_dir():
            stats["errors"].append(f"{base} is not a directory")
            return stats
        files = scan_all(base)
        notes = sorted(r for r in files if is_note_path(r) and files[r].stat().st_size
                       <= int((config.policy().get("rag") or {}).get("max_file_kb", 2048)) * 1024)
        await _index_many(notes, files, stats, force_links=force_links)
        stats["deleted"] = await store.delete_documents(SOURCE, set(notes))
        async with db.get_pool().connection() as conn:
            await conn.execute("DELETE FROM note_links WHERE NOT (src = ANY(%s))", (notes,))
            await conn.execute("DELETE FROM vault_files WHERE NOT (path = ANY(%s))", (list(files),))
        stats["files"] = len(notes)
    finally:
        await state.get_redis().delete(LOCK)
    stats["ms"] = int((time.monotonic() - t0) * 1000)
    if stats["added"] or stats["updated"] or stats["deleted"] or stats["errors"]:
        log.info("vault index: %s", stats)
    return stats


async def index_changed(changed: set[str], base: str | Path | None = None) -> dict[str, Any]:
    """Re-index only what changed (watcher). A changed attachment re-indexes the notes that embed it; anything
    that is not a plain file (a renamed/deleted folder) falls back to a full scan."""
    base = Path(base or root())
    files = scan_all(base)
    rels = {r for r in changed if (safe := safe_rel(r)) and safe == r}
    if any(r not in files and not is_note_path(r) and not is_file_target(r) for r in rels):
        return await index_vault(base, wait_s=600)     # a folder moved or was deleted
    if not await _lock(600):
        return {"status": "busy"}
    t0 = time.monotonic()
    stats = _new_stats()
    try:
        notes = {r for r in rels if is_note_path(r) and r in files}
        gone = {r for r in rels if is_note_path(r) and r not in files}
        att_keys = [link_key(r, note=False) for r in rels if not is_note_path(r)]
        if att_keys:   # notes embedding a changed file: re-hash picks up the new extraction
            names = [k.rsplit("/", 1)[-1] for k in att_keys]
            async with db.get_pool().connection() as conn:
                cur = await conn.execute("SELECT DISTINCT src FROM note_links WHERE kind='embed' AND "
                                         "(dst = ANY(%s) OR dst = ANY(%s))", (att_keys, names))
                notes |= {r["src"] for r in await cur.fetchall() if r["src"] in files}
        await _index_many(sorted(notes), files, stats)
        if gone:
            stats["deleted"] = await store.delete_paths(SOURCE, gone)
            async with db.get_pool().connection() as conn:
                await conn.execute("DELETE FROM note_links WHERE src = ANY(%s)", (list(gone),))
    finally:
        await state.get_redis().delete(LOCK)
    stats["ms"] = int((time.monotonic() - t0) * 1000)
    log.info("vault changes %s: %s", sorted(rels)[:10], {k: v for k, v in stats.items() if v})
    return stats


# ---------------------------------------------------------------- reading (MCP, tools)
async def read_note(path: str, *, private: bool, max_chars: int = 20_000) -> tuple[str | None, str]:
    """(text, error). Non-private callers never get a PRIVATE note; their text is Gate 0 redacted."""
    from ..gates.gate0 import redact

    rel = safe_rel(path)
    if rel and not is_note_path(rel):
        rel += ".md"
    if not rel:
        return None, "invalid path"
    p = root() / rel
    if not p.is_file():
        files = scan_all(root())
        hit = Resolver(files).resolve(Link("link", path), "")
        if not hit or not is_note_path(hit):
            return None, f"no note {path!r}"
        rel, p = hit, root() / hit
    raw = p.read_text(encoding="utf-8", errors="replace")
    if not private:
        meta, _ = chunking.split_frontmatter(raw)
        tag = await store.doc_privacy_tag(SOURCE, rel)
        live, _ = store.doc_privacy(rel, meta, raw, config.privacy_rules())
        if "PRIVATE" in (tag, live):
            return None, "this note is private"
        raw = redact(raw, config.privacy_rules())[0]
    text = raw[:max_chars] + ("\n…(truncated)" if len(raw) > max_chars else "")
    return f"# {rel}\n\n{text}", ""
