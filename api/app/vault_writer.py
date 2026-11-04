"""Vault writes (§10a). The agent writes ONLY under `Inbox/` and `AI/`: those two folders are the only ones mounted
read-write into the workers (/vault-rw/Inbox, /vault-rw/AI), and every path is checked here as well.

A change to any other note is a proposal: an Inbox approval item with the diff. Only agent-api (owner-approved
answer route) mounts the whole vault read-write (/vault-edit) and applies it, if the note did not change since.

Files are written atomically (`.~harness-*` temp file, then rename; Syncthing ignores the temp names).
"""
import difflib
import hashlib
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from . import config, db, inbox
from .channels import line
from .rag import chunking
from .rag.vault import is_note_path, safe_rel
from .settings import get_settings

log = logging.getLogger(__name__)

WRITABLE = ("Inbox", "AI")
TMP_PREFIX = ".~harness-"
SUMMARY_START, SUMMARY_END = "<!-- harness:summary -->", "<!-- /harness:summary -->"


class VaultWriteError(ValueError):
    pass


def cfg() -> dict[str, Any]:
    return config.policy().get("vault") or {}


def tz() -> ZoneInfo:
    return ZoneInfo(str(cfg().get("timezone") or os.environ.get("TZ") or "Asia/Taipei"))


def now() -> datetime:
    return datetime.now(tz())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def note_rel(path: str) -> str:
    rel = safe_rel(path)
    if not rel:
        raise VaultWriteError(f"invalid note path {path!r}")
    if not is_note_path(rel):
        rel += ".md"
    return rel


def is_writable(rel: str) -> bool:
    return rel.split("/", 1)[0] in WRITABLE and "/" in rel


def _inside(base: Path, p: Path) -> Path:
    """`p` resolved; refuse anything that leaves `base` (symlinks included)."""
    rp = p.resolve()
    if not rp.is_relative_to(base.resolve()):
        raise VaultWriteError("path leaves the vault")
    return rp


def writable_path(rel: str) -> Path:
    """Host file for an agent write: only Inbox/… and AI/… notes, under their own read-write mounts."""
    rel = note_rel(rel)
    if not is_writable(rel):
        raise VaultWriteError(f"the agent may only write under {', '.join(w + '/' for w in WRITABLE)} (got {rel})")
    top, rest = rel.split("/", 1)
    base = Path(get_settings().vault_rw_root) / top
    if not base.is_dir():
        raise VaultWriteError(f"{top}/ is not mounted read-write here")
    return _inside(base, base / rest)


def atomic_write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=TMP_PREFIX, suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def frontmatter(meta: dict[str, Any]) -> str:
    return "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, default_flow_style=None) + "---\n"


def set_meta(text: str, **updates: Any) -> str:
    """Update frontmatter keys (adding a frontmatter block when there is none)."""
    meta, body = chunking.split_frontmatter(text)
    meta = dict(meta or {})
    meta.update({k: v for k, v in updates.items() if v is not None})
    return frontmatter(meta) + body.lstrip("\n")


def is_private_text(text: str, path: str = "") -> bool:
    from .rag.store import doc_privacy

    return doc_privacy(path, {}, text, config.privacy_rules())[0] == "PRIVATE"


def slug(text: str, n: int = 48) -> str:
    first = next((l for l in text.splitlines() if l.strip()), "note")
    s = re.sub(r"[\\/:*?\"<>|#^\[\]]+", " ", first.lstrip("# ").strip())
    s = re.sub(r"\s+", " ", s).strip(" .")[:n].strip(" .")
    return s or "note"


# ---------------------------------------------------------------- agent writes (Inbox/, AI/)
def capture(text: str, *, title: str = "", source: str = "agent", tags: list[str] | None = None,
            private: bool = False, when: datetime | None = None) -> str:
    """New note in Inbox/ ("YYYY-MM-DD HHmm title.md"). Returns its vault path."""
    text = (text or "").strip()
    if not text:
        raise VaultWriteError("empty note")
    when = when or now()
    private = private or is_private_text(text)
    name = f"{when:%Y-%m-%d %H%M} {slug(title or text)}"
    rel = f"Inbox/{name}.md"
    p = writable_path(rel)
    i = 2
    while p.exists():
        rel = f"Inbox/{name} {i}.md"
        p, i = writable_path(rel), i + 1
    meta = {"created": when.isoformat(timespec="minutes"), "source": source,
            "tags": list(dict.fromkeys(["inbox", *(tags or [])])), "privacy": "private" if private else "normal"}
    body = (f"# {title.strip()}\n\n" if title.strip() else "") + text + "\n"
    atomic_write(p, frontmatter(meta) + "\n" + body)
    return rel


def daily_rel(day: datetime) -> str:
    folder = str(cfg().get("daily_folder") or "AI/Daily").strip("/")
    return f"{folder}/{day:%Y-%m-%d}.md"


def _daily_template(day: datetime, private: bool) -> str:
    meta = {"date": f"{day:%Y-%m-%d}", "source": "harness", "tags": ["daily"],
            "privacy": "private" if private else "normal"}
    return frontmatter(meta) + f"\n# {day:%Y-%m-%d %A}\n\n{SUMMARY_START}\n{SUMMARY_END}\n\n## Log\n"


def _flip_private(text: str, private: bool) -> str:
    """Only ever upgrades a note to private, never back."""
    meta, _ = chunking.split_frontmatter(text)
    if private and str(meta.get("privacy") or "").lower() != "private":
        return set_meta(text, privacy="private")
    return text


def append_daily(entry: str, *, private: bool = False, when: datetime | None = None) -> str:
    """`- HH:MM entry` under `## Log` in today's AI/Daily note. Returns its vault path."""
    entry = " ".join((entry or "").split())
    if not entry:
        raise VaultWriteError("empty entry")
    when = when or now()
    private = private or is_private_text(entry)
    rel = daily_rel(when)
    p = writable_path(rel)
    text = p.read_text(encoding="utf-8") if p.exists() else _daily_template(when, private)
    text = _flip_private(text, private)
    line_ = f"- {when:%H:%M} {entry}\n"
    if "\n## Log" not in text:
        text = text.rstrip("\n") + "\n\n## Log\n"
    head, _, tail = text.partition("\n## Log")
    nxt = re.search(r"\n#{1,2} ", tail[1:])          # the Log section ends at the next heading
    cut = (nxt.start() + 1) if nxt else len(tail)
    section, rest = tail[:cut].rstrip("\n") + "\n", tail[cut:]
    text = head + "\n## Log" + section + line_ + rest
    atomic_write(p, text)
    return rel


def write_daily_summary(day: datetime, summary: str, *, private: bool = True) -> str:
    """Replace the harness summary block of a day's note (created when missing); the Log and owner edits stay."""
    rel = daily_rel(day)
    p = writable_path(rel)
    text = p.read_text(encoding="utf-8") if p.exists() else _daily_template(day, private)
    text = _flip_private(text, private)
    block = f"{SUMMARY_START}\n{summary.strip()}\n{SUMMARY_END}"
    if SUMMARY_START in text and SUMMARY_END in text:
        a = text.index(SUMMARY_START)
        b = text.index(SUMMARY_END, a) + len(SUMMARY_END)
        text = text[:a] + block + text[b:]
    else:
        _, body = chunking.split_frontmatter(text)
        fm = text[: len(text) - len(body)]
        m = re.match(r"(\s*# [^\n]*\n)", body)
        text = fm + (body[: m.end()] + "\n" + block + "\n" + body[m.end():] if m else block + "\n\n" + body)
    atomic_write(p, text)
    return rel


def apply_replace(text: str, old: str, new: str) -> str:
    if not old:
        raise VaultWriteError("`old` is empty: give the exact text to replace")
    n = text.count(old)
    if n != 1:
        raise VaultWriteError(f"`old` text found {n} times in the note; it must match exactly once")
    return text.replace(old, new, 1)


def diff(rel: str, before: str, after: str, n: int = 6000) -> str:
    d = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), f"a/{rel}", f"b/{rel}", n=2))
    return d[:n] + ("\n…(diff truncated)" if len(d) > n else "")


# ---------------------------------------------------------------- edits elsewhere: owner approval
def _read_ro(rel: str) -> str | None:
    base = Path(get_settings().vault_root)
    p = _inside(base, base / rel)
    return p.read_text(encoding="utf-8") if p.is_file() else None


async def propose_edit(path: str, old: str, new: str, *, reason: str = "", private: bool = False) -> str:
    """Edit a note: applied at once under Inbox/ or AI/, otherwise an Inbox approval with the diff.
    `old` = "" creates a new note (it must not exist yet)."""
    rel = note_rel(path)
    if is_writable(rel):
        p = writable_path(rel)
        if not old:
            if p.exists():
                raise VaultWriteError(f"{rel} already exists; give the text to replace")
            atomic_write(p, new if new.endswith("\n") else new + "\n")
        else:
            if not p.exists():
                raise VaultWriteError(f"no note {rel}")
            atomic_write(p, apply_replace(p.read_text(encoding="utf-8"), old, new))
        await db.audit("vault", "write", {"path": rel})
        return f"Saved {rel}."
    cur = _read_ro(rel)
    if old:
        if cur is None:
            raise VaultWriteError(f"no note {rel}")
        after = apply_replace(cur, old, new)
    else:
        if cur is not None:
            raise VaultWriteError(f"{rel} already exists; give the text to replace")
        after = new if new.endswith("\n") else new + "\n"
    data = {"path": rel, "old": old, "new": new, "base_sha": sha(cur) if cur is not None else ""}
    for it in await inbox.list_("open", "approval", 100):
        if (it.get("detail") or {}).get("vault_edit") == data:
            return f"Already waiting for approval in the Inbox (item #{it['id']}). The note has not changed yet."
    body = (f"{reason.strip()}\n\n" if reason.strip() else "") + "```diff\n" + diff(rel, cur or "", after) + "\n```"
    item = await inbox.add("approval", f"Edit note: {rel}", body, priority="normal", private=True,
                           detail={"options": [], "vault_edit": data})
    if item is None:
        return "Could not create the approval item; the note was not changed."
    await db.audit("vault", "propose_edit", {"path": rel})
    s = get_settings()
    if s.line_owner_user_id and s.line_channel_access_token and cfg().get("notify_line", True):
        try:
            await line.push(s.line_owner_user_id, "📥 A note edit is waiting for your approval in the Inbox.")
        except Exception:
            log.exception("LINE vault push failed")
    return (f"Proposed (Inbox item #{item}). {rel} has NOT been changed yet: only Inbox/ and AI/ are written "
            "directly; the owner must approve this edit in the Inbox. Tell the owner that.")


def apply_approved(detail: dict[str, Any]) -> str:
    """Owner approved: apply on the full read-write mount (agent-api only), if the note is unchanged since."""
    e = detail["vault_edit"]
    rel = note_rel(e["path"])
    base = Path(get_settings().vault_edit_root)
    if not base.is_dir():
        raise VaultWriteError("the vault is not mounted read-write here")
    p = _inside(base, base / rel)
    cur = p.read_text(encoding="utf-8") if p.is_file() else None
    if (sha(cur) if cur is not None else "") != e.get("base_sha", ""):
        raise VaultWriteError(f"{rel} changed since the proposal; ask again")
    after = apply_replace(cur, e["old"], e["new"]) if e["old"] else (e["new"] if e["new"].endswith("\n")
                                                                      else e["new"] + "\n")
    atomic_write(p, after)
    return "applied"
