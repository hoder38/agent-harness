"""Nightly daily note (§10a): AI/Daily/YYYY-MM-DD.md gets a summary of the day, built in code (no model call):
conversation and Copilot episodes updated that day, todos completed, notes captured into Inbox/.

Always `privacy: private` (episodes and todo titles are PRIVATE). Only the block between the harness:summary
markers is rewritten, so the `## Log` lines and anything the owner adds stay. Schedule kind `vault_daily`.
"""
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import memory, todos, vault_writer as vw
from .settings import get_settings

_DAY_PREFIX = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]\s*")


def _on(ts: Any, day: datetime) -> bool:
    if not isinstance(ts, datetime):
        return False
    return ts.astimezone(day.tzinfo).date() == day.date()


def _one_line(text: str, n: int = 300) -> str:
    t = " ".join(_DAY_PREFIX.sub("", text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def build(day: datetime, *, episodes: list[dict[str, Any]], coding: list[dict[str, Any]],
          done: list[dict[str, Any]], captured: list[str]) -> str:
    """Markdown summary for `day` (pure; the inputs are already filtered to that day)."""
    out: list[str] = ["## Summary", f"_Generated {vw.now():%Y-%m-%d %H:%M} by the harness._", ""]
    if episodes:
        out += ["### Conversations", *[f"- {_one_line(e['text'])}" for e in episodes], ""]
    if coding:
        out += ["### Coding", *[f"- {_one_line(e['text'])}" for e in coding], ""]
    if done:
        out += ["### Todos done", *[f"- [x] {_one_line(t['title'], 200)}" for t in done], ""]
    if captured:
        out += ["### Captured", *[f"- [[{c.removesuffix('.md')}]]" for c in captured], ""]
    if len(out) == 3:
        out.append("Nothing recorded today.")
    return "\n".join(out).rstrip() + "\n"


def _captured(day: datetime) -> list[str]:
    base = Path(get_settings().vault_rw_root) / "Inbox"
    if not base.is_dir():
        return []
    return sorted(f"Inbox/{p.name}" for p in base.glob(f"{day:%Y-%m-%d} *.md") if p.is_file())


async def run(day: datetime | None = None) -> str:
    """Write the summary of `day` (default: today, Asia/Taipei). Returns the note's vault path."""
    day = day or vw.now()
    eps = [e for e in await memory.search("episodes", None, private=True, limit=500) if _on(e["updated_at"], day)]
    cod = [e for e in await memory.search("coding_episodes", None, private=True, limit=500)
           if _on(e["updated_at"], day)]
    done = [t for t in await todos.list_("done", 500) if _on(t["updated_at"], day)]
    eps.sort(key=lambda e: e["updated_at"])
    cod.sort(key=lambda e: e["updated_at"])
    return vw.write_daily_summary(day, build(day, episodes=eps, coding=cod, done=done, captured=_captured(day)),
                                  private=True)


async def run_for(date: str) -> str:
    """`YYYY-MM-DD` or "yesterday" (manual re-run: python -m app.tools.vault_tool daily)."""
    d = vw.now() - timedelta(days=1) if date == "yesterday" else datetime.strptime(date, "%Y-%m-%d").replace(
        tzinfo=vw.tz(), hour=23, minute=59)
    return await run(d)
