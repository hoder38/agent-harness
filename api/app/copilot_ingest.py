"""SSH Copilot → harness memory (§9b): parse a host Copilot session's `events.jsonl`, redact secrets, summarise it
with the local model and write the thread `copilot:<sessionId>` plus a `coding_episodes` memory.

The session folder is mounted read-only at `copilot_sessions_dir`; nothing here writes to it.
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from . import memory
from .settings import get_settings

log = logging.getLogger(__name__)

SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{7,63}$")
THREAD_PREFIX = "copilot:"
FILE_TOOLS = {"edit", "create", "apply_patch", "str_replace_editor", "write"}

_R = "[REDACTED]"
_SECRETS: list[tuple[re.Pattern[str], Any]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S), _R),
    (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"), _R),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), _R),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), _R),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), _R),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), _R),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{16,}"), lambda m: f"{m.group(1)} {_R}"),
    (re.compile(r"(?i)(\b[\w.-]*(?:password|passwd|secret|token|api[_-]?key|access[_-]?key)[\w.-]*\s*[=:]\s*)"
                r"(?:\"[^\"\n]+\"|'[^'\n]+'|[^\s\"',;]+)"), lambda m: m.group(1) + _R),
    (re.compile(r"(://[^/\s:@]+:)[^@\s/]+(@)"), lambda m: m.group(1) + _R + m.group(2)),
]


def redact(text: str) -> str:
    """Mask credentials before text goes to a model or a memory (best effort; the model is local anyway)."""
    for pat, repl in _SECRETS:
        text = pat.sub(repl, text)
    return text


@dataclass
class Digest:
    session_id: str
    cwd: str = ""
    repo: str = ""
    branch: str = ""
    started_at: str = ""
    last_event_id: str = ""
    n_events: int = 0
    ended: bool = False
    turns: list[tuple[str, str]] = field(default_factory=list)   # (owner prompt, final assistant answer)
    files: list[str] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)
    prs: list[str] = field(default_factory=list)

    @property
    def thread_id(self) -> str:
        return THREAD_PREFIX + self.session_id


_COMMIT = re.compile(r"^\[([\w./-]+)(?: \(root-commit\))? ([0-9a-f]{7,40})\] (.+)$", re.M)
_PR = re.compile(r"https://github\.com/([\w.-]+/[\w.-]+)/pull/(\d+)")


def _push(lst: list[str], v: str, cap: int = 50) -> None:
    if v and v not in lst and len(lst) < cap:
        lst.append(v)


def parse_events(lines: list[str], session_id: str) -> Digest:
    """Fold JSONL events into a Digest. Unparseable lines are skipped (a hook can fire mid-write)."""
    d = Digest(session_id)
    cur_prompt: str | None = None
    cur_answer = ""
    bash_calls: set[str] = set()

    def close_turn() -> None:
        nonlocal cur_prompt, cur_answer
        if cur_prompt is not None:
            d.turns.append((cur_prompt, cur_answer))
        cur_prompt, cur_answer = None, ""

    for line in lines:
        try:
            e = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(e, dict):
            continue
        t, data = e.get("type"), e.get("data") or {}
        d.n_events += 1
        d.last_event_id = str(e.get("id") or d.last_event_id)
        if t == "session.start":
            ctx = data.get("context") or {}
            d.cwd = d.cwd or str(ctx.get("cwd") or "")
            d.repo = d.repo or str(ctx.get("repository") or "")
            d.branch = d.branch or str(ctx.get("branch") or "")
            d.started_at = d.started_at or str(data.get("startTime") or "")
        elif t == "session.shutdown":
            d.ended = True
            for f in ((data.get("accountingSnapshot") or data).get("codeChanges") or {}).get("filesModified") or []:
                _push(d.files, str(f))
        elif t == "user.message":
            close_turn()
            cur_prompt = str(data.get("content") or "")
        elif t == "assistant.message":
            body = str(data.get("content") or "").strip()
            if body and not data.get("parentToolCallId"):   # sub-agent chatter is not the answer to the owner
                cur_answer = body
        elif t == "tool.execution_start":
            name, args = str(data.get("toolName") or ""), data.get("arguments") or {}
            if name in FILE_TOOLS and isinstance(args, dict):
                _push(d.files, str(args.get("path") or args.get("file_path") or ""))
            elif name == "bash" and isinstance(args, dict) and re.search(r"\bgit\s+commit\b", str(args.get("command"))):
                bash_calls.add(str(data.get("toolCallId")))
        elif t == "tool.execution_complete" and data.get("success") is not False:
            res = str((data.get("result") or {}).get("content") or "")
            if str(data.get("toolCallId")) in bash_calls:
                for m in _COMMIT.finditer(res):
                    _push(d.commits, f"{m.group(1)} {m.group(2)[:7]} {m.group(3)[:120]}")
            for m in _PR.finditer(res):
                _push(d.prs, f"{m.group(1)}#{m.group(2)}")
    close_turn()
    return d


def read_session(session_id: str, root: str | None = None) -> Digest | None:
    if not SESSION_RE.match(session_id):
        return None
    base = Path(root or get_settings().copilot_sessions_dir).resolve()
    path = (base / session_id / "events.jsonl").resolve()
    if base not in path.parents or not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as f:
        return parse_events(f.read().splitlines(), session_id)


def messages(d: Digest, *, per_msg: int = 4000) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for prompt, answer in d.turns:
        out.append(HumanMessage(redact(prompt)[:per_msg]))
        if answer:
            out.append(AIMessage(redact(answer)[:per_msg]))
    return out


def transcript(d: Digest, max_chars: int = 12000) -> str:
    lines = []
    for prompt, answer in d.turns:
        lines.append(f"Owner: {redact(prompt)[:1500]}")
        if answer:
            lines.append(f"Copilot: {redact(answer)[:1500]}")
    return "\n\n".join(lines)[-max_chars:]


SUMMARY_SYSTEM = """Summarise this coding session between the owner and GitHub Copilot as a memory for later.
3-6 sentences in the session's main language: what the owner wanted, what was changed or decided (key files,
commands, names, numbers), and anything left open. Third person ("The owner …"). No preamble. Never repeat secrets."""


def facts_block(d: Digest) -> str:
    parts = []
    if d.cwd or d.repo:
        parts.append(f"Repo: {d.repo or d.cwd}" + (f" (branch {d.branch})" if d.branch else ""))
    if d.files:
        parts.append("Files changed: " + ", ".join(d.files[:12]) + (" …" if len(d.files) > 12 else ""))
    if d.commits:
        parts.append("Commits: " + "; ".join(d.commits[:5]))
    if d.prs:
        parts.append("PRs: " + ", ".join(d.prs[:5]))
    return "\n".join(parts)


async def summarize(llm: BaseChatModel, model: str, d: Digest) -> str | None:
    tr = transcript(d)
    if not tr.strip():
        return None
    human = (facts_block(d) + "\n\n" if facts_block(d) else "") + tr
    msg = await llm.bind(max_tokens=500, **memory._local_kw(model)).ainvoke(
        [SystemMessage(SUMMARY_SYSTEM), HumanMessage(human)],
        config={"tags": ["internal", "memory"], "run_name": "copilot_episode"})
    text = re.sub(r"<think>.*?</think>", "", memory._text(msg.content), flags=re.S).strip()
    return redact(text) or None


async def write_episode(d: Digest, summary: str, *, private: bool = False) -> str:
    day = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
    head = f"[{day}] Copilot session"
    if d.repo or d.cwd:
        head += f" in {d.repo or d.cwd}"
    text = f"{head}: {summary}"
    extra = facts_block(d)
    if extra:
        text += "\n" + extra
    existing = await memory.get_store().aget((memory.ROOT, "coding_episodes"), d.session_id)
    pinned = bool(existing and (existing.value or {}).get("pinned"))
    return await memory.remember("coding_episodes", text, private=private, kind="coding_episode", key=d.session_id,
                                 pinned=pinned, source={"thread_id": d.thread_id, "session_id": d.session_id,
                                                        "channel": "copilot_ingest", "cwd": d.cwd})


async def write_thread(checkpointer: Any, d: Digest) -> None:
    """Replace the LangGraph thread `copilot:<id>` with the session's prompts and final answers."""
    from langgraph.checkpoint.base import empty_checkpoint

    msgs = messages(d)
    if not msgs:
        return
    version = time.time_ns()
    cp = empty_checkpoint()
    cp["channel_values"] = {"messages": msgs}
    cp["channel_versions"] = {"messages": version}
    await checkpointer.aput(
        {"configurable": {"thread_id": d.thread_id, "checkpoint_ns": ""}}, cp,
        {"source": "update", "step": 0, "parents": {}, "copilot_session": d.session_id}, {"messages": version})
