"""Rolling conversation compaction (policy `history`, 2026-11-09).

The model sees: one running summary of the older turns (+ the latest long paste/attachment kept verbatim), then the
turns after the summary's boundary verbatim (still capped by `executor.max_history_turns`). After each finished turn
`after_turn` decides whether to fold older turns into the summary; `compact_thread` does it on the background queue
with the local model (summaries may hold PRIVATE text), so answers never wait for it.

Kept verbatim: the last `keep_turns` turns and the latest long owner paste. Summarised: constraints/preferences
(never dropped), completed actions, question → conclusion with exact numbers, sources (title + URL), corrections,
open questions. Dropped: tool calls/results, page excerpts, thinking, small talk, superseded answers.
The checkpointed thread itself is never changed.
"""
import contextlib
import hashlib
import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from . import config

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS thread_summaries (
    thread_id       TEXT PRIMARY KEY,
    summary         TEXT NOT NULL DEFAULT '',
    pinned          TEXT NOT NULL DEFAULT '',
    covered         INTEGER NOT NULL DEFAULT 0,      -- owner turns folded into the summary
    boundary        TEXT NOT NULL DEFAULT '',        -- sha256 of the last folded owner message
    private         BOOLEAN NOT NULL DEFAULT false,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

LAST_TYPE = "history:last_type:{}"   # redis: last non-chitchat task type of a thread (topic-change trigger)
_CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")
_THINK = re.compile(r"<think>.*?</think>", re.S)


def cfg() -> dict[str, Any]:
    return config.policy().get("history") or {}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return ""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def est_tokens(text: str) -> int:
    """Rough token count: CJK ≈ 1 token/char, other text ≈ 4 chars/token."""
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk) // 4


def human_idx(messages: list[AnyMessage]) -> list[int]:
    return [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]


def boundary_index(messages: list[AnyMessage], summary: dict[str, Any] | None) -> int | None:
    """Index of the first message after the summarised part, or None when the summary does not fit this thread."""
    covered = int((summary or {}).get("covered") or 0)
    if covered <= 0:
        return None
    hs = human_idx(messages)
    want = summary.get("boundary") or ""
    if covered <= len(hs) and _sha(_text(messages[hs[covered - 1]].content)) == want:
        k = covered
    else:  # messages shifted (e.g. a turn re-run): find the boundary message by content
        k = next((j + 1 for j in range(len(hs) - 1, -1, -1) if _sha(_text(messages[hs[j]].content)) == want), None)
        if k is None:
            return None
    return hs[k] if k < len(hs) else len(messages)


def inject(messages: list[AnyMessage], summary: dict[str, Any] | None, max_turns: int,
           trim=None) -> list[AnyMessage]:
    """The model's view of the thread: [summary note, ack] + the verbatim turns after the summary boundary."""
    from .graphs.main import trim_history  # late: graphs.main imports this module
    trim = trim or trim_history
    start = boundary_index(messages, summary)
    if start is None:
        return trim(messages, max_turns)
    rest = trim(messages[start:], max_turns)
    note = (summary.get("summary") or "").strip()
    pinned = (summary.get("pinned") or "").strip()
    pin_min = int(cfg().get("pin_min_chars", 1500))
    if pinned and any(isinstance(m, HumanMessage) and len(_text(m.content)) >= pin_min for m in rest):
        pinned = ""  # a newer long paste is in the verbatim turns: only the latest version is kept
    if not note and not pinned:
        return rest
    body = "Summary of our earlier conversation (older turns, condensed; use it as context, do not answer it):\n"
    body += note or "(nothing notable)"
    if pinned:
        body += "\n\nThe owner's latest long pasted text / attachment from those turns, verbatim:\n" + pinned
    return [HumanMessage(body), AIMessage("OK."), *rest]


# ---------------------------------------------------------------- storage
async def load(thread_id: str, *, private: bool) -> dict[str, Any] | None:
    """The thread's summary for a run, or None (disabled, none yet, or PRIVATE summary on a non-private run)."""
    if not cfg().get("enabled"):
        return None
    from . import db
    try:
        async with db.get_pool().connection() as conn:
            cur = await conn.execute("SELECT * FROM thread_summaries WHERE thread_id = %s", (thread_id,))
            row = await cur.fetchone()
    except Exception:
        log.exception("history summary load failed for %s", thread_id)
        return None
    if not row or (row["private"] and not private):
        return None
    return dict(row)


async def save(thread_id: str, *, summary: str, pinned: str, covered: int, boundary: str, private: bool) -> None:
    from . import db
    async with db.get_pool().connection() as conn:
        await conn.execute(
            """INSERT INTO thread_summaries (thread_id, summary, pinned, covered, boundary, private, updated_at)
               VALUES (%s,%s,%s,%s,%s,%s, now())
               ON CONFLICT (thread_id) DO UPDATE SET summary = EXCLUDED.summary, pinned = EXCLUDED.pinned,
                 covered = EXCLUDED.covered, boundary = EXCLUDED.boundary,
                 private = thread_summaries.private OR EXCLUDED.private, updated_at = now()""",
            (thread_id, summary, pinned, covered, boundary, private))


# ---------------------------------------------------------------- when to compact
def plan(messages: list[AnyMessage], summary: dict[str, Any] | None, *, topic_changed: bool,
         c: dict[str, Any] | None = None) -> int | None:
    """How many recent owner turns to keep verbatim if the thread should be compacted now, else None."""
    c = c if c is not None else cfg()
    hs = human_idx(messages)
    start = boundary_index(messages, summary)
    covered = 0 if start is None else len(human_idx(messages[:start]))
    keep = int(c.get("keep_turns", 6))
    if len(hs) - covered - keep >= int(c.get("compact_every", 10)):
        return keep
    verbatim = "".join(_text(m.content) for m in messages[start or 0:]
                       if isinstance(m, (HumanMessage, AIMessage)) and not isinstance(m, ToolMessage))
    if len(hs) - covered > keep and est_tokens(verbatim) > int(c.get("max_tokens", 12000)):
        return keep
    tkeep = int(c.get("topic_keep_turns", 2))
    if topic_changed and len(hs) - covered > tkeep:
        return tkeep
    return None


async def after_turn(redis, *, thread_id: str, messages: list[AnyMessage], task_type: str | None,
                     private: bool) -> int | None:
    """Called by run_agent after a finished turn: queue `compact_thread` when one of the triggers fires."""
    c = cfg()
    if not c.get("enabled") or thread_id.startswith("eval:"):
        return None
    topic_changed = False
    if task_type and task_type not in set(c.get("topic_ignore_types") or ["chitchat"]):
        key = LAST_TYPE.format(thread_id)
        prev = await redis.get(key)
        prev = prev.decode() if isinstance(prev, bytes) else prev
        topic_changed = bool(prev) and prev != task_type
        await redis.set(key, task_type, ex=30 * 86400)
    keep = plan(messages, await load(thread_id, private=True), topic_changed=topic_changed, c=c)
    if keep is None:
        return None
    pol = config.policy()
    queue = (pol.get("queues") or {}).get("background", {}).get("name", "arq:background")
    n = len(human_idx(messages))
    await redis.enqueue_job("compact_thread", {"thread_id": thread_id, "keep": keep, "private": private},
                            _job_id=f"compact:{thread_id}:{n}:{keep}", _queue_name=queue)
    return keep


# ---------------------------------------------------------------- summariser
SUMMARY_SYSTEM = """You maintain the running summary of a long conversation between the owner and their assistant.
Update the previous summary with the newly folded turns. Reply with the updated summary only, at most {max_chars}
characters, in the conversation's main language (Chinese: Traditional Chinese, Taiwan wording). Use these sections,
headings translated into that language, and omit empty ones:
## Constraints & preferences: every constraint, preference or instruction the owner gave (budget, place, dates,
format, "don't …"), close to the owner's words. Never drop one unless the owner withdrew it.
## Done: actions the assistant completed (events, todos, notes, files), with names and dates.
## Key points: one line per topic, "question → conclusion", keeping exact numbers, names, dates, prices, versions
and decisions.
## Sources: "title — URL" of pages the answers relied on, one per line.
## Corrections: answers the owner said were wrong, as "X was wrong; correct: Y". Do not repeat the wrong answer
anywhere else.
## Open: questions still unanswered.
Leave out greetings, thanks, small talk, tool output, reasoning and anything later turns superseded. If space runs
out, shorten the oldest key points first; never drop constraints."""


def transcript(messages: list[AnyMessage], *, pin_min: int, max_chars_each: int = 2000) -> tuple[str, str]:
    """(plain transcript of the folded turns, latest long owner paste in them or '')."""
    pinned_i, pinned = -1, ""
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage) and len(_text(m.content)) >= pin_min:
            pinned_i, pinned = i, _text(m.content)
    lines = []
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            t = _text(m.content).strip()
            if i == pinned_i:
                t = t[:300] + " … [long text, kept verbatim separately]"
            lines.append(f"Owner: {t[:max_chars_each]}")
        elif isinstance(m, AIMessage) and not isinstance(m, ToolMessage):
            t = _THINK.sub("", _text(m.content)).strip()
            if t:
                lines.append(f"Assistant: {t[:max_chars_each]}")
    return "\n\n".join(lines), pinned


async def summarize(llm, model: str, *, previous: str, messages: list[AnyMessage], c: dict[str, Any]) -> tuple[str, str]:
    text, pinned = transcript(messages, pin_min=int(c.get("pin_min_chars", 1500)))
    max_chars = int(c.get("summary_max_chars", 3000))
    human = f"Previous summary:\n{previous.strip() or '(none)'}\n\nNewly folded turns:\n{text}"
    kw = {"extra_body": {"think": False}} if model in ("local", "local_cpu") else {}
    msg = await llm.bind(max_tokens=max(600, max_chars), **kw).ainvoke(
        [SystemMessage(SUMMARY_SYSTEM.format(max_chars=max_chars)), HumanMessage(human)],
        config={"tags": ["internal", "history"], "run_name": "history_compact"})
    out = _THINK.sub("", _text(msg.content)).strip()
    return out[: max_chars * 2], pinned[: int(c.get("pin_max_chars", 4000))]


async def compact_thread(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Background job: fold the owner turns before the last `keep` into the thread's running summary."""
    from . import state
    from .jobs import memory_jobs
    from .llm.client import make_llm
    r = state.get_redis()
    tid = payload["thread_id"]
    if why := await memory_jobs._wait_reason(r):
        return await memory_jobs._later(ctx, "compact_thread", payload, f"compact:{tid}", why)
    if await state.get_pending(r, tid):
        return {"status": "skipped", "reason": "question pending"}
    c = cfg()
    msgs = await memory_jobs.thread_messages(ctx["checkpointer"], tid)
    hs = human_idx(msgs)
    prev = await load(tid, private=True)
    start = boundary_index(msgs, prev)
    if start is None:
        prev, start = None, 0
    target = len(hs) - int(payload.get("keep", c.get("keep_turns", 6)))
    end = hs[target] if 0 <= target < len(hs) else None
    if end is None or end <= start:
        return {"status": "skipped", "reason": "nothing to fold"}
    model = str(c.get("model", "local"))
    private = bool(payload.get("private")) or await state.is_thread_private(r, tid) or bool(prev and prev["private"])
    if private:
        model = "local"
    t0 = time.monotonic()
    llm = make_llm(model, await state.get_killswitch(r), timeout_s=180)
    sem = (state.GpuSemaphore(r, int(config.policy().get("gpu_concurrency", 2))) if model in ("local", "local_cpu")
           else contextlib.nullcontext())
    async with sem:
        summary, pinned = await summarize(llm, model, previous=(prev or {}).get("summary", ""),
                                          messages=msgs[start:end], c=c)
    if not summary:
        return {"status": "empty"}
    await save(tid, summary=summary, pinned=pinned or (prev or {}).get("pinned", ""), covered=target,
               boundary=_sha(_text(msgs[hs[target - 1]].content)), private=private)
    log.info("compacted %s: %d owner turns summarised (%d chars, %d ms)", tid, target, len(summary),
             int((time.monotonic() - t0) * 1000))
    return {"status": "done", "covered": target, "chars": len(summary)}
