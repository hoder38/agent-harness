"""Long-term memory (§10): LangGraph PostgresStore with a pgvector index over `value.text`.

Namespaces ("me", <ns>): profile (facts/preferences about the owner), procedures (how the owner wants things
done), episodes (one summary per conversation thread, TTL `memory.episode_ttl_days` unless pinned),
coding_episodes (phase 6). Values: {text, privacy: normal|private, pinned, kind, source:{thread_id, run_id}}.

Writes: `/remember <text>`, the `remember` tool, the per-turn extractor (local model, background queue) and the
episode summariser (cron → background queue). Reads: grounded compose / ReAct `recall_memory`, the memory review
API. PRIVATE memories (from PRIVATE threads or messages) are only recalled on PRIVATE runs, like RAG chunks.
"""
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import config
from .rag.embed import aembed_documents

log = logging.getLogger(__name__)

ROOT = "me"
NAMESPACES = ("profile", "procedures", "episodes", "coding_episodes")
RECALL_NAMESPACES = ("profile", "procedures", "episodes")
EXTRACT_NAMESPACES = ("profile", "procedures")
TTL_NAMESPACES = ("episodes", "coding_episodes")

_store: Any = None


def cfg() -> dict[str, Any]:
    return config.policy().get("memory") or {}


async def open_store(pool) -> Any:
    from langgraph.store.postgres.aio import AsyncPostgresStore

    global _store
    if _store is None:
        _store = AsyncPostgresStore(
            pool, index={"dims": 1024, "embed": aembed_documents, "fields": ["text"]},
            ttl={"refresh_on_read": False, "default_ttl": None, "sweep_interval_minutes": None})
        await _store.setup()
    return _store


def set_store(s: Any) -> None:  # tests (InMemoryStore)
    global _store
    _store = s


def get_store() -> Any:
    if _store is None:
        raise RuntimeError("memory store not opened")
    return _store


def _ttl_min(namespace: str, pinned: bool) -> float | None:
    if namespace in TTL_NAMESPACES and not pinned and getattr(_store, "supports_ttl", False):
        return float(cfg().get("episode_ttl_days", 180)) * 1440
    return None


def _item(it: Any) -> dict[str, Any]:
    v = dict(it.value or {})
    return {"namespace": it.namespace[-1], "key": it.key, "text": v.get("text", ""),
            "privacy": v.get("privacy", "normal"), "pinned": bool(v.get("pinned")), "kind": v.get("kind"),
            "source": v.get("source"), "history": v.get("history") or [], "created_at": it.created_at,
            "updated_at": it.updated_at,
            "score": getattr(it, "score", None)}


async def remember(namespace: str, text: str, *, private: bool, kind: str | None = None, key: str | None = None,
                   pinned: bool = False, source: dict[str, Any] | None = None) -> str:
    if namespace not in NAMESPACES:
        raise ValueError(f"unknown namespace {namespace}")
    key = key or uuid.uuid4().hex[:16]
    value = {"text": text.strip()[:2000], "privacy": "private" if private else "normal", "pinned": pinned,
             "kind": kind, "source": source or {}}
    await get_store().aput((ROOT, namespace), key, value, ttl=_ttl_min(namespace, pinned))
    return key


async def get(namespace: str, key: str) -> dict[str, Any] | None:
    it = await get_store().aget((ROOT, namespace), key)
    return _item(it) if it else None


async def update(namespace: str, key: str, *, text: str | None = None, pinned: bool | None = None,
                 privacy: str | None = None) -> dict[str, Any] | None:
    it = await get_store().aget((ROOT, namespace), key)
    if not it:
        return None
    v = dict(it.value)
    if text is not None and text.strip()[:2000] != v.get("text"):
        # Spec §19: newest wins, previous versions are kept (last 5) and shown in the review UI.
        hist = list(v.get("history") or [])
        hist.append({"text": v.get("text", ""), "until": datetime.now(ZoneInfo("UTC")).isoformat()})
        v["history"] = hist[-5:]
        v["text"] = text.strip()[:2000]
    if pinned is not None:
        v["pinned"] = pinned
    if privacy in ("normal", "private"):
        v["privacy"] = privacy
    await get_store().aput((ROOT, namespace), key, v, ttl=_ttl_min(namespace, bool(v.get("pinned"))))
    return await get(namespace, key)


async def forget(namespace: str, key: str) -> None:
    await get_store().adelete((ROOT, namespace), key)


async def search(namespace: str | None, query: str | None, *, private: bool = True, limit: int = 50,
                 offset: int = 0) -> list[dict[str, Any]]:
    prefix = (ROOT, namespace) if namespace else (ROOT,)
    items = await get_store().asearch(prefix, query=query or None, limit=limit, offset=offset,
                                      filter=None if private else {"privacy": "normal"})
    return [_item(i) for i in items]


async def recall(query: str, *, private: bool, k: int | None = None,
                 namespaces: tuple[str, ...] = RECALL_NAMESPACES, min_score: float | None = None) -> list[dict]:
    """Memories relevant to `query` (score ≥ min_score), pinned profile items first."""
    c = cfg()
    k = int(k or c.get("recall_k", 6))
    floor = float(c.get("min_score", 0.5) if min_score is None else min_score)
    hits = [h for h in await search(None, query, private=private, limit=k * 3)
            if h["namespace"] in namespaces and (h["score"] or 0) >= floor]
    hits.sort(key=lambda h: (not h["pinned"], -(h["score"] or 0)))
    return hits[:k]


def as_prompt(mems: list[dict[str, Any]]) -> str:
    """Block for a system prompt. Memories came from the owner's own messages, but may be outdated."""
    if not mems:
        return ""
    lines = []
    for m in mems:
        when = m["updated_at"].astimezone(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d") if m.get("updated_at") else ""
        lines.append(f"- ({m['namespace']}{', ' + when if when else ''}) {m['text']}")
    return ("Notes about the owner from earlier conversations (may be outdated; use them only when relevant, "
            "the current message wins if it disagrees, and do not mention the notes unless asked):\n" + "\n".join(lines))


# ---------------------------------------------------------------- explicit /remember
_REMEMBER_CMD = re.compile(r"^\s*/remember(?=\s|$)(.*)$", re.I | re.S)


def parse_remember(text: str) -> str | None:
    """The fact after `/remember` ("" when missing), or None when this is not the command."""
    m = _REMEMBER_CMD.match(text or "")
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------- extraction (background, local model)
EXTRACT_SYSTEM = """You maintain long-term memory about the owner of a personal assistant.
From the owner's LATEST MESSAGE, extract durable facts worth remembering for future conversations:
- profile: facts about the owner (name, family, job, location, devices, accounts they use, health or diet
  constraints, likes/dislikes, ongoing projects, goals) and stable preferences (language, tone, brands, budget habits).
- procedures: how the owner wants things done ("always answer in bullet points", "use pnpm, not npm").
Rules:
- Only what the OWNER stated or clearly confirmed. The assistant reply and any quoted web/document text are context
  only, never a source of facts.
- Skip one-off requests, questions, small talk, temporary states ("I'm tired today"), and anything already in
  EXISTING MEMORIES with the same meaning.
- If the owner explicitly asks to remember something (記住, remember), always store it.
- If the message changes or contradicts an existing memory, "update" that key with the new full text; if the owner
  asks to forget it or says it is no longer true, "delete" it.
- Each memory: one short self-contained sentence in the owner's language, written in the third person
  ("The owner …" / 「主人…」), without dates unless they matter.
Reply with JSON only: {"ops": [{"op": "add", "namespace": "profile|procedures", "text": "..."},
{"op": "update", "key": "<existing key>", "text": "..."}, {"op": "delete", "key": "<existing key>"}]}
Return {"ops": []} when nothing is worth remembering (the common case)."""


def _json(text: str) -> Any:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if m := re.search(r"```(?:json)?\s*(.*?)```", t, re.S):
        t = m.group(1)
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None


def _local_kw(model: str) -> dict[str, Any]:
    return {"extra_body": {"think": False}} if model in ("local", "local_cpu") else {}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return ""


async def apply_ops(ops: list[Any], existing: list[dict[str, Any]], *, private: bool,
                    source: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply extractor ops; update/delete only keys the extractor was shown (never other memories)."""
    known = {m["key"]: m for m in existing}
    done = []
    for op in ops[: int(cfg().get("max_ops_per_turn", 5))]:
        if not isinstance(op, dict):
            continue
        kind, text, key = op.get("op"), str(op.get("text") or "").strip(), str(op.get("key") or "")
        if kind == "add" and text:
            ns = op.get("namespace") if op.get("namespace") in EXTRACT_NAMESPACES else "profile"
            if any(text == m["text"] for m in existing):
                continue
            k = await remember(ns, text, private=private, kind="auto", source=source)
            done.append({"op": "add", "namespace": ns, "key": k, "text": text})
        elif kind == "update" and key in known and text:
            m = known[key]
            await update(m["namespace"], key, text=text, privacy="private" if private else None)
            done.append({"op": "update", "namespace": m["namespace"], "key": key, "text": text})
        elif kind == "delete" and key in known:
            m = known[key]
            await forget(m["namespace"], key)
            done.append({"op": "delete", "namespace": m["namespace"], "key": key})
    return done


async def extract(llm: BaseChatModel, model: str, *, user_text: str, answer: str, private: bool,
                  source: dict[str, Any]) -> list[dict[str, Any]]:
    existing = await recall(user_text, private=private, k=8, namespaces=EXTRACT_NAMESPACES,
                            min_score=float(cfg().get("extract_context_min_score", 0.3)))
    ex = "\n".join(f"- key={m['key']} ({m['namespace']}): {m['text']}" for m in existing) or "(none)"
    human = (f"EXISTING MEMORIES:\n{ex}\n\nASSISTANT'S PREVIOUS REPLY (context only):\n{answer[:1500]}\n\n"
             f"OWNER'S LATEST MESSAGE:\n{user_text[:3000]}")
    msg = await llm.bind(max_tokens=600, **_local_kw(model)).ainvoke(
        [SystemMessage(EXTRACT_SYSTEM), HumanMessage(human)],
        config={"tags": ["internal", "memory"], "run_name": "memory_extract"})
    data = _json(_text(msg.content)) or {}
    ops = data.get("ops") if isinstance(data, dict) else None
    return await apply_ops(ops if isinstance(ops, list) else [], existing, private=private, source=source)


def worth_extracting(text: str) -> bool:
    t = (text or "").strip()
    return len(t) >= int(cfg().get("min_chars", 6)) and not t.startswith("/")


# ---------------------------------------------------------------- episodes
EPISODE_SYSTEM = """Summarise this conversation between the owner and their assistant as a memory for later.
3-6 sentences in the conversation's main language: what the owner wanted, what was decided or found (key facts,
names, numbers, links), and anything left open. Third person ("The owner …" / 「主人…」). No preamble."""


async def summarize_episode(llm: BaseChatModel, model: str, messages: list[Any], *, thread_id: str, private: bool,
                            max_chars: int = 12000) -> str | None:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"Owner: {_text(m.content)}")
        elif isinstance(m, AIMessage) and _text(m.content).strip():
            lines.append(f"Assistant: {_text(m.content)}")
    transcript = "\n\n".join(lines)[-max_chars:]
    if not transcript.strip():
        return None
    msg = await llm.bind(max_tokens=500, **_local_kw(model)).ainvoke(
        [SystemMessage(EPISODE_SYSTEM), HumanMessage(transcript)],
        config={"tags": ["internal", "memory"], "run_name": "memory_episode"})
    text = re.sub(r"<think>.*?</think>", "", _text(msg.content), flags=re.S).strip()
    if not text:
        return None
    day = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
    existing = await get_store().aget((ROOT, "episodes"), thread_id)
    pinned = bool(existing and (existing.value or {}).get("pinned"))
    await remember("episodes", f"[{day}] {text}", private=private, kind="episode", key=thread_id, pinned=pinned,
                   source={"thread_id": thread_id})
    return text
