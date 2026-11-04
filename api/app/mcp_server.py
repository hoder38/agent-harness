"""`harness-memory` MCP server (§9c): the owner's harness memory for Copilot CLI over SSH.

Served by agent-api at `/mcp` (streamable HTTP, host-only port 127.0.0.1:8100, harness token). Copilot is a cloud
model, so PRIVATE memories and vault notes are never returned (NORMAL text is Gate 0 redacted).
"""
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from . import config, copilot_ingest, db, inbox, memory, todos
from .auth import require_harness_token

log = logging.getLogger(__name__)

MCP_PATH = "/mcp"
SEARCH_NAMESPACES = ("profile", "procedures", "episodes", "coding_episodes")


def _line(m: dict[str, Any]) -> str:
    when = m["updated_at"].strftime("%Y-%m-%d") if m.get("updated_at") else ""
    return f"- [{m['namespace']}{', ' + when if when else ''}] {m['text']}"


async def recall(query: str, k: int | None = None) -> str:
    """Search the owner's long-term memory (profile, preferences, past conversations and coding sessions)."""
    c = config.policy().get("copilot_ingest") or {}
    k = max(1, min(int(k or c.get("mcp_recall_k", 6)), 20))
    hits = await memory.recall(query, private=False, k=k, namespaces=SEARCH_NAMESPACES)
    await db.audit("memory", "mcp_recall", {"keys": [h["key"] for h in hits]})
    return "\n".join(_line(h) for h in hits) or "No relevant memories."


async def remember(fact: str) -> str:
    """Save a durable fact or preference about the owner (one short sentence)."""
    fact = (fact or "").strip()
    if not fact:
        return "Nothing to remember."
    key = await memory.remember("profile", fact, private=False, kind="copilot",
                                source={"channel": "mcp"})
    await db.audit("memory", "mcp_remember", {"namespace": "profile", "key": key})
    await inbox.memory_written("add", "profile", key, fact, source="copilot")
    return f"Remembered (key {key})."


async def get_thread_summary(thread_id: str) -> str:
    """Summary of a conversation: `copilot:<sessionId>` for a Copilot session, or a web/LINE thread id."""
    sid = thread_id.removeprefix(copilot_ingest.THREAD_PREFIX)
    candidates = [("coding_episodes", sid), ("episodes", thread_id)]
    for ns, key in candidates:
        m = await memory.get(ns, key)
        if m and m["privacy"] != "private":
            return m["text"]
    return "No summary for that thread."


async def list_todos(status: str = "open") -> str:
    """The owner's todo list (status: open | done | all), due soonest first."""
    rows = await todos.list_(status if status in ("open", "done", "all") else "open", 50)
    return "\n".join(todos.line(t) for t in rows) or "No todos."


async def add_todo(title: str, due: str | None = None, notes: str = "") -> str:
    """Add a todo for the owner. `due` is YYYY-MM-DD or YYYY-MM-DDTHH:MM (Asia/Taipei)."""
    try:
        t = await todos.add(title, notes=notes, due_at=todos.parse_due(due), channel="copilot")
    except ValueError as e:
        return f"Not added: {e}"
    await db.audit("todo", "mcp_add", {"id": t["id"]})
    return f"Added todo #{t['id']}: {t['title']}"


async def search_notes(query: str, k: int = 6) -> str:
    """Search the owner's Obsidian vault notes (§10a). Each hit has a cite label [[Note#Heading]]; read the whole
    note with read_note. Private notes are never returned."""
    if not (config.policy().get("vault") or {}).get("enabled"):
        return "The vault is turned off."
    from .rag import store

    hits = await store.search(query, private=False, k=max(1, min(int(k or 6), 20)), source="vault")
    await db.audit("vault", "mcp_search", {"paths": [h["path"] for h in hits]})
    if not hits:
        return "No matching notes."
    return '<untrusted source="vault">\n' + store.to_json(hits) + "\n</untrusted>"


async def read_note(path: str) -> str:
    """Read one note of the owner's Obsidian vault by path ("Projects/Harness.md") or name ("Harness").
    Private notes are refused."""
    if not (config.policy().get("vault") or {}).get("enabled"):
        return "The vault is turned off."
    from .rag import vault

    target = path.strip().removeprefix("[[").removesuffix("]]").split("#", 1)[0].split("|", 1)[0]
    text, err = await vault.read_note(target, private=False)
    await db.audit("vault", "mcp_read", {"path": target[:200], "ok": not err})
    return f"Cannot read {path!r}: {err}" if err else f'<untrusted source="vault">\n{text}\n</untrusted>'


def build() -> FastMCP:
    srv = FastMCP(
        "harness-memory", stateless_http=True, json_response=True, streamable_http_path=MCP_PATH,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=["127.0.0.1:*", "localhost:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*"]))
    for fn in (recall, remember, get_thread_summary, list_todos, add_todo, search_notes, read_note):
        srv.tool()(fn)
    return srv


class TokenGate:
    """ASGI wrapper: only `/mcp` is served, and only with the harness token."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        status = 404 if scope["path"].rstrip("/") != MCP_PATH else 0
        if not status:
            try:
                await require_harness_token(headers.get("authorization"))
            except Exception:
                status = 401
        if status:
            await send({"type": "http.response.start", "status": status, "headers": [(b"content-length", b"0")]})
            return await send({"type": "http.response.body", "body": b""})
        return await self.app(scope, receive, send)
