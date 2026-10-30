"""ReAct tools over the knowledge base and long-term memory (§10). Built per run: the PRIVATE flag decides
whether private documents/memories are visible and how new memories are tagged."""
from langchain_core.tools import tool

from .. import db, inbox, memory
from ..rag import store


def kb_tools(*, private: bool) -> list:
    @tool
    async def search_knowledge(query: str) -> str:
        """Search the owner's own notes and documents. Returns passages with a `cite` label (path#heading);
        cite them as 📄 path#heading when you use them."""
        hits = await store.search(query, private=private)
        return store.to_json(hits) if hits else "No matching passages in the owner's documents."

    @tool
    async def recall_memory(query: str) -> str:
        """Look up what is remembered about the owner (facts, preferences, past conversations)."""
        mems = await memory.recall(query, private=private)
        return memory.as_prompt(mems) or "Nothing relevant remembered."

    @tool
    async def remember(text: str, kind: str = "profile") -> str:
        """Save a durable fact or preference the owner stated (kind: profile | procedures). Only use when the
        owner asks you to remember something or states a lasting preference."""
        ns = kind if kind in memory.EXTRACT_NAMESPACES else "profile"
        key = await memory.remember(ns, text, private=private, kind="tool")
        await db.audit("memory", "tool", {"namespace": ns, "key": key, "private": private})
        await inbox.memory_written("add", ns, key, text, private=private, source="tool")
        return f"Remembered ({ns})."

    return [search_knowledge, recall_memory, remember]
