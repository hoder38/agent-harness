"""Obsidian vault tools (§10a). read_note: knowledge routes. Writes: assistant route; only `Inbox/` and `AI/` are
written directly, any other note gets an Inbox approval with the diff (app/vault_writer.py)."""
from langchain_core.tools import tool

from .. import db, vault_writer as vw
from ..rag import vault


def read_tools(*, private: bool) -> list:
    @tool
    async def read_note(path: str) -> str:
        """Read a whole note from the owner's Obsidian vault by its path (e.g. "Projects/Harness.md") or note name,
        as given in a search_knowledge cite label [[path#heading]] (without the #heading)."""
        target = path.strip().removeprefix("[[").removesuffix("]]").split("#", 1)[0].split("|", 1)[0]
        text, err = await vault.read_note(target, private=private)
        if err:
            return f"Cannot read {path!r}: {err}"
        return f'<untrusted source="vault">\n{text}\n</untrusted>'

    return [read_note]


def write_tools(*, private: bool) -> list:
    @tool
    async def capture_note(text: str, title: str = "") -> str:
        """Save a new note in the owner's Obsidian vault Inbox/ folder (for later sorting by the owner). Use when
        the owner asks to save/capture/jot down something as a note."""
        try:
            rel = vw.capture(text, title=title, source="agent", private=private)
        except vw.VaultWriteError as e:
            return f"Not saved: {e}"
        await db.audit("vault", "capture", {"path": rel, "private": private})
        return f"Saved to the vault as [[{rel.removesuffix('.md')}]]."

    @tool
    async def append_daily_note(entry: str) -> str:
        """Add a one-line, timestamped entry to today's daily note (AI/Daily/YYYY-MM-DD.md) in the owner's vault."""
        try:
            rel = vw.append_daily(entry, private=private)
        except vw.VaultWriteError as e:
            return f"Not saved: {e}"
        await db.audit("vault", "daily", {"path": rel, "private": private})
        return f"Added to [[{rel.removesuffix('.md')}]]."

    @tool
    async def edit_note(path: str, old_text: str, new_text: str, reason: str = "") -> str:
        """Change a note in the owner's vault: replace `old_text` (copied exactly from read_note, must occur once)
        with `new_text`; empty `old_text` creates a new note at `path`. Notes under Inbox/ and AI/ change at once;
        any other note is only PROPOSED and needs the owner's approval in the Inbox."""
        try:
            return await vw.propose_edit(path, old_text, new_text, reason=reason, private=private)
        except vw.VaultWriteError as e:
            return f"Not changed: {e}"

    return [capture_note, append_daily_note, edit_note]


VAULT_WRITE_TOOLS = {"capture_note", "append_daily_note", "edit_note"}
