"""MVP tools. Every tool declares whether it is risky (approval) — none are in the MVP."""
import re
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

from ..settings import get_settings


@tool
def get_current_time(timezone: str = "Asia/Taipei") -> str:
    """Return the current date and time in the given IANA timezone (default Asia/Taipei)."""
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return f"Unknown timezone {timezone!r}"
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


ROUTE_TOOLS = {
    "chat": [get_current_time],
    "assistant": [get_current_time],   # + knowledge/memory (phase 10) and todo (phase 7) tools below
    "clarify": [],
    "coding": [],
}
WEB_ROUTES = {"chat", "assistant"}
KB_ROUTES = {"chat", "assistant"}
TODO_ROUTES = {"assistant"}


def tools_for(route: str, *, private: bool = False) -> list:
    tools = list(ROUTE_TOOLS.get(route, []))
    if route in TODO_ROUTES:   # cloud routes too (§6, 2026-11-05); writes are Inbox proposals
        from .. import config, gcal

        if (config.policy().get("calendar") or {}).get("enabled") and gcal.available():
            from .calendar import CALENDAR_TOOLS

            tools += CALENDAR_TOOLS
    if route in TODO_ROUTES:
        from .todos import TODO_TOOLS

        tools += TODO_TOOLS
    from .. import config

    vault_on = bool((config.policy().get("vault") or {}).get("enabled"))
    if route in TODO_ROUTES and vault_on:
        from .vault import write_tools

        tools += write_tools(private=private)
    if route in KB_ROUTES:  # local services only: PRIVATE threads get them too
        from .. import config
        from .knowledge import kb_tools

        pol = config.policy()
        kt = kb_tools(private=private)
        if not (pol.get("rag") or {}).get("enabled"):
            kt = [t for t in kt if t.name != "search_knowledge"]
        if not (pol.get("memory") or {}).get("enabled"):
            kt = [t for t in kt if t.name == "search_knowledge"]
        tools += kt
        if vault_on:
            from .vault import read_tools

            tools += read_tools(private=private)
    # Web tools send text off the machine, so PRIVATE threads never get them.
    if route in WEB_ROUTES and not private:
        from .web import fetch_page, web_search

        if get_settings().brave_api_key:
            tools.append(web_search)
        tools.append(fetch_page)
    return tools


_TODO_WORDS = re.compile(r"待辦|代辦|todo|提醒|記得|to-do", re.I)
_ADD_WORDS = re.compile(r"^\s*(?:/\w+\s+)?(?:幫我|請)?(?:加上|加入|新增|安排|預約|約|add)|開會|會議|行程|約會|預約")
_QUERY = re.compile(r"[?？]|什麼|哪些|有沒有|查|列出|顯示|list|what|完成了")
_SLOT = re.compile(r"\d{1,2}\s*[:點時]|早上|上午|中午|下午|晚上|凌晨|\d\s*(?:am|pm)", re.I)


def required_tool(tools: list, text: str) -> str | None:
    """The write tool a clear request must end up calling (the model is made to call it or the run says it failed)."""
    names = {t.name for t in tools}
    if "add_event" in names and not _TODO_WORDS.search(text) and _ADD_WORDS.search(text) and _SLOT.search(text):
        return "add_event"
    if "add_todo" in names and not _QUERY.search(text) and (
            re.search(r"新增待辦|加入待辦|新增代辦|add (?:a )?todo", text, re.I) or _TODO_WORDS.search(text)
            or re.search(r"\d{1,2}\s*[/月]\s*\d{1,2}\s*日?\s*前|前.{0,3}(?:繳|交|付|買|完成|處理)", text)):
        return "add_todo"
    return None


def narrow_for_intent(tools: list, text: str) -> list:
    """A small local model files every "add ..." as a todo, so decide the obvious cases here, without a model:
    an add request with a time of day (and no todo word) is a calendar event; an explicit todo/reminder is not."""
    names = {t.name for t in tools}
    if "add_event" not in names:
        return tools
    if _TODO_WORDS.search(text):
        return [t for t in tools if t.name != "add_event"]
    if _ADD_WORDS.search(text) and _SLOT.search(text):
        return [t for t in tools if t.name != "add_todo"]
    return tools


WRITE_TOOLS = {"add_todo", "complete_todo", "add_event", "delete_event", "remember", "capture_note",
               "append_daily_note", "edit_note"}
_REMEMBER = re.compile(r"記住|記下|remember|筆記|日記|存成|存到|note|obsidian|vault", re.I)


def for_attachments(tools: list, owner_text: str) -> list:
    """§16: attached files are data. Write tools stay only when the owner's own words ask for such an action
    ("把會議紀錄裡的待辦加進清單"), never because the file's text says so."""
    if _TODO_WORDS.search(owner_text) or _ADD_WORDS.search(owner_text) or _REMEMBER.search(owner_text):
        return tools
    return [t for t in tools if t.name not in WRITE_TOOLS]
