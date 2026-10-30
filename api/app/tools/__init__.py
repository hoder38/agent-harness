"""MVP tools. Every tool declares whether it is risky (approval) — none are in the MVP."""
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


def tools_for(route: str, *, private: bool = False, local: bool = False) -> list:
    tools = list(ROUTE_TOOLS.get(route, []))
    if route in TODO_ROUTES and local:   # calendar content is PRIVATE: local models only
        from .. import config, gcal

        if (config.policy().get("calendar") or {}).get("enabled") and gcal.available():
            from .calendar import CALENDAR_TOOLS

            tools += CALENDAR_TOOLS
    if route in TODO_ROUTES:
        from .todos import TODO_TOOLS

        tools += TODO_TOOLS
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
    # Web tools send text off the machine, so PRIVATE threads never get them.
    if route in WEB_ROUTES and not private:
        from .web import fetch_page, web_search

        if get_settings().brave_api_key:
            tools.append(web_search)
        tools.append(fetch_page)
    return tools
