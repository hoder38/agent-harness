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
    "assistant": [get_current_time],   # todos / memory / RAG tools arrive in phases 7 and 10
    "clarify": [],
    "coding": [],
}
WEB_ROUTES = {"chat", "assistant"}


def tools_for(route: str, *, private: bool = False) -> list:
    tools = list(ROUTE_TOOLS.get(route, []))
    # Web tools send text off the machine, so PRIVATE threads never get them.
    if route in WEB_ROUTES and not private:
        from .web import fetch_page, web_search

        if get_settings().brave_api_key:
            tools.append(web_search)
        tools.append(fetch_page)
    return tools
