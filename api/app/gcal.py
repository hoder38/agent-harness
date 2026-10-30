"""Google Calendar client (§10 calendar). Talks to googleapis.com through the strict egress allowlist.

The OAuth grant (client id/secret + refresh token, scope calendar.events) is written on the host by
host/google_auth.py and mounted read-only; only short-lived access tokens are cached here, in memory.
Calendar content is PRIVATE: callers must only hand it to local models.
"""
import json
import time
from datetime import datetime, timedelta
from typing import Any

import httpx

from . import todos
from .settings import get_settings

API = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
TOKEN_URL = "https://oauth2.googleapis.com/token"
_tok: dict[str, Any] = {"value": None, "exp": 0.0}


class CalendarError(Exception):
    pass


def _creds() -> dict[str, str]:
    try:
        with open(get_settings().google_credentials_path) as f:
            c = json.load(f)
    except (OSError, ValueError) as e:
        raise CalendarError("Google Calendar is not connected (run host/google_auth.py)") from e
    if not all(c.get(k) for k in ("client_id", "client_secret", "refresh_token")):
        raise CalendarError("Google Calendar credentials file is incomplete")
    return c


def available() -> bool:
    try:
        _creds()
        return True
    except CalendarError:
        return False


async def _access_token(http: httpx.AsyncClient) -> str:
    if _tok["value"] and time.monotonic() < _tok["exp"]:
        return _tok["value"]
    c = _creds()
    r = await http.post(TOKEN_URL, data={"client_id": c["client_id"], "client_secret": c["client_secret"],
                                         "refresh_token": c["refresh_token"], "grant_type": "refresh_token"})
    if r.status_code != 200:
        raise CalendarError("Google rejected the refresh token; re-run host/google_auth.py")
    d = r.json()
    _tok.update(value=d["access_token"], exp=time.monotonic() + int(d.get("expires_in", 3600)) - 120)
    return _tok["value"]


async def _call(method: str, url: str = API, **kw: Any) -> Any:
    async with httpx.AsyncClient(timeout=15) as http:
        token = await _access_token(http)
        r = await http.request(method, url, headers={"Authorization": f"Bearer {token}"}, **kw)
    if r.status_code == 404 or r.status_code == 410:
        raise CalendarError("no such event")
    if r.status_code >= 400:
        raise CalendarError(f"Google Calendar error {r.status_code}")
    return r.json() if r.content else {}


def when(value: str) -> datetime:
    """Event times need an explicit time of day (naive values are Asia/Taipei)."""
    v = (value or "").strip()
    if ":" not in v:
        raise ValueError(f"bad time {value!r}: use YYYY-MM-DDTHH:MM or 'tomorrow 09:00'")
    return todos.parse_due(v)


def day_range(start: str | None, days: int) -> tuple[datetime, datetime]:
    v = (start or "today").strip() or "today"
    d = todos.parse_due(v)   # a bare day resolves to 23:59 of that day; only the date matters here
    d = d.astimezone(todos.TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    return d, d + timedelta(days=max(1, min(int(days), 31)))


def _fmt_time(ev: dict[str, Any]) -> str:
    s, e = ev.get("start", {}), ev.get("end", {})
    if "date" in s:
        return f"{s['date']} (all day)"
    a = datetime.fromisoformat(s["dateTime"]).astimezone(todos.TZ)
    b = datetime.fromisoformat(e["dateTime"]).astimezone(todos.TZ)
    return f"{a:%Y-%m-%d %H:%M}-{b:%H:%M}"


def line(ev: dict[str, Any]) -> str:
    where = f" @ {ev['location']}" if ev.get("location") else ""
    return f"{ev['id']} | {_fmt_time(ev)} | {' '.join((ev.get('summary') or '(no title)').split())}{where}"


async def list_events(start: datetime, end: datetime, limit: int = 50) -> list[dict[str, Any]]:
    d = await _call("GET", params={"timeMin": start.isoformat(), "timeMax": end.isoformat(), "singleEvents": "true",
                                   "orderBy": "startTime", "maxResults": limit})
    return d.get("items", [])


async def get_event(event_id: str) -> dict[str, Any]:
    return await _call("GET", f"{API}/{event_id}")


async def insert_event(title: str, start: datetime, end: datetime, notes: str = "") -> dict[str, Any]:
    body = {"summary": title, "description": notes,
            "start": {"dateTime": start.isoformat(), "timeZone": "Asia/Taipei"},
            "end": {"dateTime": end.isoformat(), "timeZone": "Asia/Taipei"}}
    return await _call("POST", json=body)


async def delete_event(event_id: str) -> None:
    await _call("DELETE", f"{API}/{event_id}")
