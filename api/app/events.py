"""Inbound events (phase 11, §4 Events): `POST /events/{source}`, HMAC-signed, recorded and put in the Inbox.

Each source needs a secret in EVENT_SECRETS (JSON, e.g. '{"github": "..."}') and an entry under policy.yaml
`events.sources`; anything else is a 404. The body is signed GitHub-style: `X-Hub-Signature-256` or
`X-Signature-256` = "sha256=" + HMAC-SHA256(secret, raw body). An event never starts an agent run: its payload is
untrusted text, so it is only stored and shown (Inbox, and a title-only LINE line when the source allows it).
"""
import hashlib
import hmac
import json
import logging
from typing import Any

from psycopg.types.json import Jsonb

from . import config, db, inbox
from .channels import line
from .settings import get_settings

log = logging.getLogger(__name__)


def source_cfg(source: str) -> dict[str, Any] | None:
    return ((config.policy().get("events") or {}).get("sources") or {}).get(source)


def secret(source: str) -> str | None:
    return (get_settings().event_secrets or {}).get(source) or None


def valid_signature(key: str, body: bytes, header: str | None) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    want = hmac.new(key.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(want, header.removeprefix("sha256="))


def title_of(source: str, payload: Any, c: dict[str, Any]) -> str:
    """`title_field` (dotted path) from the payload, else "<source> event"."""
    v: Any = payload
    for k in str(c.get("title_field") or "title").split("."):
        v = v.get(k) if isinstance(v, dict) else None
    return f"{source}: {v}"[:200] if isinstance(v, (str, int, float)) and str(v).strip() else f"{source} event"


async def record(source: str, event_id: str, payload: Any, c: dict[str, Any]) -> bool:
    """False when this event id was already seen (provider retries)."""
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("INSERT INTO events (source, event_id, payload) VALUES (%s,%s,%s) "
                                 "ON CONFLICT (source, event_id) DO NOTHING", (source, event_id, Jsonb(payload)))
        if not cur.rowcount:
            return False
    private = bool(c.get("private", True))
    title = title_of(source, payload, c)
    await inbox.add("event", title, json.dumps(payload, ensure_ascii=False, indent=1)[:6000],
                    priority=c.get("priority", "normal"), private=private,
                    detail={"source": source, "event_id": event_id})
    s = get_settings()
    if c.get("line") and s.line_owner_user_id and s.line_channel_access_token:
        try:
            await line.push(s.line_owner_user_id, f"🔔 {source} event" if private else f"🔔 {title}")
        except Exception:
            log.exception("event LINE push failed")
    return True
