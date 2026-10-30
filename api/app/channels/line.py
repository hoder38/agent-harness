"""LINE Messaging API: signature check, reply (free) and push (quota)."""
import base64
import hashlib
import hmac
import logging
import time

import httpx

from ..settings import get_settings

log = logging.getLogger(__name__)
MAX_TEXT = 5000


def verify_signature(body: bytes, signature: str | None, secret: str) -> bool:
    if not signature or not secret:
        return False
    mac = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(mac).decode(), signature)


def _messages(text: str) -> list[dict]:
    text = text.strip() or "(empty)"
    chunks = [text[i:i + MAX_TEXT] for i in range(0, len(text), MAX_TEXT)][:5]
    return [{"type": "text", "text": c} for c in chunks]


async def _post(path: str, payload: dict) -> None:
    s = get_settings()
    if not s.line_channel_access_token:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN not set")
    async with httpx.AsyncClient(timeout=10) as c:  # trust_env → goes through egress proxy
        r = await c.post(
            f"{s.line_api_base}{path}",
            headers={"Authorization": f"Bearer {s.line_channel_access_token}"},
            json=payload,
        )
        r.raise_for_status()


async def reply(reply_token: str, text: str) -> None:
    await _post("/v2/bot/message/reply", {"replyToken": reply_token, "messages": _messages(text)})


def thread_id(user_id: str) -> str:
    """The owner's LINE conversation. ":2" retired the first thread (it was locked PRIVATE with old private history)."""
    return f"line:{user_id}:2"


async def push(to: str, text: str) -> None:
    await _post("/v2/bot/message/push", {"to": to, "messages": _messages(text)})


async def deliver(to: str, text: str, reply_token: str | None, reply_ts: float | None, ttl_s: float) -> str:
    """Use the reply token while fresh; otherwise (or on failure) push. Returns 'reply' or 'push'."""
    if reply_token and reply_ts and time.time() - reply_ts < ttl_s:
        try:
            await reply(reply_token, text)
            return "reply"
        except Exception as e:  # expired / already used
            log.info("LINE reply failed (%s); falling back to push", e)
    await push(to, text)
    return "push"
