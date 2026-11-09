"""Telegram Bot API, send-only (§19): proactive notices go here instead of LINE push (no monthly quota)."""
import httpx

from ..settings import get_settings

MAX_TEXT = 4096


def configured() -> bool:
    s = get_settings()
    return bool(s.telegram_bot_token and s.telegram_owner_chat_id)


async def send(text: str) -> None:
    s = get_settings()
    if not configured():
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_OWNER_CHAT_ID not set")
    text = text.strip() or "(empty)"
    async with httpx.AsyncClient(timeout=10) as c:  # trust_env → goes through egress proxy
        for i in range(0, min(len(text), MAX_TEXT * 5), MAX_TEXT):
            r = await c.post(f"{s.telegram_api_base}/bot{s.telegram_bot_token}/sendMessage",
                             json={"chat_id": s.telegram_owner_chat_id, "text": text[i:i + MAX_TEXT],
                                   "disable_web_page_preview": True})
            r.raise_for_status()
