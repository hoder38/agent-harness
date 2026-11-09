"""The single exit for proactive messages (§19): Telegram when configured, else LINE push."""
import logging

from ..settings import get_settings
from . import line, telegram

log = logging.getLogger(__name__)


def available() -> bool:
    s = get_settings()
    return telegram.configured() or bool(s.line_owner_user_id and s.line_channel_access_token)


async def notify(text: str) -> bool:
    """Send a proactive notice to the owner. Never raises; returns True if delivered."""
    s = get_settings()
    try:
        if telegram.configured():
            await telegram.send(text)
        elif s.line_owner_user_id and s.line_channel_access_token:
            await line.push(s.line_owner_user_id, text)
        else:
            return False
        return True
    except Exception:
        log.exception("notify failed")
        return False
