"""Read-only rendering fallback (§12b). When a plain fetch returns an HTML shell with little text (JS apps) or a
status listed in policy `browser.render_on_status`, ask the `browser` service (headless Chromium) to render it.
No clicks, forms or logins; those are phase 8b (§12b). Failures return None, so callers keep the plain result.
"""
import logging
import re
from typing import Any

import httpx

from .. import config
from ..settings import get_settings

log = logging.getLogger(__name__)


def _cfg() -> dict[str, Any]:
    return config.policy().get("browser") or {}


_WALL_URL = re.compile(r"/(verify/traffic|captcha|challenge|cdn-cgi/challenge|sorry/index|blocked)|anti_bot|"
                       r"[?&](captcha|challenge)=", re.I)
_WALL_TEXT = re.compile(r"captcha|verify (that )?you are (a )?human|are you a robot|unusual traffic|just a moment\.\.\.|"
                        r"access denied|attention required|enable javascript and cookies|驗證碼|人機驗證|機器人|"
                        r"異常流量|请完成安全验证|安全驗證", re.I)


def is_bot_wall(final_url: str, title: str, text: str) -> bool:
    """Anti-bot / CAPTCHA interstitial (e.g. Shopee /verify/traffic, Cloudflare). Never usable as a source."""
    if _WALL_URL.search(final_url or ""):
        return True
    return len(text.strip()) < 1500 and bool(_WALL_TEXT.search(f"{title}\n{text}"))


def should_render(status: int | None, content_type: str, text: str) -> bool:
    c = _cfg()
    if not c.get("enabled"):
        return False
    if status in (c.get("render_on_status") or []):
        return True
    return status == 200 and "html" in content_type and len(text.strip()) < int(c.get("min_text_chars", 500))


async def render(url: str) -> dict[str, Any] | None:
    """{final_url, status, title, html} from headless Chromium, or None if the renderer is off/down/failed."""
    c = _cfg()
    timeout = float(c.get("timeout_s", 20))
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=timeout + 5) as client:
            r = await client.post(f"{get_settings().browser_url}/render",
                                  json={"url": url, "timeout_s": timeout, "settle_ms": int(c.get("settle_ms", 1500))})
    except httpx.HTTPError as e:
        log.warning("render %s failed: %s", url, type(e).__name__)
        return None
    if r.status_code != 200:
        log.info("render %s: HTTP %s %s", url, r.status_code, r.text[:200])
        return None
    out = r.json()
    return out if out.get("status") == 200 and out.get("html") else None
