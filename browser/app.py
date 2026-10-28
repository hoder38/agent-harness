"""Read-only page renderer (ARCHITECTURE.md §12b).

POST /render {url} -> {final_url, status, title, html}. Headless Chromium, one fresh incognito context per
request (no cookies, storage or logins survive), images/media/fonts skipped, downloads off. Every request,
loopback included, goes through the egress proxy's open port, which refuses private/LAN/metadata addresses.
The container sits on the internal `agent` network only, so there is no other way out.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from playwright.async_api import Error as PWError
from playwright.async_api import TimeoutError as PWTimeout
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field

PROXY = os.environ.get("BROWSER_PROXY", "http://egress-proxy:4751")
CONCURRENCY = int(os.environ.get("BROWSER_CONCURRENCY", "2"))
MAX_HTML = 3_000_000
SKIP_TYPES = {"image", "media", "font"}
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/140.0.0.0 Safari/537.36 agent-harness-render")

log = logging.getLogger("browser")
_pw = None
_browser = None
_launch_lock = asyncio.Lock()
_sem = asyncio.Semaphore(CONCURRENCY)


async def _get_browser():
    global _browser
    async with _launch_lock:
        if _browser is None or not _browser.is_connected():
            _browser = await _pw.chromium.launch(
                headless=True,
                proxy={"server": PROXY},
                # route loopback through the proxy too (Chromium bypasses it by default) -> squid denies it
                args=["--proxy-bypass-list=<-loopback>", "--disable-dev-shm-usage", "--no-first-run",
                      "--disable-background-networking", "--disable-sync", "--disable-extensions"],
            )
        return _browser


@asynccontextmanager
async def lifespan(_app):
    global _pw
    _pw = await async_playwright().start()
    await _get_browser()
    yield
    if _browser:
        await _browser.close()
    await _pw.stop()


app = FastAPI(lifespan=lifespan)


class RenderIn(BaseModel):
    url: str = Field(max_length=2048)
    timeout_s: float = Field(default=20, ge=1, le=45)
    settle_ms: int = Field(default=1500, ge=0, le=5000)


async def _route(route):
    req = route.request
    if req.resource_type in SKIP_TYPES or urlsplit(req.url).scheme not in ("http", "https", "data", "blob"):
        await route.abort()
    else:
        await route.continue_()


@app.post("/render")
async def render(body: RenderIn):
    if urlsplit(body.url).scheme not in ("http", "https"):
        raise HTTPException(400, "only http/https")
    async with _sem:
        browser = await _get_browser()
        ctx = await browser.new_context(user_agent=UA, accept_downloads=False, service_workers="block",
                                        java_script_enabled=True, locale="en-US", viewport={"width": 1280, "height": 2000})
        try:
            page = await ctx.new_page()
            await page.route("**/*", _route)
            resp = await page.goto(body.url, wait_until="domcontentloaded", timeout=body.timeout_s * 1000)
            if body.settle_ms:
                try:
                    await page.wait_for_load_state("networkidle", timeout=body.settle_ms)
                except PWTimeout:
                    pass
            html = await page.content()
            return {"final_url": page.url, "status": resp.status if resp else None,
                    "title": (await page.title())[:300], "html": html[:MAX_HTML]}
        except PWTimeout:
            raise HTTPException(504, "render timeout")
        except PWError as e:
            msg = str(e).splitlines()[0][:200]
            raise HTTPException(502, "blocked by the egress proxy" if "TUNNEL" in msg or "PROXY" in msg else msg)
        finally:
            await ctx.close()


@app.get("/healthz")
async def healthz():
    if not (_browser and _browser.is_connected()):
        raise HTTPException(503, "browser down")
    return {"ok": True}
