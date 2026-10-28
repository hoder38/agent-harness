"""Web tools: Brave search (strict egress allowlist) and page fetch (open egress port, private IPs blocked).

Never offered on PRIVATE threads (see tools_for), so private text cannot leak as a query or URL.
"""
import html
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

import httpx
from langchain_core.tools import tool

from ..settings import get_settings
from . import browser

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
MAX_PAGE_CHARS = 8000
MAX_DOWNLOAD_BYTES = 2_000_000
_TAG_RE = re.compile(r"<[^>]+>")
_SKIP = {"script", "style", "noscript", "svg", "head", "template", "iframe"}
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre", "table", "ul", "ol"}


def _clean(s: str | None) -> str:
    return html.unescape(_TAG_RE.sub("", s or "")).strip()


def format_results(data: dict, limit: int = 5) -> str:
    results = (data.get("web") or {}).get("results") or []
    if not results:
        return "No results."
    lines = []
    for i, r in enumerate(results[:limit], 1):
        lines.append(f"{i}. {_clean(r.get('title'))}\n   {r.get('url', '')}\n   {_clean(r.get('description'))}")
    return '<untrusted source="web_search">\n' + "\n".join(lines) + "\n</untrusted>"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in _SKIP:
            self._skip += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _SKIP and self._skip:
            self._skip -= 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(doc: str) -> tuple[str, str]:
    p = _TextExtractor()
    p.feed(doc)
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.parts))
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    return p.title.strip(), text


def check_url(url: str) -> str | None:
    """Return an error message for URLs the tool must not fetch (the proxy also blocks private IPs)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "Only http(s) URLs with a host are allowed."
    if parts.username or parts.password:
        return "URLs with credentials are not allowed."
    return None


@tool
async def web_search(query: str) -> str:
    """Search the web (Brave). Returns the top results as title, URL and snippet.
    Use for current events, recent facts, prices, or anything you are not sure about."""
    key = get_settings().brave_api_key
    if not key:
        return "Web search is not configured."
    async with httpx.AsyncClient(timeout=15) as client:  # trust_env → strict egress proxy (allowlisted)
        try:
            r = await client.get(
                BRAVE_URL,
                params={"q": query[:400], "count": 5},
                headers={"X-Subscription-Token": key, "Accept": "application/json"},
            )
        except httpx.HTTPError as e:
            return f"Search failed: {type(e).__name__}"
    if r.status_code == 429:
        return "Search rate limit reached; try again shortly or answer without searching."
    if r.status_code != 200:
        return f"Search failed: HTTP {r.status_code}"
    return format_results(r.json())


@tool
async def fetch_page(url: str) -> str:
    """Fetch a web page (http/https) and return its readable text, truncated to about 8000 characters.
    Use after web_search to read a result in detail."""
    if err := check_url(url):
        return err
    s = get_settings()
    status, ctype, body = None, "", ""
    try:
        async with httpx.AsyncClient(proxy=s.fetch_proxy, trust_env=False, timeout=20, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0 (agent-harness fetch_page)"}) as client:
            async with client.stream("GET", url) as r:
                status, ctype = r.status_code, r.headers.get("content-type", "").lower()
                if status == 200 and any(t in ctype for t in ("text/", "json", "xml")):
                    buf = bytearray()
                    async for chunk in r.aiter_bytes():
                        buf += chunk
                        if len(buf) >= MAX_DOWNLOAD_BYTES:
                            break
                    body = buf.decode(r.encoding or "utf-8", errors="replace")
    except httpx.ProxyError:
        return "Fetch blocked by the egress proxy (private or disallowed address)."
    except httpx.HTTPError as e:
        return f"Fetch failed: {type(e).__name__}"
    title, text = (html_to_text(body) if "html" in ctype else ("", body)) if body else ("", "")
    if browser.should_render(status, ctype, text) and (rd := await browser.render(url)):
        rtitle, rtext = html_to_text(rd["html"])
        if len(rtext) > len(text) and not check_url(rd["final_url"]) and not browser.is_bot_wall(rd["final_url"], rtitle, rtext):
            status, ctype, body, title, text = 200, "text/html", rd["html"], rtitle, rtext
    if status != 200:
        return f"Fetch failed: HTTP {status}"
    if not body:
        return f"Unsupported content type: {ctype or 'unknown'}"
    if browser.is_bot_wall(url, title, text):
        return "Fetch failed: blocked by bot protection (CAPTCHA); use another source."
    truncated = len(text) > MAX_PAGE_CHARS
    text = text[:MAX_PAGE_CHARS] + ("\n[truncated]" if truncated else "")
    return f'<untrusted source="{url}" title="{html.escape(title)}">\n{text}\n</untrusted>'


WEB_TOOLS = [web_search, fetch_page]
