"""Structured web tools for the grounded graph (§12a). Called by graph nodes, never by LLM tool-calling,
so per-turn budgets are enforceable. The string tools in tools/web.py remain for the ReAct graph.

- Idempotency ledger: Redis `ledger:{ledger_key}:{sha1(args)}` (1 d). A retried node gets the same result
  without calling Brave or the page again.
- Fetch cache: `fetch:{sha1(url)}` (24 h, keyed by requested and final URL). Cache hits are free.
"""
import hashlib
import json
import re
import unicodedata
from typing import Any, Literal

import httpx

from .. import config, state
from ..gates.gate0 import gate0
from ..models import Job
from ..settings import get_settings
from . import browser
from .authority import domain_of
from .web import BRAVE_URL, MAX_DOWNLOAD_BYTES, _clean, check_url, html_to_text

LEDGER_TTL_S = 86400
FETCH_CACHE_TTL_S = 86400
FOCUS_RADIUS = 600
Freshness = Literal["pd", "pw", "pm", "py"]


def _sha1(v: Any) -> str:
    return hashlib.sha1(json.dumps(v, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


async def _ledger_get(key: str) -> dict | None:
    raw = await state.get_redis().get(key)
    return json.loads(raw) if raw else None


async def _ledger_put(key: str, value: dict, ttl_s: int = LEDGER_TTL_S) -> None:
    await state.get_redis().set(key, json.dumps(value, ensure_ascii=False), ex=ttl_s)


def query_blocked(query: str) -> str | None:
    """Gate 0 on outgoing queries: secrets/PII or private keywords never leave the machine."""
    g = gate0(Job(channel="event", user_id="-", thread_id="-", text=query, dedupe_key="-"), config.privacy_rules())
    if g.findings or g.privacy_tag == "PRIVATE":
        return "query contains private or secret data"
    return None


# ---------------------------------------------------------------- search
async def search_web(query: str, *, count: int = 5, freshness: Freshness | None = None,
                     ledger_key: str) -> dict[str, Any]:
    """{query, hits[{url,title,snippet,domain,age}], error, cached, ledger}"""
    args = {"q": query[:400], "count": count, "freshness": freshness}
    lkey = f"ledger:{ledger_key}:search:{_sha1(args)}"
    if hit := await _ledger_get(lkey):
        return {**hit, "ledger": True}
    out: dict[str, Any] = {"query": query, "hits": [], "error": None, "cached": False, "ledger": False}
    if err := query_blocked(query):
        out["error"] = err
        return out  # nothing sent, nothing charged
    key = get_settings().brave_api_key
    if not key:
        out["error"] = "web search is not configured"
        return out
    params = {k: v for k, v in args.items() if v is not None}
    try:
        async with httpx.AsyncClient(timeout=15) as client:  # trust_env → strict egress proxy (allowlisted)
            r = await client.get(BRAVE_URL, params=params,
                                 headers={"X-Subscription-Token": key, "Accept": "application/json"})
        if r.status_code == 429:
            out["error"] = "search rate limit reached"
        elif r.status_code != 200:
            out["error"] = f"search failed: HTTP {r.status_code}"
        else:
            for item in ((r.json().get("web") or {}).get("results") or [])[:count]:
                url = item.get("url") or ""
                if check_url(url):
                    continue
                out["hits"].append({"url": url, "title": _clean(item.get("title")),
                                    "snippet": _clean(item.get("description")), "domain": domain_of(url),
                                    "age": item.get("page_age") or item.get("age")})
    except httpx.HTTPError as e:
        out["error"] = f"search failed: {type(e).__name__}"
    await _ledger_put(lkey, out)
    return out


# ---------------------------------------------------------------- fetch
_META_PUB = re.compile(r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|date|pubdate|dc\.date)["\'][^>]*'
                       r'content=["\']([^"\']+)', re.I)
_META_PUB2 = re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\']'
                        r'(?:article:published_time|date|pubdate|dc\.date)["\']', re.I)
_TIME_TAG = re.compile(r'<time[^>]+datetime=["\']([^"\']+)', re.I)


def published_at(body: str, headers: dict[str, str]) -> str | None:
    for rx in (_META_PUB, _META_PUB2, _TIME_TAG):
        if m := rx.search(body[:200_000]):
            return m.group(1).strip()
    return headers.get("last-modified")


def normalize(s: str) -> str:
    """NFKC (full-width → half-width), unified quotes/dashes, collapsed whitespace, casefolded."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "\u00a0": " "}))
    return re.sub(r"\s+", " ", s).strip().casefold()


def focus_windows(text: str, focus: list[str], max_chars: int, radius: int = FOCUS_RADIUS) -> tuple[str, bool]:
    """Windows of ±radius chars around focus-term matches, merged, up to max_chars. Falls back to the head."""
    low = text.casefold()
    spans: list[list[int]] = []
    for term in {t.strip().casefold() for t in focus if t and len(t.strip()) >= 2}:
        start = 0
        while (i := low.find(term, start)) != -1 and len(spans) < 200:
            spans.append([max(0, i - radius), min(len(text), i + len(term) + radius)])
            start = i + len(term)
    if not spans:
        return text[:max_chars], len(text) > max_chars
    spans.sort()
    merged = [spans[0]]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    parts, total = [], 0
    for a, b in merged:
        piece = text[a:b]
        if total + len(piece) > max_chars:
            piece = piece[: max_chars - total]
        parts.append(piece)
        total += len(piece)
        if total >= max_chars:
            break
    excerpt = "\n…\n".join(parts)
    return excerpt, total < len(text)


async def _download(url: str) -> dict[str, Any]:
    s = get_settings()
    async with httpx.AsyncClient(proxy=s.fetch_proxy, trust_env=False, timeout=20, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 (agent-harness fetch_source)"}) as client:
        async with client.stream("GET", url) as r:
            res = {"status": r.status_code, "final_url": str(r.url), "headers": {k.lower(): v for k, v in r.headers.items()},
                   "content_type": r.headers.get("content-type", "").lower(), "body": ""}
            if r.status_code != 200 or not any(t in res["content_type"] for t in ("text/", "json", "xml")):
                return res
            buf = bytearray()
            async for chunk in r.aiter_bytes():
                buf += chunk
                if len(buf) >= MAX_DOWNLOAD_BYTES:
                    break
            res["body"] = buf.decode(r.encoding or "utf-8", errors="replace")
            return res


async def fetch_source(url: str, *, focus: list[str], max_chars: int = 8000, ledger_key: str) -> dict[str, Any]:
    """{url, final_url, status, content_type, title, published_at, excerpt, text_sha256, truncated, error, cached,
    rendered}. JS-only pages and listed statuses fall back to headless Chromium (tools/browser.py)."""
    out: dict[str, Any] = {"url": url, "final_url": url, "status": None, "content_type": "", "title": "",
                           "published_at": None, "excerpt": "", "text_sha256": "", "truncated": False,
                           "error": None, "cached": False, "ledger": False, "rendered": False}
    if err := check_url(url):
        return {**out, "error": err}
    lkey = f"ledger:{ledger_key}:fetch:{_sha1([url, sorted(set(focus)), max_chars])}"
    if hit := await _ledger_get(lkey):
        return {**hit, "ledger": True}
    r = state.get_redis()
    page: dict | None = None
    if raw := await r.get(f"fetch:{_sha1(url)}"):
        page, out["cached"] = json.loads(raw), True
    else:
        try:
            dl = await _download(url)
        except httpx.ProxyError:
            out["error"] = "blocked by the egress proxy"
        except httpx.HTTPError as e:
            out["error"] = f"fetch failed: {type(e).__name__}"
        else:
            title, text = "", ""
            if dl["status"] == 200 and dl["body"]:
                title, text = html_to_text(dl["body"]) if "html" in dl["content_type"] else ("", dl["body"])
            rendered = False
            if browser.should_render(dl["status"], dl["content_type"], text) and (rd := await browser.render(url)):
                rtitle, rtext = html_to_text(rd["html"])
                if len(rtext) > len(text) and not browser.is_bot_wall(rd["final_url"], rtitle, rtext):
                    dl = {**dl, "status": 200, "final_url": rd["final_url"], "content_type": "text/html",
                          "body": rd["html"], "headers": {}}
                    title, text, rendered = rtitle or rd.get("title", ""), rtext, True
            out.update(status=dl["status"], final_url=dl["final_url"], content_type=dl["content_type"])
            if dl["status"] != 200:
                out["error"] = f"HTTP {dl['status']}"
            elif "pdf" in dl["content_type"]:
                out["error"] = "pdf"
            elif not dl["body"]:
                out["error"] = f"unsupported content type: {dl['content_type'] or 'unknown'}"
            else:
                page = {"final_url": dl["final_url"], "status": 200, "content_type": dl["content_type"],
                        "title": title, "text": text, "published_at": published_at(dl["body"], dl["headers"]),
                        "rendered": rendered}
                blob = json.dumps(page, ensure_ascii=False)
                for u in {url, dl["final_url"]}:
                    await r.set(f"fetch:{_sha1(u)}", blob, ex=FETCH_CACHE_TTL_S)
    if page:
        excerpt, truncated = focus_windows(page["text"], focus, max_chars)
        out.update(final_url=page["final_url"], status=page["status"], content_type=page["content_type"],
                   title=page["title"][:300], published_at=page["published_at"], excerpt=excerpt,
                   truncated=truncated, text_sha256=hashlib.sha256(page["text"].encode()).hexdigest(),
                   rendered=page.get("rendered", False))
        if err := check_url(out["final_url"]):
            out.update(error=err, excerpt="")
        elif browser.is_bot_wall(out["final_url"], out["title"], page["text"]):
            out.update(error="blocked by bot protection (CAPTCHA)", excerpt="")
    await _ledger_put(lkey, out)
    return out


# ---------------------------------------------------------------- local knowledge base (PRIVATE threads, §10a)
async def kb_search(query: str, *, count: int = 5, ledger_key: str) -> dict[str, Any]:
    return {"query": query, "hits": [], "error": "local KB not available", "cached": False, "ledger": False}


async def kb_read(path: str, *, focus: list[str], max_chars: int = 8000, ledger_key: str) -> dict[str, Any]:
    return {"url": path, "final_url": path, "excerpt": "", "error": "local KB not available", "cached": False}
