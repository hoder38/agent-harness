"""Product search for the shopping flow (§12c). Store adapters that return real, current listings:

- PChome 24h: public JSON search API (price filter `price=lo-hi`).
- momo, Yahoo購物中心: search pages carry a schema.org ItemList of Product/Offer (JSON-LD), so no HTML scraping.
Shopee blocks automated clients (CAPTCHA) and is not used.

Requests go through the open egress port (private IPs blocked). Results are cached in Redis for `cache_s`.
Listings are untrusted data: names/descriptions are only shown, never followed as instructions.
"""
import asyncio
import hashlib
import html as htmlmod
import json
import logging
import re
from typing import Any, TypedDict
from urllib.parse import quote, urlsplit

import httpx

from .. import config, state
from ..settings import get_settings
from .web import check_url

log = logging.getLogger(__name__)
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
STORE_NAMES = {"pchome": "PChome 24h", "momo": "momo購物網", "yahoo": "Yahoo購物中心"}
_LD = re.compile(r"<script[^>]*application/ld\+json[^>]*>(.*?)</script>", re.S | re.I)
_BRACKET_BRAND = re.compile(r"^\s*[【\[]([^】\]]{1,30})[】\]]")


class Product(TypedDict, total=False):
    id: str            # store:item id
    store: str
    name: str
    brand: str | None
    price: int         # TWD
    url: str
    rating: float | None
    reviews: int | None
    in_stock: bool
    desc: str


def _cfg() -> dict[str, Any]:
    return config.policy().get("shopping") or {}


def _clean(s: Any, n: int = 200) -> str:
    s = htmlmod.unescape(re.sub(r"<[^>]+>", " ", str(s or "")))
    return re.sub(r"\s+", " ", s).strip()[:n]


def _num(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _range(lo: int | None, hi: int | None) -> tuple[int, int]:
    return max(0, int(lo or 0)), int(hi) if hi else 9_999_999


# ---------------------------------------------------------------- adapters: (url builder, parser)
def pchome_url(q: str, lo: int | None, hi: int | None) -> str:
    u = f"https://ecshweb.pchome.com.tw/search/v4.3/all/results?q={quote(q)}&page=1&sort=sale/dc"
    return u + (f"&price={_range(lo, hi)[0]}-{_range(lo, hi)[1]}" if lo or hi else "")


def parse_pchome(body: str) -> list[Product]:
    try:
        prods = (json.loads(body) or {}).get("Prods") or []
    except ValueError:
        return []
    out: list[Product] = []
    for p in prods:
        pid, price = str(p.get("Id") or ""), _num(p.get("Price"))
        if not pid or price is None or p.get("isNC17"):
            continue
        name = _clean(p.get("Name"))
        m = _BRACKET_BRAND.match(name)
        out.append({"id": f"pchome:{pid}", "store": "pchome", "name": name,
                    "brand": _clean(p.get("Brand"), 40) or (m.group(1) if m else None), "price": int(price),
                    "url": f"https://24h.pchome.com.tw/prod/{pid}", "rating": _num(p.get("ratingValue")) or None,
                    "reviews": int(_num(p.get("reviewCount")) or 0) or None, "in_stock": True,
                    "desc": _clean(p.get("Describe"), 160)})
    return out


def momo_url(q: str, lo: int | None, hi: int | None) -> str:
    u = f"https://www.momoshop.com.tw/search/searchShop.jsp?keyword={quote(q)}"
    return u + (f"&_advPriceS={_range(lo, hi)[0]}&_advPriceE={_range(lo, hi)[1]}" if lo or hi else "")


def yahoo_url(q: str, lo: int | None, hi: int | None) -> str:
    u = f"https://tw.buy.yahoo.com/search/product?p={quote(q)}"
    return u + (f"&minp={_range(lo, hi)[0]}&maxp={_range(lo, hi)[1]}" if lo or hi else "")


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _canonical(store: str, url: str) -> tuple[str, str] | None:
    """(item id, clean product URL) without tracking parameters."""
    if store == "momo":
        m = re.search(r"i_code=(\d+)", url)
        return (m.group(1), f"https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code={m.group(1)}") if m else None
    if store == "yahoo":
        m = re.search(r"-(\d+)\.html", url)
        return (m.group(1), url.split("?")[0]) if m else None
    return None


def parse_ld_itemlist(body: str, store: str) -> list[Product]:
    out: list[Product] = []
    seen: set[str] = set()
    for block in _LD.findall(body):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        for node in _walk(data):
            if node.get("@type") != "Product":
                continue
            offer = node.get("offers") or {}
            offer = offer[0] if isinstance(offer, list) and offer else offer
            if not isinstance(offer, dict):
                continue
            price = _num(offer.get("price") or offer.get("lowPrice"))
            canon = _canonical(store, str(offer.get("url") or node.get("url") or ""))
            if price is None or not canon or canon[0] in seen:
                continue
            seen.add(canon[0])
            name = _clean(node.get("name"))
            brand = node.get("brand")
            brand = brand.get("name") if isinstance(brand, dict) else brand
            if brand in ("其他品牌", "其他", "無品牌", "No Brand"):
                brand = None
            if not brand and (m := _BRACKET_BRAND.match(name)):
                brand = m.group(1)
            rating = node.get("aggregateRating") or {}
            avail = str(offer.get("availability") or "InStock")
            out.append({"id": f"{store}:{canon[0]}", "store": store, "name": name,
                        "brand": _clean(brand, 40) or None, "price": int(price), "url": canon[1],
                        "rating": _num(rating.get("ratingValue")) or None,
                        "reviews": int(_num(rating.get("reviewCount")) or 0) or None,
                        "in_stock": "OutOfStock" not in avail and "SoldOut" not in avail,
                        "desc": _clean(node.get("description"), 160)})
    return out


ADAPTERS = {
    "pchome": (pchome_url, parse_pchome),
    "momo": (momo_url, lambda b: parse_ld_itemlist(b, "momo")),
    "yahoo": (yahoo_url, lambda b: parse_ld_itemlist(b, "yahoo")),
}


async def _get(url: str) -> str:
    async with httpx.AsyncClient(proxy=get_settings().fetch_proxy, trust_env=False, timeout=15,
                                 follow_redirects=True,
                                 headers={"User-Agent": UA, "Accept-Language": "zh-TW,zh;q=0.9"}) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
        return r.text


async def search_store(store: str, query: str, lo: int | None = None, hi: int | None = None) -> dict[str, Any]:
    """{store, query, products, error, cached}"""
    out: dict[str, Any] = {"store": store, "query": query, "products": [], "error": None, "cached": False}
    if store not in ADAPTERS:
        return {**out, "error": "unknown store"}
    build, parse = ADAPTERS[store]
    url = build(query, lo, hi)
    if err := check_url(url):
        return {**out, "error": err}
    key = "shop:" + hashlib.sha1(url.encode()).hexdigest()
    r = state.get_redis()
    if raw := await r.get(key):
        return {**out, "products": json.loads(raw), "cached": True}
    try:
        products = parse(await _get(url))
    except httpx.HTTPError as e:
        log.info("shop %s %r failed: %s", store, query, e)
        return {**out, "error": f"{type(e).__name__}: {str(e)[:80]}"}
    products = [p for p in products if urlsplit(p["url"]).hostname and not check_url(p["url"])]
    products = products[: int(_cfg().get("per_store", 20))]
    await r.set(key, json.dumps(products, ensure_ascii=False), ex=int(_cfg().get("cache_s", 3600)))
    return {**out, "products": products}


_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")
_WORD = re.compile(r"[a-z0-9]{2,}")


def _grams(text: str) -> set[str]:
    """CJK character bigrams + latin words: a cheap, tokenizer-free overlap measure for product titles."""
    t = (text or "").lower()
    out = {w for w in _WORD.findall(t)}
    for run in _CJK_RUN.findall(t):
        out |= {run[i:i + 2] for i in range(len(run) - 1)} if len(run) > 1 else {run}
    return out


def strong_terms(terms: list[str]) -> list[str]:
    """Drop bare modifiers ("電動", "品牌") that match half the store; keep all if nothing else is left."""
    strong = [t for t in terms if len(_grams(t)) >= 2]
    return strong or list(terms)


def relevance(name: str, terms: list[str]) -> float:
    """Best share of a search term's grams found in the listing name (0..1)."""
    low = (name or "").lower()
    best = 0.0
    for t in strong_terms(terms):
        g = _grams(t)
        if g:
            best = max(best, sum(1 for x in g if x in low) / len(g))
    return best


def filter_relevant(items: list[Product], terms: list[str], min_score: float = 0.5) -> list[Product]:
    """Keep listings that are actually the searched item (stores pad results with loosely related goods),
    most relevant first; order is otherwise preserved, so the store interleave survives within a score."""
    scored = [(relevance(p["name"], terms), i, p) for i, p in enumerate(items)]
    return [p for sc, _, p in sorted(scored, key=lambda x: (-round(x[0], 1), x[1])) if sc >= min_score]


async def search_products(queries: list[str], stores: list[str], lo: int | None, hi: int | None) -> dict[str, Any]:
    """All stores x queries in parallel; returns {products (deduped, interleaved by store), errors, calls}."""
    jobs = [search_store(st, q, lo, hi) for q in queries for st in stores]
    results = await asyncio.gather(*jobs)
    per_store: dict[str, list[Product]] = {}
    seen: set[str] = set()
    errors = []
    for res in results:
        if res["error"]:
            errors.append(f"{res['store']}: {res['error']}")
        for p in res["products"]:
            key = re.sub(r"\W+", "", p["name"].lower())[:40]
            if p["id"] in seen or key in seen:
                continue
            seen.update({p["id"], key})
            per_store.setdefault(p["store"], []).append(p)
    lo_, hi_ = _range(lo, hi)
    for st, items in per_store.items():
        per_store[st] = [p for p in items if p.get("in_stock", True) and lo_ <= p["price"] <= hi_]
    merged: list[Product] = []
    while any(per_store.values()):  # round-robin so one store does not crowd out the others
        for st in list(per_store):
            if per_store[st]:
                merged.append(per_store[st].pop(0))
    return {"products": merged, "errors": errors, "calls": len(jobs),
            "cached": all(r["cached"] for r in results) if results else False}
