import pytest

from app.tools import browser, sources, web

SHELL = "<html><head><title>App</title></head><body><div id=root></div><script>boot()</script></body></html>"
FULL = "<html><head><title>Rendered</title></head><body><p>" + "Price is 999 USD. " * 60 + "</p></body></html>"


@pytest.fixture
def renders(monkeypatch):
    calls = []

    async def fake_render(url):
        calls.append(url)
        return {"final_url": url, "status": 200, "title": "Rendered", "html": FULL}

    monkeypatch.setattr(browser, "render", fake_render)
    return calls


def dl(monkeypatch, status=200, body=SHELL, ctype="text/html"):
    async def fake(url):
        return {"status": status, "final_url": url, "headers": {}, "content_type": ctype, "body": body}

    monkeypatch.setattr(sources, "_download", fake)


async def test_js_shell_is_rendered(redis, monkeypatch, renders):
    dl(monkeypatch)
    out = await sources.fetch_source("https://app.example/p", focus=["Price"], ledger_key="t")
    assert renders == ["https://app.example/p"]
    assert out["rendered"] and out["error"] is None and "999 USD" in out["excerpt"] and out["title"] == "Rendered"


async def test_403_is_rendered_and_full_pages_are_not(redis, monkeypatch, renders):
    dl(monkeypatch, status=403, body="")
    assert (await sources.fetch_source("https://a.example/x", focus=[], ledger_key="t"))["rendered"]
    dl(monkeypatch, body=FULL.replace("Rendered", "Plain"))
    out = await sources.fetch_source("https://b.example/y", focus=[], ledger_key="t")
    assert renders == ["https://a.example/x"] and not out["rendered"] and out["title"] == "Plain"


async def test_renderer_down_keeps_plain_result(redis, monkeypatch):
    async def down(url):
        return None

    monkeypatch.setattr(browser, "render", down)
    dl(monkeypatch, status=403, body="")
    out = await sources.fetch_source("https://a.example/x", focus=[], ledger_key="t")
    assert out["error"] == "HTTP 403" and not out["rendered"]


def test_disabled_never_renders(monkeypatch):
    monkeypatch.setattr(browser, "_cfg", lambda: {"enabled": False})
    assert not browser.should_render(403, "text/html", "")


async def test_fetch_page_tool_falls_back(monkeypatch, renders):
    class Resp:
        status_code, headers, encoding = 200, {"content-type": "text/html"}, "utf-8"

        async def aiter_bytes(self):
            yield SHELL.encode()

    class Stream:
        async def __aenter__(self):
            return Resp()

        async def __aexit__(self, *a):
            return False

    class Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a):
            return Stream()

    monkeypatch.setattr(web.httpx, "AsyncClient", Client)
    out = await web.fetch_page.ainvoke({"url": "https://app.example/p"})
    assert "999 USD" in out and 'title="Rendered"' in out


async def test_captcha_page_is_never_a_source(redis, monkeypatch):
    wall = "https://shopee.tw/verify/traffic?anti_bot_tracking_id=x"

    async def to_wall(url):
        return {"final_url": wall, "status": 200, "title": "Shopee", "html": FULL}

    monkeypatch.setattr(browser, "render", to_wall)
    dl(monkeypatch)
    out = await sources.fetch_source("https://shopee.tw/item", focus=[], ledger_key="t")
    assert not out["rendered"] and out["final_url"] == "https://shopee.tw/item"
    dl(monkeypatch, body="<html><title>Just a moment...</title><body>Verify you are human</body></html>")
    monkeypatch.setattr(browser, "_cfg", lambda: {"enabled": False})
    out = await sources.fetch_source("https://cf.example/p", focus=[], ledger_key="t")
    assert out["error"].startswith("blocked by bot protection") and out["excerpt"] == ""
