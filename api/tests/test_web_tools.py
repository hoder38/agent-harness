from app.settings import get_settings
from app.tools import tools_for
from app.tools.web import check_url, format_results, html_to_text


def _names(tools):
    return [t.name for t in tools]


def test_web_tools_never_offered_on_private_threads(monkeypatch):
    monkeypatch.setattr(get_settings(), "brave_api_key", "k")
    assert {"web_search", "fetch_page"} <= set(_names(tools_for("chat")))
    assert not {"web_search", "fetch_page"} & set(_names(tools_for("chat", private=True)))
    assert not {"web_search", "fetch_page"} & set(_names(tools_for("clarify")))


def test_web_search_omitted_without_key(monkeypatch):
    monkeypatch.setattr(get_settings(), "brave_api_key", "")
    names = _names(tools_for("assistant"))
    assert "web_search" not in names and "fetch_page" in names


def test_format_results_strips_markup_and_marks_untrusted():
    out = format_results({"web": {"results": [
        {"title": "A &amp; B", "url": "https://a.example", "description": "<strong>hot</strong> news"},
    ]}})
    assert out.startswith('<untrusted source="web_search">')
    assert "1. A & B\n   https://a.example\n   hot news" in out
    assert format_results({}) == "No results."


def test_html_to_text_drops_scripts_and_keeps_blocks():
    title, text = html_to_text(
        "<html><head><title>T</title><style>x{}</style></head>"
        "<body><script>evil()</script><h1>Head</h1><p>one  two</p><p>three</p></body></html>"
    )
    assert title == "T"
    assert text == "Head\none two\nthree"


def test_check_url_rejects_non_http_and_credentials():
    assert check_url("https://example.com/x") is None
    assert check_url("file:///etc/passwd")
    assert check_url("http://user:pw@example.com/")
    assert check_url("https:///nohost")
