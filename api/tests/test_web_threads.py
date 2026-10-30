import pytest
from fastapi import HTTPException

from app.routes.web import _public_tid, _thread_id, _thread_title


def test_line_alias_maps_to_owner_thread_and_back():
    assert _thread_id("line") == "line:Uowner:2"
    assert _public_tid("line:Uowner") == "line"
    assert _thread_id("wabc") == "web:wabc"
    assert _public_tid("web:wabc") == "wabc"
    with pytest.raises(HTTPException):
        _thread_id("line:Uother")


def test_thread_title_strips_override_and_whitespace():
    assert _thread_title("web:w1", "/local  Search the web\nfor news") == "Search the web for news"
    assert _thread_title("web:w1", None) == "(untitled)"
    assert _thread_title("line:Uowner", "hi") == "LINE"


def test_empty_reply_after_tool_shows_tool_result():
    from app.routes.web import _fill_empty_replies
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": ["add_todo"]},
            {"role": "tool", "name": "add_todo", "content": "Added todo #1"}, {"role": "assistant", "content": ""}]
    _fill_empty_replies(msgs)
    assert msgs[1]["content"] == "" and msgs[3]["content"] == "Added todo #1"
