"""Phase 7b: Google Calendar tools. Writes are only Inbox proposals; Google is touched on approval."""
from datetime import datetime

import pytest
from fastapi import HTTPException

from app import gcal, inbox, todos
from app.routes import inbox as inbox_routes
from app.tools import calendar as cal, tools_for


class FakeInbox:
    def __init__(self):
        self.items: dict[int, dict] = {}

    async def add(self, kind, title, body="", **kw):
        i = len(self.items) + 1
        self.items[i] = {"id": i, "kind": kind, "title": title, "body": body, "status": "open", "run_id": None, **kw}
        return i

    async def list_(self, status="open", kind=None, limit=100):
        return [i for i in self.items.values() if i["status"] == status]

    async def get(self, i):
        return self.items.get(i)

    async def set_status(self, i, status):
        self.items[i]["status"] = status
        return True


class FakeGoogle:
    def __init__(self):
        self.calls: list[tuple] = []

    async def insert_event(self, title, start, end, notes=""):
        self.calls.append(("insert", title, start.isoformat(), end.isoformat()))
        return {"id": "e1"}

    async def delete_event(self, event_id):
        self.calls.append(("delete", event_id))

    async def get_event(self, event_id):
        if event_id == "nope":
            raise gcal.CalendarError("no such event")
        return {"id": event_id, "summary": "Dentist", "start": {"dateTime": "2026-11-02T09:00:00+08:00"},
                "end": {"dateTime": "2026-11-02T10:00:00+08:00"}}


@pytest.fixture
def env(monkeypatch):
    fi, g = FakeInbox(), FakeGoogle()
    for n in ("add", "list_", "get", "set_status"):
        monkeypatch.setattr(inbox, n, getattr(fi, n))
    for n in ("insert_event", "delete_event", "get_event"):
        monkeypatch.setattr(gcal, n, getattr(g, n))
    monkeypatch.setattr(cal.get_settings(), "line_owner_user_id", "")
    return fi, g


async def test_add_event_only_proposes_then_approval_creates_it(env):
    fi, g = env
    out = await cal.add_event.ainvoke({"title": "Dentist", "start": "2026-11-02T09:00"})
    assert "NOT been changed" in out and not g.calls
    item = fi.items[1]
    assert item["kind"] == "approval" and item["private"] and item["detail"]["calendar"]["end"].startswith("2026-11-02T10:00")
    assert "Already waiting" in await cal.add_event.ainvoke({"title": "Dentist", "start": "2026-11-02T09:00"})
    assert len(fi.items) == 1
    res = await inbox_routes.answer_item(1, inbox_routes.AnswerBody(value="Approve"), None, "owner")
    assert res["result"] == "added" and g.calls == [("insert", "Dentist", "2026-11-02T09:00:00+08:00", "2026-11-02T10:00:00+08:00")]
    assert item["status"] == "done"
    with pytest.raises(HTTPException):   # approving twice must not create a second event
        await inbox_routes.answer_item(1, inbox_routes.AnswerBody(value="Approve"), None, "owner")


async def test_delete_event_is_a_proposal_and_bad_input_is_refused(env):
    fi, g = env
    assert "Not proposed" in await cal.delete_event.ainvoke({"event_id": "nope"})
    assert "Not proposed" in await cal.add_event.ainvoke({"title": "x", "start": "2026-11-02"})   # needs a time
    assert "Not proposed" in await cal.add_event.ainvoke({"title": "x", "start": "2026-11-02T10:00", "end": "2026-11-02T09:00"})
    await cal.delete_event.ainvoke({"event_id": "e9"})
    assert not g.calls and fi.items[1]["title"] == "Delete calendar event: Dentist"
    await inbox_routes.answer_item(1, inbox_routes.AnswerBody(value="Approve"), None, "owner")
    assert g.calls == [("delete", "e9")]


def test_calendar_tools_are_local_only_and_need_a_connected_grant(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "policy", lambda: {"calendar": {"enabled": True}})
    monkeypatch.setattr(gcal, "available", lambda: True)
    assert "add_event" in {t.name for t in tools_for("assistant", local=True)}
    assert "add_event" not in {t.name for t in tools_for("assistant", local=False)}
    monkeypatch.setattr(gcal, "available", lambda: False)
    assert "add_event" not in {t.name for t in tools_for("assistant", local=True)}


def test_day_range_and_event_line():
    a, b = gcal.day_range("2026-11-02", 3)
    assert (a.isoformat(), (b - a).days) == ("2026-11-02T00:00:00+08:00", 3)
    assert (gcal.day_range("tomorrow", 1)[0].date() - datetime.now(todos.TZ).date()).days == 1
    ev = {"id": "e1", "summary": "A  B", "start": {"date": "2026-11-02"}, "end": {"date": "2026-11-03"}}
    assert gcal.line(ev) == "e1 | 2026-11-02 (all day) | A B"


def test_private_line_notice_says_what_really_happened_and_where_to_look():
    from langchain_core.messages import HumanMessage, ToolMessage

    from app.jobs.worker import _private_line_notice as n

    def tool(name, text):
        return ToolMessage(text, tool_call_id="1", name=name)

    base = "https://x.test"
    assert "Inbox" in n([HumanMessage("x"), tool("add_event", "Proposed (Inbox item #5). NOT")], base)
    assert "https://x.test/Inbox" in n([HumanMessage("x"), tool("add_event", "Proposed (Inbox item #5).")], base)
    assert "已新增待辦" in n([HumanMessage("x"), tool("add_todo", "Added todo #1: a")], base)
    assert "沒有新增" in n([HumanMessage("x"), tool("add_todo", "Not added: bad")], base)
    assert "沒有新增或修改" in n([HumanMessage("x")], base)   # the model only talked: say nothing was saved
    assert "已新增待辦" not in n([tool("add_todo", "Added todo #1: a"), HumanMessage("later")], base)   # earlier turn


def test_assistant_history_hides_earlier_tool_calls_but_keeps_the_current_turn():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from app.graphs.main import drop_old_tool_turns

    call = {"name": "add_todo", "args": {}, "id": "1"}
    msgs = [HumanMessage("加上 A"), AIMessage("", tool_calls=[call]), ToolMessage("Added todo", tool_call_id="1", name="add_todo"),
            AIMessage("好"), HumanMessage("加上 B"), AIMessage("", tool_calls=[call]), ToolMessage("Added", tool_call_id="1")]
    out = drop_old_tool_turns(msgs)
    assert [type(m).__name__ for m in out] == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage", "ToolMessage"]
    assert out[1].content == "好" and out[3].tool_calls


def test_intent_narrowing_picks_calendar_or_todo_without_a_model():
    from app.tools import narrow_for_intent
    from app.tools.todos import TODO_TOOLS

    tools = [*TODO_TOOLS, cal.add_event, cal.list_events]

    def names(text):
        return {t.name for t in narrow_for_intent(tools, text)}

    assert "add_todo" not in names("加上明天晚上要回桃園吃晚餐") and "add_event" in names("加上明天晚上要回桃園吃晚餐")
    assert "add_todo" not in names("/local 新增後天早上9點半要到士林捷運站")
    assert "add_event" not in names("新增待辦 明天早上9點交報告") and "add_todo" in names("提醒我晚上買牛奶")
    assert {"add_todo", "add_event"} <= names("新增 買牛奶")   # ambiguous: keep both
    assert narrow_for_intent(TODO_TOOLS, "加上明天晚上吃飯") == TODO_TOOLS   # no calendar connected


def test_day_after_tomorrow_is_understood():
    assert (todos.parse_due("day after tomorrow 09:30").date() - datetime.now(todos.TZ).date()).days == 2
    assert (todos.parse_due("後天 09:30").date() - datetime.now(todos.TZ).date()).days == 2


def test_api_times_are_taipei_not_utc():
    from datetime import timezone

    t = datetime(2026, 10, 31, 0, 30, tzinfo=timezone.utc)
    assert todos.out({"due_at": t, "created_at": t, "updated_at": t})["due_at"] == "2026-10-31T08:30:00+08:00"
    assert inbox.out({"created_at": t, "updated_at": t})["created_at"] == "2026-10-31T08:30:00+08:00"


async def test_a_model_that_claims_success_without_calling_the_tool_is_made_to_call_it():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    from app.graphs.main import build_graph
    from app.tools import required_tool

    assert required_tool([cal.add_event], "加上明天早上8點半要到士林捷運站") == "add_event"
    assert required_tool([cal.add_event], "新增待辦 明天早上8點半") is None and required_tool([cal.add_event], "今天天氣") is None

    from langchain_core.tools import tool

    @tool
    def add_event(title: str, start: str) -> str:
        """stub"""
        return "Proposed (Inbox item #1)."

    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    call = {"name": "add_event", "args": {"title": "x", "start": "tomorrow 08:30"}, "id": "c1"}
    lying = [AIMessage("已為您建立提案"), AIMessage("", tool_calls=[call]), AIMessage("done")]
    g = build_graph(Fake(messages=iter(lying)), [add_event], require_tool="add_event")
    names = [type(m).__name__ for m in (await g.ainvoke({"messages": [HumanMessage("加上明天早上8點半 x")]}))["messages"]]
    assert "ToolMessage" in names   # the retry produced a real tool call

    stubborn = [AIMessage("已為您建立提案"), AIMessage("還是沒有")]
    g = build_graph(Fake(messages=iter(stubborn)), [add_event], require_tool="add_event")
    last = (await g.ainvoke({"messages": [HumanMessage("加上明天早上8點半 x")]}))["messages"][-1]
    assert "沒有執行成功" in last.content


def test_month_day_due_dates_and_deadline_phrases_require_add_todo():
    from app.tools import required_tool
    from app.tools.todos import TODO_TOOLS

    now = datetime.now(todos.TZ)
    d = todos.parse_due("11/5")
    assert (d.month, d.day, d.hour, d.minute) == (11, 5, 23, 59) and d.date() >= now.date()
    assert todos.parse_due("11月5日 09:30").hour == 9
    assert todos.parse_due(f"{now.month}/{now.day}").date() == now.date()
    with pytest.raises(ValueError):
        todos.parse_due("13/45")
    for text in ("11/5前繳電費", "記得買牛奶", "新增待辦 洗車"):
        assert required_tool(TODO_TOOLS, text) == "add_todo"
    for text in ("我有什麼待辦?", "列出待辦", "今天天氣"):
        assert required_tool(TODO_TOOLS, text) is None
