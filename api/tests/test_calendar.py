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
