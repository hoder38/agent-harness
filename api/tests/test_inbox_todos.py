"""Phase 7: todos (parse, tools, MCP) and the Inbox (approval notify, answer after Redis expiry, bulk rules)."""
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import inbox, mcp_server, state, todos
from app.routes import inbox as inbox_routes
from app.tools import todos as todo_tools, tools_for

REAL = {n: getattr(inbox, n) for n in ("notify_question", "memory_written", "resolve_run")}   # conftest stubs them


def test_parse_due_accepts_date_and_datetime_in_taipei():
    assert todos.parse_due("2026-11-02").isoformat() == "2026-11-02T23:59:00+08:00"
    assert todos.parse_due("2026-11-02T09:30").isoformat() == "2026-11-02T09:30:00+08:00"
    assert todos.parse_due("2026-11-02T01:00:00Z").utcoffset().total_seconds() == 0
    assert todos.parse_due("") is None and todos.parse_due(None) is None
    with pytest.raises(ValueError):
        todos.parse_due("next friday")


class FakeTodos:
    """In-memory stand-in for the todos SQL layer."""

    def __init__(self):
        self.rows: dict[int, dict] = {}

    async def add(self, title, *, notes="", due_at=None, channel="web"):
        t = {"id": len(self.rows) + 1, "title": title, "notes": notes, "status": "open", "due_at": due_at,
             "source_channel": channel}
        self.rows[t["id"]] = t
        return t

    async def list_(self, status="open", limit=200):
        return [t for t in self.rows.values() if status == "all" or t["status"] == status]

    async def update(self, todo_id, **kw):
        t = self.rows.get(todo_id)
        if t and kw.get("status"):
            t["status"] = kw["status"]
        return t


@pytest.fixture
def fake_todos(monkeypatch):
    f = FakeTodos()
    for n in ("add", "list_", "update"):
        monkeypatch.setattr(todos, n, getattr(f, n))
    return f


async def test_agent_and_mcp_tools_share_the_todo_table(fake_todos):
    out = await todo_tools.add_todo.ainvoke({"title": "Buy tea", "due": "2026-11-02"})
    assert out == "Added todo #1: Buy tea"
    assert fake_todos.rows[1]["source_channel"] == "agent" and fake_todos.rows[1]["due_at"].hour == 23
    assert "Not added" in await todo_tools.add_todo.ainvoke({"title": "x", "due": "someday"})
    assert "Added todo #2" in await mcp_server.add_todo("Review PR", "2026-11-03T10:00")
    assert fake_todos.rows[2]["source_channel"] == "copilot"
    assert (await mcp_server.list_todos()).splitlines()[0] == "[ ] #1 Buy tea (due 2026-11-02 23:59)"
    assert (await todo_tools.complete_todo.ainvoke({"todo_id": 1})).startswith("Done: #1")
    assert "#1" not in await mcp_server.list_todos() and "No todo #9" in await todo_tools.complete_todo.ainvoke({"todo_id": 9})


def test_only_the_assistant_route_gets_todo_tools_and_none_can_delete():
    names = {t.name for t in tools_for("assistant")}
    assert {"list_todos", "add_todo", "complete_todo"} <= names and not any("delete" in n for n in names)
    assert not {"list_todos", "add_todo"} & {t.name for t in tools_for("chat")}


# ---------------------------------------------------------------- inbox
class FakeInbox:
    def __init__(self):
        self.items: dict[int, dict] = {}
        self.pushed: list[str] = []

    async def add(self, kind, title, body="", **kw):
        if not inbox.cfg().get("enabled", True):
            return None
        i = len(self.items) + 1
        self.items[i] = {"id": i, "kind": kind, "title": title, "body": body, "status": "open", **kw}
        return i

    async def get(self, i):
        return self.items.get(i)

    async def set_status(self, i, status):
        self.items[i]["status"] = status
        return True


@pytest.fixture
def fake_inbox(monkeypatch, redis):
    f = FakeInbox()
    for n, fn in REAL.items():
        monkeypatch.setattr(inbox, n, fn)
    monkeypatch.setattr(inbox, "add", f.add)
    monkeypatch.setattr(inbox, "get", f.get)
    monkeypatch.setattr(inbox, "set_status", f.set_status)
    monkeypatch.setattr(inbox.get_settings(), "line_owner_user_id", "Uowner")
    monkeypatch.setattr(inbox.get_settings(), "line_channel_access_token", "tok")

    async def push(to, text):
        f.pushed.append(text)

    monkeypatch.setattr(inbox.line, "push", push)
    return f


async def ask(**kw):
    base = dict(run_id="r1", thread_id="web:t1", channel="web", q={"kind": "approval", "question": "Change ready",
                "options": ["Approve", "Reject"]}, round_=1, ttl_s=259200, private=False, coding=True,
                background=False, repo="harness")
    await inbox.notify_question(**{**base, **kw})


async def test_coding_approval_gets_item_and_line_push_but_plain_chat_clarify_does_not(fake_inbox):
    await ask(run_id="chat", coding=False)
    assert not fake_inbox.items   # a live chat shows its question inline
    await ask()
    item = fake_inbox.items[1]
    assert (item["kind"], item["priority"], item["title"]) == ("approval", "high", "Approve code change in harness")
    assert item["detail"]["pending_kind"] == "coding" and fake_inbox.pushed == ["📥 Approve code change in harness\nOpen the Inbox on the web to answer."]
    await ask(run_id="r2", channel="line")
    assert len(fake_inbox.pushed) == 1   # LINE already got the question itself
    await ask(run_id="r3", coding=False, background=True, q={"question": "Which account?"}, private=True)
    assert fake_inbox.items[3]["kind"] == "question" and fake_inbox.items[3]["private"]
    assert fake_inbox.pushed[-1] == "📥 A private task is waiting for you in the Inbox."   # no private text to LINE


async def test_answering_after_redis_pending_expired_restores_it_and_resumes(fake_inbox, redis, monkeypatch):
    await ask()
    sent = {}

    async def submit(request, owner, run_id, thread_id, pending, value):
        sent.update(run_id=run_id, thread_id=thread_id, pending=pending, value=value)
        return SimpleNamespace(run_id="r-resume")

    monkeypatch.setattr(inbox_routes, "submit_answer", submit)
    assert await state.get_pending(redis, "web:t1") is None   # expired days ago
    out = await inbox_routes.answer_item(1, inbox_routes.AnswerBody(value="Approve"), None, "owner")
    assert out == {"run_id": "r-resume", "thread_id": "t1"}
    assert sent["pending"]["run_id"] == "r1" and sent["pending"]["kind"] == "coding" and sent["value"] == "Approve"


async def test_answer_refuses_when_thread_moved_on_or_item_not_actionable(fake_inbox, redis):
    await ask()
    await state.set_pending(redis, "web:t1", run_id="other", round_=1, ttl_s=60)
    with pytest.raises(HTTPException) as e:
        await inbox_routes.answer_item(1, inbox_routes.AnswerBody(value="Approve"), None, "owner")
    assert e.value.status_code == 409
    await inbox.memory_written("add", "profile", "k1", "The owner likes tea", source="tool")
    note = max(fake_inbox.items)
    assert fake_inbox.items[note]["priority"] == "low"
    with pytest.raises(HTTPException) as e:
        await inbox_routes.answer_item(note, inbox_routes.AnswerBody(value="x"), None, "owner")
    assert e.value.status_code == 409


async def test_dismissing_an_approval_rejects_it_through_the_graph(fake_inbox, redis, monkeypatch):
    await ask()
    sent = {}

    async def submit(request, owner, run_id, thread_id, pending, value):
        sent["value"] = value
        return SimpleNamespace(run_id="r2")

    monkeypatch.setattr(inbox_routes, "submit_answer", submit)
    out = await inbox_routes.dismiss_item(1, None, "owner")
    assert out == {"ok": True, "rejected": True} and sent["value"] is None   # None = /skip = reject, sandbox discarded


async def test_dismissing_a_question_cancels_the_run_and_bulk_dismiss_never_touches_actionable(fake_inbox, redis):
    await ask(run_id="rq", coding=False, background=True, q={"question": "Which?"})
    await state.set_pending(redis, "web:t1", run_id="rq", round_=1, ttl_s=60)
    assert await inbox_routes.dismiss_item(1, None, "owner") == {"ok": True}
    assert fake_inbox.items[1]["status"] == "dismissed" and await state.get_pending(redis, "web:t1") is None
    assert await redis.get("cancel:rq") == "1"
    with pytest.raises(ValueError):
        await inbox.dismiss_bulk("approval")
    assert datetime  # silence unused import in minimal envs
