"""Phase 7: todos (parse, tools, MCP) and the Inbox (approval notify, answer after Redis expiry, bulk rules)."""
from datetime import datetime, timezone
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
             "source_channel": channel, "created_at": datetime.now(timezone.utc)}
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
    assert out == "Added todo #1: Buy tea (due 2026-11-02 23:59)"
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


# ---------------------------------------------------------------- due reminders
from datetime import timedelta, timezone  # noqa: E402

from app.jobs import todo_jobs  # noqa: E402

NOW = datetime(2026, 11, 2, 9, 0, tzinfo=todos.TZ)   # a Monday


def todo(due_in_s, sent=0, status="open", id_=1):
    return {"id": id_, "title": "Pay rent (private)", "status": status, "reminders_sent": sent,
            "due_at": NOW + timedelta(seconds=due_in_s)}


def test_reminder_stages_fire_once_each_and_never_go_backwards():
    st = lambda t: todos.reminder_stage(t, NOW, 3600, 3600)   # noqa: E731
    assert st(todo(7200)) == 0 and st(todo(3000)) == 1 and st(todo(3000, sent=1)) == 0
    assert st(todo(-1800, sent=1)) == 0 and st(todo(-3600, sent=1)) == 2 and st(todo(-3600, sent=2)) == 0
    assert st(todo(-86400)) == 2   # created long overdue: one overdue reminder, no "due soon" first
    assert st(todo(3000, status="done")) == 0


def test_quiet_hours_are_weekday_nights_only():
    q = {"start_hour": 0, "end_hour": 7, "weekdays_only": True}
    at = lambda day, hour: datetime(2026, 11, day, hour, 30, tzinfo=todos.TZ).astimezone(timezone.utc)   # noqa: E731
    assert todos.in_quiet_hours(at(2, 3), q) and not todos.in_quiet_hours(at(2, 7), q)    # Mon 03:30 / 07:30
    assert not todos.in_quiet_hours(at(1, 3), q) and not todos.in_quiet_hours(at(7, 3), q)  # Sun, Sat nights are not quiet
    assert not todos.in_quiet_hours(at(2, 23), q)


async def test_reminder_cron_sends_generic_line_text_full_title_in_inbox_and_respects_quiet(monkeypatch, fake_inbox):
    sent, claimed = [], []

    async def cands(now, before_s):
        return [todo(3000, id_=1), todo(-4000, sent=1, id_=2)]

    async def claim(i, old, stage):
        claimed.append((i, stage))
        return True

    async def push(to, text):
        sent.append(text)

    monkeypatch.setattr(todos, "reminder_candidates", cands)
    monkeypatch.setattr(todos, "claim_reminder", claim)
    monkeypatch.setattr(todo_jobs.line, "push", push)
    monkeypatch.setattr(todo_jobs.get_settings(), "line_owner_user_id", "Uowner")
    monkeypatch.setattr(todo_jobs.get_settings(), "line_channel_access_token", "tok")
    monkeypatch.setattr(todo_jobs, "datetime", SimpleNamespace(now=lambda tz: NOW.astimezone(tz)))
    assert await todo_jobs.todo_reminders_cron({}) == 2
    assert claimed == [(1, 1), (2, 2)]
    assert sent == ["📌 2 todo reminders (1 overdue). Open the Inbox on the web to see them."]
    assert "Pay rent" not in sent[0] and fake_inbox.items[1]["title"] == "Todo due soon: Pay rent (private)"
    assert fake_inbox.items[2]["priority"] == "high"
    monkeypatch.setattr(todo_jobs, "datetime", SimpleNamespace(now=lambda tz: NOW.replace(hour=3).astimezone(tz)))
    sent.clear(); claimed.clear()
    assert await todo_jobs.todo_reminders_cron({}) == 0 and not sent and not claimed   # quiet: nothing, retried later


def test_empty_final_reply_after_a_tool_call_falls_back_to_the_tool_result():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from app.jobs.worker import _tool_fallback
    msgs = [HumanMessage("old"), ToolMessage("stale", tool_call_id="0"), HumanMessage("add a todo"),
            AIMessage("", tool_calls=[{"name": "add_todo", "args": {}, "id": "1"}]),
            ToolMessage("Added todo #6: 到士林捷運站", tool_call_id="1"), AIMessage("")]
    assert _tool_fallback(msgs) == "Added todo #6: 到士林捷運站"
    assert _tool_fallback([HumanMessage("hi"), AIMessage("")]) == ""   # nothing from an earlier turn leaks in


def test_parse_due_resolves_today_and_tomorrow_words_in_taipei():
    now = datetime.now(todos.TZ)
    d = todos.parse_due("tomorrow 07:30")
    assert (d.date() - now.date()).days == 1 and (d.hour, d.minute) == (7, 30)
    assert todos.parse_due("明天07:30") == d and todos.parse_due("today").hour == 23
    assert (todos.parse_due("後天 9:05").date() - now.date()).days == 2


async def test_repeated_identical_add_todo_call_does_not_create_a_duplicate(fake_todos):
    first = await todo_tools.add_todo.ainvoke({"title": "Buy milk", "due": "2026-11-02"})
    again = await todo_tools.add_todo.ainvoke({"title": "Buy  milk", "due": "2026-11-02"})
    assert first.startswith("Added todo #1") and again == "Already added: todo #1: Buy milk" and len(fake_todos.rows) == 1
    assert (await todo_tools.add_todo.ainvoke({"title": "Buy milk", "due": "2026-11-03"})).startswith("Added todo #2")
