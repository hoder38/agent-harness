import hashlib
import hmac
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import brief, events, schedules
from app.routes import events as event_routes
from app.todos import TZ
from app.tools import answer_eval


def tp(*a):
    return datetime(*a, tzinfo=TZ)


# --- cron -------------------------------------------------------------------------------------------------------

def test_cron_matches_in_taipei_time():
    assert schedules.matches("0 7 * * *", tp(2026, 11, 4, 7, 0))
    assert schedules.matches("0 7 * * *", datetime(2026, 11, 3, 23, 0, tzinfo=timezone.utc))  # 07:00 Taipei
    assert not schedules.matches("0 7 * * *", tp(2026, 11, 4, 7, 1))


def test_cron_fields_steps_ranges_and_days():
    assert schedules.parse("5/15 * * * *")[0] == {5, 20, 35, 50}
    assert schedules.parse("*/20 9-11 * * *")[:2] == [{0, 20, 40}, {9, 10, 11}]
    # 2026-11-08 is a Sunday: dow 0 and 7 both mean Sunday
    assert schedules.matches("0 9 * * 7", tp(2026, 11, 8, 9, 0))
    assert schedules.matches("0 9 * * 0", tp(2026, 11, 8, 9, 0))
    assert not schedules.matches("0 9 * * 1-5", tp(2026, 11, 8, 9, 0))
    # both day fields restricted: either may match (standard cron)
    assert schedules.matches("0 9 1 * 0", tp(2026, 11, 8, 9, 0))


@pytest.mark.parametrize("bad", ["* * * *", "60 * * * *", "* 24 * * *", "x * * * *", "5-1 * * * *", "*/0 * * * *"])
def test_cron_rejects_bad_expressions(bad):
    with pytest.raises(ValueError):
        schedules.parse(bad)


# --- brief ------------------------------------------------------------------------------------------------------

def test_due_todos_splits_overdue_and_today():
    now = tp(2026, 11, 4, 7, 0)
    rows = [{"title": "a", "status": "open", "due_at": now - timedelta(hours=2)},
            {"title": "b", "status": "open", "due_at": now + timedelta(hours=3)},
            {"title": "c", "status": "open", "due_at": now + timedelta(days=2)},
            {"title": "d", "status": "open", "due_at": None},
            {"title": "e", "status": "done", "due_at": now - timedelta(hours=1)}]
    overdue, due = brief.due_todos(rows, tp(2026, 11, 5, 0, 0), now)
    assert [t["title"] for t in overdue] == ["a"] and [t["title"] for t in due] == ["b"]


def test_brief_line_notice_has_no_calendar_or_todo_text():
    now = tp(2026, 11, 4, 7, 0)
    ev = {"summary": "看牙醫", "location": "永和診所", "start": {"dateTime": "2026-11-04T10:00:00+08:00"},
          "end": {"dateTime": "2026-11-04T11:00:00+08:00"}}
    full, notice = brief.compose(day=now, weather_lines=["中和 晴 18–25°C 降雨 10%"], events=[ev],
                                 overdue=[{"title": "繳電費", "due_at": now - timedelta(days=1)}],
                                 due=[{"title": "回信給房東", "due_at": now + timedelta(hours=5)}],
                                 counts={"waiting": 1, "verify": 2}, spend=0.1234)
    for secret in ("看牙醫", "永和診所", "繳電費", "回信給房東"):
        assert secret in full and secret not in notice
    assert "10:00–11:00" in full and "11/04（三）" in full
    assert "中和 晴" in notice and "1 個行程" in notice and "1 個逾期" in notice and "US$0.12" in notice


def test_brief_without_calendar_or_spend():
    full, notice = brief.compose(day=tp(2026, 11, 4, 7), weather_lines=[], events=None, overdue=[], due=[],
                                 counts={}, spend=None)
    assert "行事曆未連線" in full and "行程" not in notice and "花費" not in full


# --- events -----------------------------------------------------------------------------------------------------

def test_event_signature_and_title():
    body = b'{"x": 1}'
    sig = "sha256=" + hmac.new(b"k", body, hashlib.sha256).hexdigest()
    assert events.valid_signature("k", body, sig)
    assert not events.valid_signature("k", body + b" ", sig)
    assert not events.valid_signature("k", body, sig.removeprefix("sha256="))
    assert not events.valid_signature("k", body, None)
    c = {"title_field": "repository.full_name"}
    assert events.title_of("github", {"repository": {"full_name": "a/b"}}, c) == "github: a/b"
    assert events.title_of("github", {"repository": "x"}, c) == "github event"
    assert events.title_of("hook", {"title": "  "}, {}) == "hook event"


@pytest.fixture
def event_client(monkeypatch):
    seen: list = []

    async def record(source, event_id, payload, c):
        dup = (source, event_id) in [(s, i) for s, i, _ in seen]
        seen.append((source, event_id, payload))
        return not dup

    monkeypatch.setattr(events, "source_cfg", lambda s: {"line": False} if s == "gh" else None)
    monkeypatch.setattr(events, "secret", lambda s: {"gh": "k", "nokey": ""}.get(s))
    monkeypatch.setattr(events, "record", record)
    app = FastAPI()
    app.include_router(event_routes.router)
    return TestClient(app), seen


def _sig(body: bytes) -> dict[str, str]:
    return {"X-Hub-Signature-256": "sha256=" + hmac.new(b"k", body, hashlib.sha256).hexdigest()}


def test_event_route(event_client):
    client, seen = event_client
    body = b'{"title": "deploy"}'
    assert client.post("/events/other", content=body, headers=_sig(body)).status_code == 404
    assert client.post("/events/gh", content=body, headers={"X-Hub-Signature-256": "sha256=00"}).status_code == 401
    r = client.post("/events/gh", content=body, headers={**_sig(body), "X-GitHub-Delivery": "d1"})
    assert r.json() == {"ok": True, "duplicate": False} and seen[0] == ("gh", "d1", {"title": "deploy"})
    again = client.post("/events/gh", content=body, headers={**_sig(body), "X-GitHub-Delivery": "d1"})
    assert again.json()["duplicate"]
    assert client.post("/events/gh", content=b"nope", headers=_sig(b"nope")).status_code == 400


# --- eval -------------------------------------------------------------------------------------------------------

def test_eval_pass_and_regressions():
    assert answer_eval.passes("台灣最高的山是玉山，海拔 3952 公尺", ["玉山"])
    assert answer_eval.passes("The symbol is AU.", ["Au"]) and not answer_eval.passes("金", ["Au"])
    c = {"max_answer_drop": 2, "max_recall_drop": 0.05}
    prev = {"passed": 11, "metrics": {"recall@5": 0.9}}
    assert answer_eval.regressions(None, 0, 0.0, c) == []
    assert answer_eval.regressions(prev, 10, 0.86, c) == []
    assert answer_eval.regressions(prev, 9, 0.8, c) == ["answers passed 11 → 9", "KB recall@5 0.90 → 0.80"]
    assert answer_eval.regressions({"passed": 11, "metrics": {}}, 11, 0.1, c) == []
