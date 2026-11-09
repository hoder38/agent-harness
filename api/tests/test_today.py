import contextlib
import json
import time
from datetime import datetime, timedelta

from app import db, state, today, todos
from app.jobs.alerts import HEALTH_KEY

NOW = datetime(2026, 11, 9, 15, 0, tzinfo=todos.TZ)


def test_overdue_days_are_calendar_days_in_taipei():
    t = {"id": 1, "title": "x", "status": "open", "due_at": datetime(2026, 11, 7, 23, 59, tzinfo=todos.TZ)}
    assert todos.overdue_days(t, NOW) == 2
    assert todos.overdue_days({**t, "due_at": NOW - timedelta(hours=1)}, NOW) == 0
    assert todos.overdue_days({**t, "due_at": NOW + timedelta(hours=1)}, NOW) is None
    assert todos.overdue_days({**t, "status": "done"}, NOW) is None
    assert todos.overdue_days({**t, "due_at": None}, NOW) is None
    o = todos.out({**t, "created_at": NOW, "updated_at": NOW}, NOW)
    assert o["overdue"] and o["overdue_days"] == 2
    assert todos.overdue_label(2) == "逾期 2 天" and todos.overdue_label(0) == "今天已逾期"


def test_list_line_says_overdue():
    t = {"id": 3, "title": "pay rent", "status": "open", "due_at": datetime(2020, 1, 1, tzinfo=todos.TZ)}
    assert "— OVERDUE" in todos.line(t)
    assert "OVERDUE" not in todos.line({**t, "due_at": datetime(2999, 1, 1, tzinfo=todos.TZ)})


def _pool(monkeypatch, *, regressions=None, failed=(), alerts=0, verify=0):
    seen = {}

    class Cur:
        def __init__(self, one=None, all_=()):
            self.one, self.all = one, list(all_)

        async def fetchone(self):
            return self.one

        async def fetchall(self):
            return self.all

    class Conn:
        async def execute(self, sql, params=None):
            if "FROM eval_runs" in sql:
                return Cur({"metrics": {"regressions": regressions or []}})
            if "FROM schedules" in sql:
                return Cur(all_=[{"name": n, "last_status": "error boom"} for n in failed])
            seen["live"] = params[1]
            return Cur({"alerts": alerts, "verify": verify})

    class Pool:
        @contextlib.asynccontextmanager
        async def connection(self):
            yield Conn()

    monkeypatch.setattr(db, "get_pool", lambda: Pool())
    return seen


async def test_healthy_harness_has_no_cards(redis, monkeypatch):
    _pool(monkeypatch)
    await redis.set(HEALTH_KEY, json.dumps({"ts": time.time(), "issues": []}))
    assert await today.status_cards(NOW) == []


async def test_each_problem_is_one_card_most_severe_first(redis, monkeypatch):
    seen = _pool(monkeypatch, regressions=["answers 10 → 8"], failed=["brief"], alerts=1, verify=2)
    await redis.set(HEALTH_KEY, json.dumps({"ts": time.time(), "issues": [
        {"key": "budget:80", "message": "daily spend at 85% of budget"},
        {"key": "container:ollama:down", "message": "ollama is exited"}]}))
    await state.set_killswitch(redis, "cloud")
    await redis.set("gpu_state_override", "GAMING")
    cards = await today.status_cards(NOW)
    keys = [c["key"] for c in cards]
    assert keys[:2] == ["killswitch", "container:ollama:down"]
    assert set(keys) == {"killswitch", "container:ollama:down", "gpu:override", "budget:80", "eval:regression",
                         "schedule:brief", "inbox:alert", "inbox:verify"}
    assert sorted(seen["live"]) == ["budget:80", "container:ollama:down"]  # live issues are not counted twice


async def test_stale_watcher_is_a_card(redis, monkeypatch):
    _pool(monkeypatch)
    await redis.set(HEALTH_KEY, json.dumps({"ts": time.time() - 1000, "issues": []}))
    assert [c["key"] for c in await today.status_cards(NOW)] == ["watcher:stale"]
