import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("POLICY_PATH", str(ROOT / "config" / "policy.yaml"))
os.environ.setdefault("PRIVACY_RULES_PATH", str(ROOT / "config" / "privacy_rules.yaml"))
os.environ.setdefault("AUTHORITY_PATH", str(ROOT / "config" / "authority.yaml"))
os.environ.setdefault("AGENT_JWT_SECRET", "test-secret")
os.environ.setdefault("OWNER_ID", "owner")
os.environ.setdefault("LINE_CHANNEL_SECRET", "line-secret")
os.environ.setdefault("LINE_OWNER_USER_ID", "Uowner")


@pytest.fixture
async def redis():
    import fakeredis

    from app import state

    r = fakeredis.FakeAsyncRedis(decode_responses=True)
    state.set_redis(r)
    yield r
    await r.aclose()


@pytest.fixture(autouse=True)
def no_db(monkeypatch):
    """Postgres writes are fire-and-forget logging; record them instead."""
    from app import db

    calls: list[tuple[str, dict]] = []

    async def rec(_kind, **kw):
        calls.append((_kind, kw))

    monkeypatch.setattr(db, "log_decision", lambda **kw: rec("decision", **kw))
    monkeypatch.setattr(db, "update_outcome", lambda run_id, outcome, cost_usd=0.0: rec("outcome", run_id=run_id, outcome=outcome))
    monkeypatch.setattr(db, "log_usage", lambda *a: rec("usage", args=a))
    monkeypatch.setattr(db, "audit", lambda kind, name, detail, run_id=None: rec("audit", kind=kind, name=name))
    monkeypatch.setattr(db, "log_alert", lambda key, message, delivered: rec("alert", key=key))
    return calls


@pytest.fixture
def policy():
    """Real policy, with RAG/memory off so graph tests never reach Postgres/Ollama (tests inject kb/recall)."""
    from app import config

    pol = config.policy()
    return {**pol, "rag": {**(pol.get("rag") or {}), "enabled": False},
            "memory": {**(pol.get("memory") or {}), "enabled": False}}
