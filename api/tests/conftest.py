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
    from app import db, inbox

    calls: list[tuple[str, dict]] = []

    async def rec(_kind, **kw):
        calls.append((_kind, kw))

    monkeypatch.setattr(db, "log_decision", lambda **kw: rec("decision", **kw))
    monkeypatch.setattr(db, "update_outcome", lambda run_id, outcome, cost_usd=0.0: rec("outcome", run_id=run_id, outcome=outcome))
    monkeypatch.setattr(db, "log_usage", lambda *a: rec("usage", args=a))
    monkeypatch.setattr(db, "audit", lambda kind, name, detail, run_id=None: rec("audit", kind=kind, name=name))
    monkeypatch.setattr(db, "log_alert", lambda key, message, delivered: rec("alert", key=key))

    async def noop(*a, **kw):
        return None

    for name in ("add", "resolve_run", "notify_question", "memory_written"):
        monkeypatch.setattr(inbox, name, noop)
    from app import history
    monkeypatch.setattr(history, "load", noop)  # no rolling summary unless a test sets one
    return calls


@pytest.fixture(autouse=True)
def verify_scores(monkeypatch):
    """L9 model checks never reach Jev/Ollama in tests: every answer scores 1.0 unless a test queues scores."""
    from app import verify

    scores: list[float] = []

    async def score(request, answer, timeout_s):
        return scores.pop(0) if scores else 1.0

    monkeypatch.setattr(verify, "jev_score", score)
    monkeypatch.setattr(verify, "qwen_score", score)
    return scores


@pytest.fixture(autouse=True)
def all_features_on(monkeypatch):
    """The live policy may switch routes/shopping off for a while (routes.disabled, shopping.enabled); tests cover
    them all."""
    from app import config

    real = config.policy

    def pol():
        p = real()
        return {**p, "routes": {"disabled": []}, "shopping": {**(p.get("shopping") or {}), "enabled": True}}

    monkeypatch.setattr(config, "policy", pol)


@pytest.fixture
def policy():
    """Real policy, with RAG/memory off so graph tests never reach Postgres/Ollama (tests inject kb/recall), and
    grounding.always_search off so graphs without a stubbed `search` never reach the real web search."""
    from app import config

    pol = config.policy()
    return {**pol, "rag": {**(pol.get("rag") or {}), "enabled": False},
            "memory": {**(pol.get("memory") or {}), "enabled": False},
            "grounding": {**(pol.get("grounding") or {}), "always_search": False}}
