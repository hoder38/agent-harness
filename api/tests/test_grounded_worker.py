"""Grounded graph inside the worker + pipeline: streaming, clarify interrupt → pending → resume, /skip, LINE, GAMING."""
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app import pipeline, state
from app.jobs import worker
from app.models import Job, RouterResult
from app.tools import sources as src
from test_grounded import FACT, TaskLLM, Web, verify


@pytest.fixture
def enq():
    calls = []

    async def enqueue(fn, payload, job_id, queue):
        calls.append({"fn": fn, "payload": payload, "job_id": job_id, "queue": queue})

    enqueue.calls = calls
    return enqueue


@pytest.fixture
def confs(monkeypatch):
    """Router intent confidences, in order (default 0.95 = clear)."""
    seq: list[float] = []

    async def fake_route(**kw):
        return RouterResult(intent="chat", intent_confidence=seq.pop(0) if seq else 0.95, tier="local", source="jev")

    monkeypatch.setattr(pipeline, "route", fake_route)
    return seq


@pytest.fixture
def llm(monkeypatch):
    """Install a TaskLLM for the worker; returns a setter taking the script and recording model aliases."""
    box = {"models": []}

    def install(script, web=None):
        box["llm"] = TaskLLM(script=script, calls=[], seen=[])
        web = web or Web()
        monkeypatch.setattr(src, "search_web", web.search)
        monkeypatch.setattr(src, "fetch_source", web.fetch)

        def make(alias, ks):
            box["models"].append(alias)
            return box["llm"]

        monkeypatch.setattr(worker, "make_llm", make)
        return box

    return install


def job(text, key, **kw):
    return Job(channel=kw.pop("channel", "web"), mode=kw.pop("mode", "interactive"), user_id="owner",
               thread_id=kw.pop("thread_id", "web:t"), text=text, dedupe_key=key, **kw)


async def events(r, run_id):
    return [(f["type"], json.loads(f["data"])) for _, f in await r.xrange(f"run:{run_id}")]


CTX = lambda: {"checkpointer": InMemorySaver(), "redis": None, "job_try": 1}  # noqa: E731


@pytest.fixture(autouse=True)
def verify_mode(monkeypatch):
    pol = worker.config.policy()
    pol = {**pol, "grounding": {**pol["grounding"], "mode": "verify"}}
    monkeypatch.setattr(worker.config, "policy", lambda: pol)
    return pol


async def test_compose_tokens_stream_and_done_has_tail_and_grounding(redis, enq, confs, llm):
    web = Web(hits={"": ["https://openai.com/pricing"]}, pages={"https://openai.com/pricing": "GPT-9 costs $20 per month."})
    llm({"draft": "GPT-9 costs $20 per month.", "extract_claims": FACT,
         "plan_queries": {"queries": [{"query": "gpt-9 price", "claims": ["c1"]}]},
         "verify_support": verify(("c1", "supports", "GPT-9 costs $20 per month")),
         "compose": "GPT-9 costs $20 per month [1]."}, web)
    res = await pipeline.submit(job("price of gpt-9?", "a"), enq)
    assert (await worker.run_agent(CTX(), enq.calls[-1]["payload"]))["status"] == "done"
    evs = await events(redis, res.run_id)
    assert "".join(d for t, d in evs if t == "token") == "GPT-9 costs $20 per month [1]."  # only compose streams
    t, done = evs[-1]
    assert t == "done" and "Sources:\n[1]" in done["text"]
    assert done["grounding"]["verdict"] == "complete" and done["usage"]["output"] > 0


async def test_clarify_interrupt_then_answer_resumes_the_same_turn(redis, enq, confs, llm):
    box = llm({"assess_ambiguity": [{"ambiguous": True, "question_to_user": "Which phone?", "options": ["iPhone", "Pixel"]},
                                    {"ambiguous": False}],
               "draft": "Hello", "extract_claims": {"claims": []}})
    ctx = CTX()
    first = await pipeline.submit(job("how much is that one", "a"), enq)
    assert (await worker.run_agent(ctx, enq.calls[-1]["payload"]))["status"] == "awaiting_input"
    t, q = (await events(redis, first.run_id))[-1]
    assert t == "interrupt" and q["question"] == "Which phone?" and q["options"] == ["iPhone", "Pixel"]
    assert (await state.get_pending(redis, "web:t"))["run_id"] == first.run_id
    assert (await state.get_run_meta(redis, first.run_id))["status"] == "awaiting_input"
    assert (await state.get_clarify_ctx(redis, "web:t"))["question"] == "Which phone?"

    confs.append(0.3)  # a short answer looks ambiguous on its own: must not trigger a new clarify route
    second = await pipeline.submit(job("2", "b"), enq)
    payload = enq.calls[-1]["payload"]
    assert second.decision.route == "chat"
    assert payload["resume"] == {"value": "2", "parent_run_id": first.run_id}
    assert await state.get_pending(redis, "web:t") is None
    assert (await worker.run_agent(ctx, payload))["status"] == "done"
    assert box["llm"].calls.count("draft") == 1
    draft_in = next(m for t, m in box["llm"].seen if t == "draft")[-1].content
    assert draft_in.endswith("[Assistant asked] Which phone?\n[User answered] Pixel")  # option number mapped
    assert await state.get_clarify_ctx(redis, "web:t") is None
    snap = await worker.build_grounded_graph(None, policy={}, checkpointer=ctx["checkpointer"]).aget_state(
        {"configurable": {"thread_id": "web:t"}})
    assert [m.content for m in snap.values["messages"]] == ["how much is that one", "Which phone?\n1. iPhone\n2. Pixel",
                                                            "Pixel", "Hello"]


async def test_skip_answers_with_assumption_and_without_pending_is_a_noop(redis, enq, confs, llm):
    llm({"assess_ambiguity": {"ambiguous": True, "question_to_user": "Which?", "assumption_if_skipped": "the newest"},
         "draft": "Hi", "extract_claims": {"claims": []}})
    assert (await pipeline.submit(job("/skip", "z"), enq)).command_reply.startswith("Nothing to skip")
    ctx = CTX()
    await pipeline.submit(job("that", "a"), enq)
    await worker.run_agent(ctx, enq.calls[-1]["payload"])
    res = await pipeline.submit(job("/SKIP", "b"), enq)
    assert enq.calls[-1]["payload"]["resume"]["value"] is None
    assert (await worker.run_agent(ctx, enq.calls[-1]["payload"]))["status"] == "done"
    done = (await events(redis, res.run_id))[-1][1]
    assert "Assumptions:" in done["text"] and "the newest" in done["text"]


async def test_router_clarify_route_forces_a_question(redis, enq, confs, llm):
    llm({"assess_ambiguity": {"ambiguous": False, "question_to_user": "Which one do you mean?"}})
    confs.append(0.4)
    res = await pipeline.submit(job("that", "a"), enq)
    assert res.decision.route == "clarify"
    assert (await worker.run_agent(CTX(), enq.calls[-1]["payload"]))["status"] == "awaiting_input"


async def test_line_gets_the_question_with_options(redis, enq, confs, llm, monkeypatch):
    sent = []

    async def deliver(to, text, token, ts, ttl):
        sent.append((to, text, token))

    monkeypatch.setattr(worker.line, "deliver", deliver)
    llm({"assess_ambiguity": {"ambiguous": True, "question_to_user": "哪一支手機？", "options": ["iPhone", "Pixel"]}})
    await pipeline.submit(job("那個多少錢", "a", channel="line", thread_id="line:U"), enq,
                          delivery={"line_user_id": "U", "reply_token": "R", "reply_ts": 0})
    await worker.run_agent(CTX(), enq.calls[-1]["payload"])
    (to, text, token), = sent
    assert to == "U" and token == "R"
    assert text.startswith("哪一支手機？\n1. iPhone\n2. Pixel") and "/skip" in text


async def test_cancel_drops_the_pending_question_and_resume_endpoint_validates(redis, enq, confs, llm):
    from types import SimpleNamespace

    from fastapi import HTTPException

    from app.routes import web

    llm({"assess_ambiguity": {"ambiguous": True, "question_to_user": "Which?"}})
    res = await pipeline.submit(job("that", "a"), enq)
    await worker.run_agent(CTX(), enq.calls[-1]["payload"])
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(enqueue=enq)), headers={})
    ok = await web.resume_run(res.run_id, web.ResumeRequest(value="this one"), req, owner="owner")
    assert enq.calls[-1]["payload"]["resume"]["parent_run_id"] == res.run_id and ok["run_id"]

    await worker.run_agent(CTX(), enq.calls[0]["payload"])  # ask again on a fresh thread state
    pending = await state.get_pending(redis, "web:t")
    await web.cancel_run(pending["run_id"], req, owner="owner")
    assert await state.get_pending(redis, "web:t") is None
    with pytest.raises(HTTPException) as e:
        await web.resume_run(pending["run_id"], web.ResumeRequest(value="x"), req, owner="owner")
    assert e.value.status_code == 409


# ---------------------------------------------------------------- GAMING (§11)
async def test_queued_local_run_moves_to_cloud_when_gaming_started(redis, enq, confs, llm):
    box = llm({"draft": "Hi", "extract_claims": {"claims": []}})
    await pipeline.submit(job("hi", "a"), enq)
    await redis.set("gpu_state", "GAMING")
    await worker.run_agent(CTX(), enq.calls[-1]["payload"])
    assert box["models"] == ["cloud_fast"]


async def test_private_background_run_is_deferred_while_gaming(redis, enq, confs, llm):
    llm({})
    res = await pipeline.submit(job("my bank 1234", "a", mode="background", source="finance"), enq)
    await redis.set("gpu_state", "GAMING")
    assert (await worker.run_agent(CTX(), enq.calls[-1]["payload"]))["status"] == "deferred"
    assert res.run_id in await redis.hgetall("deferred")


async def test_long_local_run_is_requeued_on_cloud_and_continues_from_checkpoint(redis, enq, confs, llm, monkeypatch):
    box = llm({"draft": "Hi", "extract_claims": {"claims": []}})
    pol = {**worker.config.policy(), "gpu_governor": {"inflight_short_s": 0}}
    monkeypatch.setattr(worker.config, "policy", lambda: pol)
    requeued = []

    class FakeArq:
        async def enqueue_job(self, fn, payload, **kw):
            requeued.append(payload)

    states = iter(["FREE"])

    async def gpu(_r):
        return next(states, "GAMING")

    ctx = {"checkpointer": InMemorySaver(), "redis": FakeArq(), "job_try": 1}
    await pipeline.submit(job("hi", "a"), enq)
    monkeypatch.setattr(state, "get_gpu_state", gpu)  # FREE when the run starts, GAMING right after
    assert (await worker.run_agent(ctx, enq.calls[-1]["payload"]))["status"] == "requeued"
    (payload,) = requeued
    assert payload["model"] == "cloud_fast"
    assert (await worker.run_agent(ctx, payload))["status"] == "done"
    assert box["models"] == ["local", "cloud_fast"]
    assert box["llm"].calls.count("draft") == 1  # continued, not restarted
