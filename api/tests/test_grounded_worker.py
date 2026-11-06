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
    await state.mark_thread_private(redis, "web:t")
    res = await pipeline.submit(job("my bank 1234", "a", mode="background"), enq)
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


async def test_task_type_question_then_answer_streams_the_understanding_line_first(redis, enq, llm, monkeypatch):
    async def fake_route(**kw):
        return RouterResult(intent="chat", intent_confidence=0.95, tier="local", source="jev", task_type="shopping",
                            task_probs={"shopping": 0.5, "advice": 0.45})

    monkeypatch.setattr(pipeline, "route", fake_route)
    web = Web(hits={"": ["https://openai.com/pricing"]}, pages={"https://openai.com/pricing": "GPT-9 costs $20 per month."})
    llm({"draft": "GPT-9 costs $20 per month.", "extract_claims": FACT,
         "plan_queries": {"queries": [{"query": "gpt-9 price", "claims": ["c1"]}]},
         "verify_support": verify(("c1", "supports", "GPT-9 costs $20 per month")),
         "compose": "GPT-9 costs $20 per month [1]."}, web)
    ctx = CTX()
    first = await pipeline.submit(job("gpt-9 plans", "a"), enq)
    assert first.decision.confirm_task == ["shopping", "advice"]
    assert (await worker.run_agent(ctx, enq.calls[-1]["payload"]))["status"] == "awaiting_input"
    t, q = (await events(redis, first.run_id))[-1]
    assert t == "interrupt" and q["options"] == ["Find a product to buy", "Knowledge or advice"]
    second = await pipeline.submit(job("2", "b"), enq)
    assert (await worker.run_agent(ctx, enq.calls[-1]["payload"]))["status"] == "done"
    evs = await events(redis, second.run_id)
    echo = 'My understanding: Knowledge or advice — "gpt-9 plans"'
    assert "".join(d for t, d in evs if t == "token") == f"{echo}\n\nGPT-9 costs $20 per month [1]."
    assert evs[-1][1]["text"].startswith(f"{echo}\n\nGPT-9 costs $20 per month [1].\n\nSources:")


# ---------------------------------------------------------------- L9 verify + escalation (§12, 10b)
class FakeArq:
    def __init__(self):
        self.jobs = []

    async def enqueue_job(self, fn, payload, **kw):
        if fn == "run_agent":  # not the memory jobs
            self.jobs.append(payload)

    def __getattr__(self, name):  # memory jobs enqueue through other calls
        async def noop(*a, **kw):
            return None
        return noop


async def _messages(ctx):
    snap = await worker.build_grounded_graph(None, policy={}, checkpointer=ctx["checkpointer"]).aget_state(
        {"configurable": {"thread_id": "web:t"}})
    return [m.content for m in snap.values["messages"]]


async def test_failed_answer_is_rerun_on_cloud_from_the_turn_start(redis, enq, confs, llm, verify_scores):
    box = llm({"draft": ["Hello!", "I'll look it up.", "Paris."], "extract_claims": {"claims": []}})
    ctx = {**CTX(), "redis": FakeArq()}
    await pipeline.submit(job("hi", "0"), enq)
    await worker.run_agent(ctx, enq.calls[-1]["payload"])  # an earlier turn stays in the history
    res = await pipeline.submit(job("capital of France?", "a"), enq)
    verify_scores.extend([0.05, 0.9])
    assert (await worker.run_agent(ctx, enq.calls[-1]["payload"]))["status"] == "requeued"
    (payload,) = ctx["redis"].jobs
    assert payload["model"] == "cloud_fast" and payload["escalate"]["history"][0]["model"] == "local"
    assert (await worker.run_agent(ctx, payload))["status"] == "done"
    assert box["models"][-2:] == ["local", "cloud_fast"]
    evs = await events(redis, res.run_id)
    assert ("status", "requeued") in [(t, d.get("status")) for t, d in evs if t == "status"]
    done = evs[-1][1]
    assert done["text"].startswith("Paris.") and done["verify"]["escalations"] == 1
    assert await _messages(ctx) == ["hi", "Hello!", "capital of France?", "Paris."]  # failed answer dropped


async def test_resumed_turn_is_rerun_with_the_owners_answer(redis, enq, confs, llm, verify_scores):
    box = llm({"assess_ambiguity": [{"ambiguous": True, "question_to_user": "Which phone?"}, {"ambiguous": False}],
         "draft": ["Dunno.", "The Pixel costs $499."], "extract_claims": {"claims": []}})
    ctx = {**CTX(), "redis": FakeArq()}
    await pipeline.submit(job("how much is that one", "a"), enq)
    await worker.run_agent(ctx, enq.calls[-1]["payload"])
    await pipeline.submit(job("Pixel", "b"), enq)
    verify_scores.extend([0.1, 0.9])
    assert (await worker.run_agent(ctx, enq.calls[-1]["payload"]))["status"] == "requeued"
    assert (await worker.run_agent(ctx, ctx["redis"].jobs[-1]))["status"] == "done"
    msgs = await _messages(ctx)
    assert msgs[:3] == ["how much is that one", "Which phone?", "Pixel"] and len(msgs) == 4  # "Dunno." dropped
    assert "The Pixel costs $499." in msgs[3]
    assert box["llm"].calls.count("assess_ambiguity") == 2  # the re-run does not ask again


async def test_after_max_escalations_the_answer_is_sent_with_a_warning_and_an_inbox_item(
        redis, enq, confs, llm, verify_scores, monkeypatch):
    added = []

    async def add(kind, title, body="", **kw):
        added.append((kind, title, kw))

    monkeypatch.setattr(worker.inbox, "add", add)
    box = llm({"draft": "Wrong.", "extract_claims": {"claims": []}})
    ctx = {**CTX(), "redis": FakeArq()}
    res = await pipeline.submit(job("capital of France?", "a"), enq)
    verify_scores.extend([0.1, 0.1, 0.1])
    payload = enq.calls[-1]["payload"]
    while (await worker.run_agent(ctx, payload))["status"] == "requeued":
        payload = ctx["redis"].jobs[-1]
    assert box["models"] == ["local", "cloud_fast", "cloud_strong"]
    done = (await events(redis, res.run_id))[-1][1]
    assert done["text"].startswith("Wrong.") and "⚠️" in done["text"]
    assert done["verify"]["stopped"] == "max escalations" and done["verify"]["escalations"] == 2
    ((kind, _, kw),) = added
    assert kind == "verify" and kw["run_id"] == res.run_id


async def test_private_answer_is_checked_locally_and_never_escalated(redis, enq, confs, llm, verify_scores, monkeypatch):
    called = []

    async def jev(*a):
        called.append("jev")
        return 1.0

    async def qwen(*a):
        called.append("qwen")
        return 0.1

    monkeypatch.setattr(worker.verify, "jev_score", jev)
    monkeypatch.setattr(worker.verify, "qwen_score", qwen)
    box = llm({"draft": "Wrong.", "extract_claims": {"claims": []}})
    ctx = {**CTX(), "redis": FakeArq()}
    await state.mark_thread_private(redis, "web:t")
    await pipeline.submit(job("my balance?", "a"), enq)
    payload = enq.calls[-1]["payload"]
    assert payload["decision"]["private"]
    assert (await worker.run_agent(ctx, payload))["status"] == "done"
    assert called == ["qwen"] and not ctx["redis"].jobs and set(box["models"]) == {"local"}
