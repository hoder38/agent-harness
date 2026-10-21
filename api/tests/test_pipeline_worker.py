"""Pipeline (dedupe, commands, redaction) + LINE webhook + worker streaming/cancel."""
import asyncio
import base64
import hashlib
import hmac
import json
import time

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver

from app import pipeline, state
from app.gates import jev_router
from app.models import Job, RouterResult


@pytest.fixture
def enq():
    calls = []

    async def enqueue(fn, payload, job_id, queue):
        calls.append({"fn": fn, "payload": payload, "job_id": job_id, "queue": queue})

    enqueue.calls = calls
    return enqueue


@pytest.fixture(autouse=True)
def fake_router(monkeypatch):
    async def fake_route(**kw):
        return RouterResult(intent="chat", intent_confidence=0.95, tier="cloud_fast", source="jev")

    monkeypatch.setattr(pipeline, "route", fake_route)


def job(text, key="k1", **kw):
    return Job(**{"channel": "web", "user_id": "owner", "thread_id": "web:t", "text": text, "dedupe_key": key, **kw})


async def test_duplicate_is_dropped(redis, enq):
    assert (await pipeline.submit(job("hi"), enq)).run_id
    assert (await pipeline.submit(job("hi"), enq)).duplicate
    assert len(enq.calls) == 1


async def test_killswitch_commands(redis, enq):
    res = await pipeline.submit(job("/stop cloud", key="a"), enq)
    assert "cloud" in res.command_reply and await state.get_killswitch(redis) == "cloud"
    res = await pipeline.submit(job("hello", key="b"), enq)
    assert res.decision.model == "local"  # cloud blocked
    await pipeline.submit(job("/resume", key="c"), enq)
    assert await state.get_killswitch(redis) == "off"


async def test_payload_text_is_redacted_and_decision_logged(redis, enq, no_db):
    res = await pipeline.submit(job("my key is sk-abcdefghijklmnopqrstuvwxyz"), enq)
    p = enq.calls[0]["payload"]
    assert "sk-abc" not in p["text"] and p["decision"]["model"] == "cloud_fast"
    assert enq.calls[0]["queue"] == "arq:interactive" and enq.calls[0]["job_id"] == res.run_id
    logged = [kw for name, kw in no_db if name == "decision"][0]
    assert "sk-abc" not in json.dumps(logged, default=str)


async def test_private_thread_is_sticky_and_local(redis, enq):
    await pipeline.submit(job("check this", key="a", source="finance"), enq)
    res = await pipeline.submit(job("and now?", key="b"), enq)
    assert res.decision.private and res.decision.model == "local"


# ---------------------------------------------------------------- LINE
def _sign(body: bytes) -> str:
    return base64.b64encode(hmac.new(b"line-secret", body, hashlib.sha256).digest()).decode()


def test_line_signature():
    from app.channels.line import verify_signature

    body = b'{"events":[]}'
    assert verify_signature(body, _sign(body), "line-secret")
    assert not verify_signature(body, "bad", "line-secret")
    assert not verify_signature(body, None, "line-secret")


async def test_line_owner_only_and_dedupe(redis, enq, monkeypatch):
    from app.routes import line as line_route

    replies = []

    async def fake_reply(token, text):
        replies.append(text)

    monkeypatch.setattr(line_route.line, "reply", fake_reply)
    ev = {"type": "message", "webhookEventId": "E1", "replyToken": "R", "source": {"userId": "Ustranger"},
          "message": {"type": "text", "id": "1", "text": "hi"}}
    await line_route.handle_event(ev, enq)
    assert not enq.calls

    ev["source"]["userId"] = "Uowner"
    await line_route.handle_event(ev, enq)
    await line_route.handle_event(ev, enq)  # redelivery
    assert len(enq.calls) == 1
    d = enq.calls[0]["payload"]["delivery"]
    assert d["line_user_id"] == "Uowner" and d["reply_token"] == "R"
    assert enq.calls[0]["payload"]["job"]["thread_id"] == "line:Uowner"


# ---------------------------------------------------------------- worker
class SlowFake(GenericFakeChatModel):
    delay: float = 0.05

    def bind_tools(self, tools, **kw):
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kw):
        for word in "one two three four five six seven eight nine ten".split():
            await asyncio.sleep(self.delay)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


async def _events(r, run_id):
    out = await r.xrange(f"run:{run_id}")
    return [(f["type"], json.loads(f["data"])) for _, f in out]


async def _prepare(redis, enq, text="hello"):
    res = await pipeline.submit(job(text, key=str(time.time_ns())), enq)
    return enq.calls[-1]["payload"], res.run_id


async def test_worker_streams_tokens_and_persists_thread(redis, enq, monkeypatch):
    from app.jobs import worker

    monkeypatch.setattr(worker, "make_llm", lambda alias, ks: SlowFake(messages=iter([AIMessage("x")]), delay=0))
    ctx = {"checkpointer": InMemorySaver(), "redis": None, "job_try": 1}
    payload, run_id = await _prepare(redis, enq)
    assert (await worker.run_agent(ctx, payload))["status"] == "done"
    evs = await _events(redis, run_id)
    tokens = "".join(d for t, d in evs if t == "token")
    assert tokens.startswith("one two") and evs[-1][0] == "done"
    assert evs[-1][1]["text"].strip().endswith("ten")


async def test_worker_cancel_stops_mid_stream(redis, enq, monkeypatch):
    from app.jobs import worker

    monkeypatch.setattr(worker, "make_llm", lambda alias, ks: SlowFake(messages=iter([AIMessage("x")]), delay=0.1))
    ctx = {"checkpointer": InMemorySaver(), "redis": None, "job_try": 1}
    payload, run_id = await _prepare(redis, enq)

    async def cancel_soon():
        await asyncio.sleep(0.35)
        await state.request_cancel(redis, run_id)

    t0 = time.monotonic()
    _, result = await asyncio.gather(cancel_soon(), worker.run_agent(ctx, payload))
    assert result["status"] == "cancelled" and time.monotonic() - t0 < 0.9
    evs = await _events(redis, run_id)
    assert evs[-1][0] == "cancelled"
    assert len([1 for t, _ in evs if t == "token"]) < 10


async def test_worker_killswitch_all_pauses_background(redis, enq, monkeypatch):
    from app.jobs import worker

    requeued = []

    class FakeArq:
        async def enqueue_job(self, *a, **kw):
            requeued.append(kw)

    res = await pipeline.submit(job("bg", key="bg", mode="background"), enq)
    await state.set_killswitch(redis, "all")
    out = await worker.run_agent({"redis": FakeArq(), "checkpointer": InMemorySaver()}, enq.calls[-1]["payload"])
    assert out["status"] == "paused" and requeued[0]["_queue_name"] == "arq:background"
    assert res.decision.queue == "background"
