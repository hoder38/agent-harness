"""Rolling conversation compaction (app/history.py)."""
import fakeredis
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app import history
from app.graphs.grounded import plain_history

C = {"enabled": True, "keep_turns": 2, "compact_every": 3, "max_tokens": 10_000, "topic_keep_turns": 1,
     "pin_min_chars": 50, "pin_max_chars": 100, "summary_max_chars": 500}


def convo(n: int) -> list:
    out = []
    for i in range(n):
        out += [HumanMessage(f"q{i}"), AIMessage(f"a{i}")]
    return out


def summ(covered: int, msgs: list, **kw) -> dict:
    hs = history.human_idx(msgs)
    return {"summary": "S", "pinned": "", "covered": covered, "private": False,
            "boundary": history._sha(msgs[hs[covered - 1]].content), **kw}


def test_inject_replaces_folded_turns_with_the_summary_and_follows_a_shifted_boundary():
    msgs = convo(5)
    view = history.inject(msgs, summ(3, msgs), 20)
    assert "S" in view[0].content and isinstance(view[1], AIMessage)
    assert [m.content for m in view[2:]] == ["q3", "a3", "q4", "a4"]
    shifted = [HumanMessage("q-extra"), AIMessage("x"), *msgs]   # covered count off by one: found by content
    assert [m.content for m in history.inject(shifted, summ(3, msgs), 20)[2:]] == ["q3", "a3", "q4", "a4"]
    gone = convo(2)                                              # boundary message not in the thread: no summary
    assert history.inject(gone, summ(3, msgs), 20) == gone
    assert [m.content for m in plain_history(msgs, 20, summ(3, msgs))][2:] == ["q3", "a3", "q4", "a4"]


def test_pinned_paste_is_shown_only_while_no_newer_long_paste_is_verbatim(monkeypatch):
    monkeypatch.setattr(history, "cfg", lambda: C)
    msgs = convo(4)
    s = summ(2, msgs, pinned="OLD PASTE")
    assert "OLD PASTE" in history.inject(msgs, s, 20)[0].content
    msgs[-2] = HumanMessage("x" * 60)
    assert "OLD PASTE" not in history.inject(msgs, s, 20)[0].content


def test_plan_triggers_on_turn_count_tokens_and_topic_change():
    assert history.plan(convo(4), None, topic_changed=False, c=C) is None
    assert history.plan(convo(5), None, topic_changed=False, c=C) == 2          # 5 - 0 - 2 >= 3
    msgs = convo(5)
    assert history.plan(msgs, summ(2, msgs), topic_changed=False, c=C) is None  # only 1 unsummarised beyond keep
    big = convo(3)
    big[0] = HumanMessage("字" * 20_000)
    assert history.plan(big, None, topic_changed=False, c=C) == 2
    assert history.plan(convo(2), None, topic_changed=True, c=C) == 1
    assert history.plan(convo(1), None, topic_changed=True, c=C) is None


def test_transcript_drops_tools_and_thinking_and_pins_the_latest_long_paste():
    msgs = [HumanMessage("p" * 60), AIMessage("", tool_calls=[{"name": "web_search", "args": {}, "id": "1"}]),
            ToolMessage("raw tool output", tool_call_id="1"), AIMessage("<think>hmm</think>answer"),
            HumanMessage("L" * 70), AIMessage("ok")]
    text, pinned = history.transcript(msgs, pin_min=50)
    assert "raw tool output" not in text and "hmm" not in text and "answer" in text
    assert pinned == "L" * 70 and "p" * 60 in text


class Arq(fakeredis.FakeAsyncRedis):
    async def enqueue_job(self, fn, payload, **kw):
        self.jobs.append((fn, payload))


async def test_after_turn_queues_compaction_on_topic_change_but_ignores_chitchat(monkeypatch):
    monkeypatch.setattr(history, "cfg", lambda: C)
    r = Arq(decode_responses=True)
    r.jobs = []
    msgs = convo(3)
    assert await history.after_turn(r, thread_id="web:t", messages=msgs, task_type="factual", private=False) is None
    assert await history.after_turn(r, thread_id="web:t", messages=msgs, task_type="chitchat", private=False) is None
    assert await history.after_turn(r, thread_id="web:t", messages=msgs, task_type="howto", private=True) == 1
    assert r.jobs == [("compact_thread", {"thread_id": "web:t", "keep": 1, "private": True})]
    await r.aclose()


class FakeLLM:
    def __init__(self):
        self.seen = []

    def bind(self, **kw):
        return self

    async def ainvoke(self, msgs, config=None):
        self.seen.append(msgs)
        return AIMessage("<think>x</think>NEW SUMMARY")


async def test_compact_thread_folds_turns_before_keep_into_the_previous_summary(monkeypatch, redis):
    from app.jobs import memory_jobs
    from app.llm import client

    msgs = convo(6)
    prev = summ(2, msgs, summary="OLD")
    saved = {}
    llm = FakeLLM()

    async def load(tid, private):
        return prev

    async def save(tid, **kw):
        saved.update(kw)

    async def thread_messages(cp, tid):
        return msgs

    monkeypatch.setattr(history, "cfg", lambda: C)
    monkeypatch.setattr(history, "load", load)
    monkeypatch.setattr(history, "save", save)
    monkeypatch.setattr(memory_jobs, "thread_messages", thread_messages)
    monkeypatch.setattr(client, "make_llm", lambda *a, **kw: llm)
    out = await history.compact_thread({"checkpointer": None}, {"thread_id": "web:t", "keep": 2})
    assert out == {"status": "done", "covered": 4, "chars": 11}
    sent = llm.seen[0][1].content
    assert "OLD" in sent and "q2" in sent and "q3" in sent and "q1" not in sent and "q4" not in sent
    assert isinstance(llm.seen[0][0], SystemMessage)
    assert saved["summary"] == "NEW SUMMARY" and saved["covered"] == 4
    assert history.inject(msgs, {**prev, **saved}, 20)[2].content == "q4"
