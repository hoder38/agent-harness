"""Coding route (§9a): sandbox client faked; graph flow, approval interrupt, and worker/pipeline dispatch."""
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app import pipeline, state
from app.graphs.coding import branch_name, build_coding_graph, pick_repo
from app.jobs import worker
from app.models import Job, RouterResult
from app.tools.sandbox import SandboxError

CONF = {
    "sandbox": {},
    "coding": {"max_retries": 1, "poll_s": 0, "diff_chars": 500},
    "repos": {
        "agent-harness": {"path": "/r/h", "aliases": ["harness"], "test": {"command": ["pytest"]}},
        "nodeReact": {"path": "/r/n", "aliases": ["frontend", "前端"], "test": {"command": ["jest"]}},
    },
}
DIFF = {"stat": " a.py | 1 +", "files": ["M\ta.py"], "diff": "+x = 1\n", "truncated": False}


class FakeSandbox:
    def __init__(self, tests=(0,), diff=DIFF, copilot_exit=0):
        self.calls: list[tuple] = []
        self.tests = list(tests)
        self.diff_res = diff
        self.copilot_exit = copilot_exit
        self.prompts: list[str] = []

    async def create(self, task_id, repo):
        self.calls.append(("create", repo))
        return {"task_id": task_id}

    async def run_job(self, task_id, kind, prompt=None, **kw):
        self.calls.append(("job", kind))
        if kind == "copilot":
            self.prompts.append(prompt)
            return {"status": "exited", "exit_code": self.copilot_exit, "final": "Added x." if not self.copilot_exit else "",
                    "tools": ["edit"], "usage": {"premium_requests": 1, "ai_credits": 2.5}, "stderr": "boom"}
        code = self.tests.pop(0) if len(self.tests) > 1 else self.tests[0]
        return {"status": "exited", "exit_code": code, "output": "FAILED test_a" if code else "1 passed"}

    async def diff(self, task_id, max_chars=0):
        self.calls.append(("diff",))
        return self.diff_res

    async def apply(self, task_id, branch, message):
        self.calls.append(("apply", branch))
        return {"branch": branch, "commit": "abcdef123456", "base_branch": "master", "path": "/r/h"}

    async def delete(self, task_id):
        self.calls.append(("delete",))


CFG = {"configurable": {"thread_id": "t"}}


async def run(graph, inp):
    out = await graph.ainvoke(inp, CFG)
    snap = await graph.aget_state(CFG)
    return out, (snap.interrupts[0].value if snap.interrupts else None)


def start(text, **kw):
    return {"messages": [("user", text)], "run_id": "0123456789abcdef", **kw}


def test_pick_repo_and_branch_name():
    repos = CONF["repos"]
    assert pick_repo("fix the harness router", repos) == "agent-harness"
    assert pick_repo("改一下前端的按鈕", repos) == "nodeReact"
    assert pick_repo("fix harness and frontend", repos) is None   # several → ask
    assert pick_repo("fix the bug", repos) is None
    assert pick_repo("harnessing", repos) is None                  # whole words only
    assert branch_name("Add a /health endpoint!", "tabc-123456") == "agent/add-a-health-endpoint-123456"
    assert branch_name("加一個按鈕", "tabc-123456") == "agent/task-123456"


async def test_happy_path_asks_approval_then_commits_branch():
    sb = FakeSandbox()
    g = build_coding_graph(sb, config=CONF, checkpointer=InMemorySaver())
    _, q = await run(g, start("add x to the harness"))
    assert q["kind"] == "approval" and q["options"] == ["Approve", "Reject"]
    assert "Added x." in q["question"] and "✅ tests passed" in q["question"] and "+x = 1" in q["question"]
    assert [c[0] for c in sb.calls] == ["create", "job", "job", "diff"]
    assert "add x to the harness" in sb.prompts[0] and "Do not commit, push" in sb.prompts[0]
    out, q = await run(g, Command(resume="approve"))
    assert q is None and ("apply", "agent/add-x-to-the-harness-" + out["coding"]["task_id"][-6:]) in sb.calls
    assert sb.calls[-1] == ("delete",)
    assert "Committed to local branch" in out["messages"][-1].content and "Nothing was pushed" in out["messages"][-1].content
    assert out["coding"]["usage"] == {"premium_requests": 1, "ai_credits": 2.5}


async def test_failed_tests_retry_once_with_the_failure_output():
    sb = FakeSandbox(tests=(1, 1))
    g = build_coding_graph(sb, config=CONF, checkpointer=InMemorySaver())
    _, q = await run(g, start("fix harness"))
    assert [c[1] for c in sb.calls if c[0] == "job"] == ["copilot", "test", "copilot", "test"]   # max_retries=1
    assert "FAILED" in sb.prompts[1] and "tests FAILED" not in sb.prompts[0]
    assert "❌ tests failed (exit 1)" in q["question"]          # still shown; the owner decides


async def test_unclear_repo_is_asked_with_options():
    sb = FakeSandbox()
    g = build_coding_graph(sb, config=CONF, checkpointer=InMemorySaver())
    _, q = await run(g, start("fix the login bug"))
    assert q["type"] == "clarify" and q["options"] == ["agent-harness", "nodeReact"] and not sb.calls
    _, q = await run(g, Command(resume="2"))
    assert sb.calls[0] == ("create", "nodeReact") and q["kind"] == "approval"


async def test_reject_discards_and_revise_reruns_copilot():
    sb = FakeSandbox()
    g = build_coding_graph(sb, config=CONF, checkpointer=InMemorySaver())
    await run(g, start("fix harness"))
    _, q = await run(g, Command(resume="also add a test for y"))
    assert "also add a test for y" in sb.prompts[1] and q["kind"] == "approval" and q["round"] == 2
    out, q = await run(g, Command(resume="Reject"))
    assert q is None and not any(c[0] == "apply" for c in sb.calls) and sb.calls[-1] == ("delete",)
    assert "Discarded" in out["messages"][-1].content


async def test_private_killswitch_no_changes_and_copilot_errors():
    sb = FakeSandbox()
    g = build_coding_graph(sb, config=CONF, checkpointer=InMemorySaver())
    out, _ = await run(g, start("fix harness", private=True))
    assert "Private" in out["messages"][-1].content and not sb.calls
    g = build_coding_graph(sb, config=CONF, checkpointer=InMemorySaver())
    out, _ = await run(g, start("fix harness", killswitch="cloud"))
    assert "Kill switch" in out["messages"][-1].content and not sb.calls

    sb = FakeSandbox(diff={"files": [], "stat": "", "diff": ""})
    g = build_coding_graph(sb, config=CONF, checkpointer=InMemorySaver())
    out, q = await run(g, start("fix harness"))
    assert q is None and "made no changes" in out["messages"][-1].content and ("delete",) in sb.calls

    sb = FakeSandbox(copilot_exit=1)
    g = build_coding_graph(sb, config=CONF, checkpointer=InMemorySaver())
    out, _ = await run(g, start("fix harness"))
    assert "copilot exited 1: boom" in out["messages"][-1].content and sb.calls[-1] == ("delete",)


async def test_sandbox_errors_end_the_run_with_a_message():
    class Broken(FakeSandbox):
        async def create(self, task_id, repo):
            raise SandboxError("sandbox-manager POST /tasks: 503 down")

    g = build_coding_graph(Broken(), config=CONF, checkpointer=InMemorySaver())
    out, _ = await run(g, start("fix harness"))
    assert "Coding sandbox failed" in out["messages"][-1].content


# ---------------------------------------------------------------- worker + pipeline
@pytest.fixture
def enq():
    calls = []

    async def enqueue(fn, payload, job_id, queue):
        calls.append({"fn": fn, "payload": payload, "job_id": job_id, "queue": queue})

    enqueue.calls = calls
    return enqueue


@pytest.fixture
def sandbox(monkeypatch):
    sb = FakeSandbox()
    monkeypatch.setattr(worker, "SandboxClient", lambda: sb)
    monkeypatch.setattr(worker, "coding_config", lambda: CONF)
    monkeypatch.setattr(worker, "make_llm", lambda alias, ks: None)

    async def fake_route(**kw):   # an approval word on its own looks like chat to the router
        return RouterResult(intent="chat", intent_confidence=0.95, tier="local", source="jev")

    monkeypatch.setattr(pipeline, "route", fake_route)
    return sb


def job(text, key):
    return Job(channel="web", mode="interactive", user_id="owner", thread_id="web:t", text=text, dedupe_key=key)


async def test_code_command_runs_the_coding_graph_and_approval_resumes_it(redis, enq, sandbox):
    ctx = {"checkpointer": InMemorySaver(), "redis": None, "job_try": 1}
    first = await pipeline.submit(job("/code add x to the harness", "a"), enq)
    assert first.decision.route == "coding" and enq.calls[-1]["queue"] == "arq:background"
    assert (await worker.run_agent(ctx, enq.calls[-1]["payload"]))["status"] == "awaiting_input"
    evs = [(f["type"], json.loads(f["data"])) for _, f in await redis.xrange(f"run:{first.run_id}")]
    assert any(t == "token" and "Copilot is working" in d for t, d in evs)
    assert evs[-1][0] == "interrupt" and evs[-1][1]["kind"] == "approval"
    pending = await state.get_pending(redis, "web:t")
    assert pending["kind"] == "coding" and pending["run_id"] == first.run_id

    second = await pipeline.submit(job("1", "b"), enq)        # option 1 = Approve; router said chat
    assert second.decision.route == "coding"
    assert (await worker.run_agent(ctx, enq.calls[-1]["payload"]))["status"] == "done"
    t, done = [(f["type"], json.loads(f["data"])) for _, f in await redis.xrange(f"run:{second.run_id}")][-1]
    assert t == "done" and "Committed to local branch" in done["text"]
    assert done["coding"]["status"] == "applied" and done["coding"]["tests_passed"] is True
    assert any(c[0] == "apply" for c in sandbox.calls)
