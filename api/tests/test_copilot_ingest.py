"""Phase 6 (§9b/§9c): Copilot session ingest (parse, redact, job, hook debounce) and the harness-memory MCP tools."""
import json

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app import copilot_ingest, mcp_server, memory
from app.jobs import copilot_jobs
from test_memory_rag import Arq, mem  # noqa: F401  (mem is a fixture)

SID = "17bc759f-0372-46dd-b1d9-9cd85e41c306"


def ev(type_, i, **data):
    return json.dumps({"type": type_, "id": f"e{i}", "data": data})


def session_lines(*, ended=False):
    lines = [
        ev("session.start", 1, startTime="2026-10-29T01:00:00Z", context={"cwd": "/home/hoder/agent-harness"}),
        ev("user.message", 2, content="Fix the GPU governor; my token=ghp_abcdefghijklmnopqrstuvwx12345"),
        ev("tool.execution_start", 3, toolName="edit", toolCallId="t1", arguments={"path": "api/app/governor.py"}),
        ev("tool.execution_start", 4, toolName="bash", toolCallId="t2", arguments={"command": "git commit -m fix"}),
        ev("tool.execution_complete", 5, toolCallId="t2", success=True,
           result={"content": "[main abc1234def] fix governor unload\n 1 file changed\nsee "
                              "https://github.com/hoder38/agent-harness/pull/7"}),
        ev("assistant.message", 6, content="Working on it"),
        ev("assistant.message", 7, content="Fixed: unload now runs on GAMING."),
        ev("user.message", 8, content="Thanks, I prefer pnpm over npm."),
        ev("assistant.message", 9, content="Noted."),
    ]
    if ended:
        lines.append(ev("session.shutdown", 10, accountingSnapshot={"codeChanges": {"filesModified": ["README.md"]}}))
    return lines + ["{truncated json"]


def test_parse_events_collects_turns_files_commits_prs_and_skips_bad_lines():
    d = copilot_ingest.parse_events(session_lines(ended=True), SID)
    assert d.thread_id == "copilot:" + SID and d.cwd == "/home/hoder/agent-harness" and d.ended
    assert [a for _, a in d.turns] == ["Fixed: unload now runs on GAMING.", "Noted."]  # last answer per prompt
    assert d.files == ["api/app/governor.py", "README.md"]
    assert d.commits == ["main abc1234 fix governor unload"] and d.prs == ["hoder38/agent-harness#7"]
    assert d.last_event_id == "e10"
    msgs = copilot_ingest.messages(d)
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage, HumanMessage, AIMessage]
    assert "ghp_" not in msgs[0].content and "[REDACTED]" in msgs[0].content


def test_redact_masks_common_credentials_only():
    t = ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123 and postgres://agent:hunter2@db/x and "
         "AWS AKIAABCDEFGHIJKLMNOP and password: 'p w' but run `git status` for the token flow")
    out = copilot_ingest.redact(t)
    for secret in ("abcdefghijklmnopqrstuvwxyz0123", "hunter2", "AKIAABCDEFGHIJKLMNOP", "p w"):
        assert secret not in out
    assert "git status" in out and "postgres://agent:[REDACTED]@db/x" in out


def test_read_session_rejects_path_tricks(tmp_path):
    (tmp_path / SID).mkdir()
    (tmp_path / SID / "events.jsonl").write_text("\n".join(session_lines()))
    assert copilot_ingest.read_session(SID, str(tmp_path)).n_events == 9
    assert copilot_ingest.read_session("../" + SID, str(tmp_path)) is None
    assert copilot_ingest.read_session("00000000-0000-0000-0000-000000000000", str(tmp_path)) is None


async def test_agent_stop_hooks_are_debounced_and_session_end_runs_now(redis):
    r = Arq(decode_responses=True)
    r.jobs = []
    assert await copilot_jobs.schedule(r, SID, final=False) == "queued"
    assert await copilot_jobs.schedule(r, SID, final=False) == "debounced"
    assert await copilot_jobs.schedule(r, SID, final=True) == "queued"
    assert [(f, p["final"], kw.get("_defer_by")) for f, p, kw in r.jobs] == [
        ("ingest_copilot", False, 60.0), ("ingest_copilot", True, None)]
    assert await r.hget(copilot_jobs.STATE + SID, "ended") == "1"
    await r.aclose()


async def test_ingest_job_writes_thread_episode_and_facts_once(redis, mem, monkeypatch, tmp_path):  # noqa: F811
    (tmp_path / SID).mkdir()
    (tmp_path / SID / "events.jsonl").write_text("\n".join(session_lines(ended=True)))
    monkeypatch.setenv("COPILOT_SESSIONS_DIR", str(tmp_path))
    from app.settings import get_settings
    get_settings.cache_clear()
    facts = {"ops": [{"op": "add", "namespace": "procedures", "text": "The owner prefers pnpm over npm"}]}
    llm = GenericFakeChatModel(messages=iter([
        AIMessage("The owner had Copilot fix the GPU governor unload on GAMING."),
        AIMessage("```json\n" + json.dumps(facts) + "\n```")]))
    monkeypatch.setattr(copilot_jobs, "make_llm", lambda *a, **k: llm)
    saver = InMemorySaver()
    out = await copilot_jobs.ingest_copilot({"checkpointer": saver, "redis": redis}, {"session_id": SID})
    assert out["status"] == "done" and out["episode"] and [o["op"] for o in out["ops"]] == ["add"]

    ep = await memory.get("coding_episodes", SID)
    assert "GPU governor" in ep["text"] and "api/app/governor.py" in ep["text"] and ep["privacy"] == "normal"
    assert ep["source"]["thread_id"] == "copilot:" + SID
    assert (await memory.search("procedures", None))[0]["text"] == "The owner prefers pnpm over npm"
    tup = await saver.aget_tuple({"configurable": {"thread_id": "copilot:" + SID}})
    msgs = tup.checkpoint["channel_values"]["messages"]
    assert [m.content for m in msgs][1] == "Fixed: unload now runs on GAMING." and "ghp_" not in msgs[0].content

    again = await copilot_jobs.ingest_copilot({"checkpointer": saver, "redis": redis}, {"session_id": SID})
    assert again == {"status": "skipped", "reason": "no new events"}  # same last event, facts already extracted
    get_settings.cache_clear()


async def test_reconciler_queues_idle_sessions_without_session_end(redis, monkeypatch, tmp_path):
    import os
    import time

    for sid, age in (("aaaaaaaa-1111", 7200), ("bbbbbbbb-2222", 7200), ("cccccccc-3333", 60), ("dddddddd-4444", 10 ** 7)):
        (tmp_path / sid).mkdir()
        f = tmp_path / sid / "events.jsonl"
        f.write_text("{}")
        os.utime(f, (time.time() - age, time.time() - age))
    monkeypatch.setenv("COPILOT_SESSIONS_DIR", str(tmp_path))
    from app.settings import get_settings
    get_settings.cache_clear()
    r = Arq(decode_responses=True)
    r.jobs = []
    await r.hset(copilot_jobs.STATE + "bbbbbbbb-2222", "ended", "1")
    await copilot_jobs.reconcile_cron({"redis": r})
    assert [p["session_id"] for _, p, _ in r.jobs] == ["aaaaaaaa-1111"]   # too fresh / too old / already ended skipped
    await r.aclose()
    get_settings.cache_clear()


# ---------------------------------------------------------------- harness-memory MCP
async def test_mcp_tools_never_return_private_memories(mem):  # noqa: F811
    await memory.remember("coding_episodes", "Copilot session fixed the governor unload", private=False, key=SID)
    await memory.remember("profile", "The owner's governor unload secret plan", private=True)
    out = await mcp_server.recall("governor unload")
    assert "Copilot session" in out and "secret plan" not in out
    assert "Remembered" in await mcp_server.remember("The owner uses vim")
    assert (await memory.search("profile", "vim", private=False))[0]["source"] == {"channel": "mcp"}
    assert "fixed the governor" in await mcp_server.get_thread_summary("copilot:" + SID)
    await memory.remember("episodes", "private chat", private=True, key="web:t1")
    assert await mcp_server.get_thread_summary("web:t1") == "No summary for that thread."


async def test_mcp_gate_serves_only_mcp_path_with_token(monkeypatch):
    from app.settings import get_settings
    monkeypatch.setattr(get_settings(), "harness_token", "tok")
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["path"])

    gate = mcp_server.TokenGate(inner)

    async def call(path, auth):
        sent = []

        async def send(m):
            sent.append(m)

        hdrs = [(b"authorization", auth.encode())] if auth else []
        await gate({"type": "http", "path": path, "headers": hdrs}, None, send)
        return sent[0]["status"] if sent else 200

    assert await call("/mcp", None) == 401 and await call("/mcp", "Bearer nope") == 401
    assert await call("/other", "Bearer tok") == 404 and seen == []
    assert await call("/mcp", "Bearer tok") == 200 and seen == ["/mcp"]
