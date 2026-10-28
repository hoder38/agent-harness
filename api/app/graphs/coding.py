"""Coding route (§9a, phase 5): Copilot CLI in a throwaway sandbox → harness-run tests → diff → approval → branch.

No LLM of our own: the owner's request goes to Copilot verbatim. The approval is a clarify interrupt
(options Approve / Reject; any other reply = revise), so web and LINE need nothing new.
"""
from __future__ import annotations

import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from ..tools.sandbox import JobCancelled, SandboxError

SKIP = "/skip"
_APPROVE = re.compile(r"^\s*(1|approve|approved|yes|y|ok|okay|lgtm|ship it|好|好的|可以|同意|核准|批准|確認|ok啦)\s*[.!。！]*\s*$",
                      re.IGNORECASE)
_REJECT = re.compile(r"^\s*(2|reject|no|n|cancel|discard|不要|不用|取消|拒絕|放棄|丟掉)\s*[.!。！]*\s*$", re.IGNORECASE)
_CJK = re.compile(r"[\u4e00-\u9fff]")

Notify = Callable[[str], Awaitable[None]]
Cancelled = Callable[[], Awaitable[bool]]


class CodingState(MessagesState, total=False):
    run_id: str
    thread_id: str
    channel: str
    private: bool
    model: str
    force_clarify: bool
    killswitch: str
    coding: dict[str, Any]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content or [])


def pick_repo(text: str, repos: dict[str, Any]) -> str | None:
    """The one repo the text names (by key or alias); None if none or several."""
    low = (text or "").lower()
    hits = set()
    for name, r in repos.items():
        for a in [name, *(r.get("aliases") or [])]:
            a = str(a).lower()
            if a and (re.search(rf"(?<![\w/-]){re.escape(a)}(?![\w-])", low) if a.isascii() else a in low):
                hits.add(name)
    return hits.pop() if len(hits) == 1 else None


def resolve_choice(value: str, names: list[str], repos: dict[str, Any]) -> str | None:
    v = value.strip()
    if v.isdigit() and 1 <= int(v) <= len(names):
        return names[int(v) - 1]
    return next((n for n in names if n.lower() == v.lower()), None) or pick_repo(v, repos)


def branch_name(request: str, task_id: str) -> str:
    words = re.findall(r"[a-z0-9]+", request.lower())[:5]
    slug = "-".join(words)[:40].strip("-") or "task"
    return f"agent/{slug}-{task_id[-6:]}"


def build_prompt(c: dict[str, Any], repo: dict[str, Any]) -> str:
    test = repo.get("test") or {}
    cmd = test.get("command")
    cmd = " ".join(cmd) if isinstance(cmd, list) else str(cmd or "")
    parts = [
        f'You are working in a throwaway git clone of the repository "{c["repo"]}" '
        f'({repo.get("description", "")}) at /workspace.',
        f"Task from the owner:\n{c['request']}",
        "Rules:\n"
        "- Edit the files directly. Keep the change focused; follow the repo's existing style and any "
        "copilot-instructions / AGENTS / README guidance in it.\n"
        "- Do not commit, push, or change git config or remotes: the harness reviews the diff and commits it "
        "after the owner approves.\n"
        "- There is no internet access except package registries, and nothing is installed beyond the repo. "
        f"After you finish, the harness runs the tests itself ({cmd[:300] or 'none configured'}), so add or "
        "update tests for the behaviour you change.\n"
        "- End with a short summary: what you changed, why, and anything the owner should check. "
        "Answer in the owner's language.",
    ]
    if c.get("feedback"):
        parts.append("Your earlier changes are already in /workspace (see `git diff`). " + c["feedback"])
    return "\n\n".join(parts)


def _zh(c: dict[str, Any]) -> bool:
    return bool(_CJK.search(c.get("request") or ""))


def approval_question(c: dict[str, Any], diff_chars: int) -> str:
    zh = _zh(c)
    d = c.get("diff") or {}
    t = c.get("tests") or {}
    cp = c.get("copilot") or {}
    if t.get("skipped"):
        tests = "⚪ 未設定測試" if zh else "⚪ no tests configured"
    elif t.get("passed"):
        tests = "✅ 測試通過" if zh else "✅ tests passed"
    else:
        tail = (t.get("output") or "")[-1200:]
        tests = (f"❌ 測試失敗 (exit {t.get('exit_code')})" if zh else f"❌ tests failed (exit {t.get('exit_code')})")
        tests += f"\n```\n{tail}\n```"
    body = (d.get("diff") or "")[:diff_chars]
    more = d.get("truncated") or len(d.get("diff") or "") > diff_chars
    head = (f"**{c['repo']} 的修改已完成**（task `{c['task_id']}`）" if zh
            else f"**Change ready in {c['repo']}** (task `{c['task_id']}`)")
    tail = (f"核准 → 在 {c['repo']} 建立本地分支 `{c['branch']}` 並提交（不會 push）。拒絕 → 丟棄。"
            "或直接回覆要修改的地方。" if zh else
            f"Approve → commit to a new local branch `{c['branch']}` in {c['repo']} (nothing is pushed). "
            "Reject → discard. Or reply with what to change.")
    return "\n\n".join(x for x in [
        head,
        (cp.get("final") or "").strip()[:1500],
        tests,
        f"```\n{(d.get('stat') or '').strip()}\n```",
        f"```diff\n{body}\n```" + (("\n（diff 已截斷）" if zh else "\n(diff truncated)") if more else ""),
        tail,
    ] if x)


def build_coding_graph(client, *, config: dict[str, Any] | None = None, notify: Notify | None = None,
                       cancelled: Cancelled | None = None, checkpointer=None):
    config = config or {}
    repos: dict[str, Any] = config.get("repos") or {}
    names = list(repos)
    sb = config.get("sandbox") or {}
    cc = config.get("coding") or {}
    max_retries = int(cc.get("max_retries", 1))
    max_revisions = int(cc.get("max_revisions", 2))
    poll_s = float(cc.get("poll_s", 3))
    diff_chars = int(cc.get("diff_chars", 6000))

    async def say(text: str) -> None:
        if notify:
            await notify(text)

    async def start(s: CodingState) -> dict[str, Any]:
        msgs = s.get("messages") or []
        req = next((_text(m.content) for m in reversed(msgs) if isinstance(m, HumanMessage)), "")
        c: dict[str, Any] = {"request": req.strip(), "repo": None, "task_id": None, "attempt": 0, "revisions": 0,
                             "asked": 0, "status": "new", "usage": {"premium_requests": 0, "ai_credits": 0.0}}
        zh = _zh(c)
        if s.get("private"):
            c["status"], c["reply"] = "refused", ("🔒 私密對話不能使用雲端的 Copilot 寫程式。" if zh else
                                                  "🔒 Private threads can't use the cloud coding sandbox (Copilot).")
        elif s.get("killswitch") in ("cloud", "all"):
            c["status"], c["reply"] = "refused", (f"⏸ Kill switch={s.get('killswitch')}：Copilot 是雲端服務，暫停中。"
                                                  if zh else f"⏸ Kill switch={s.get('killswitch')}: Copilot is a "
                                                  "cloud service, so coding is paused.")
        elif not c["request"]:
            c["status"], c["reply"] = "refused", ("要我改什麼？用 /code 加上需求。" if zh else
                                                  "What should I change? Send /code followed by the task.")
        else:
            c["repo"] = names[0] if len(names) == 1 else pick_repo(c["request"], repos)
        return {"coding": c}

    async def ask_repo(s: CodingState) -> dict[str, Any]:
        c = dict(s["coding"])
        zh = _zh(c)
        q = "要改哪個專案？" if zh else "Which repository should I work on?"
        value = interrupt({"type": "clarify", "round": c["asked"] + 1, "question": q, "options": names,
                           "private": bool(s.get("private"))})
        value = "" if value is None else str(value).strip()
        c["asked"] += 1
        if not value or value == SKIP:
            c["status"], c["reply"] = "cancelled", ("已取消：沒有選專案。" if zh else "Cancelled: no repository chosen.")
        elif repo := resolve_choice(value, names, repos):
            c["repo"] = repo
        elif c["asked"] >= 2:
            c["status"], c["reply"] = "cancelled", ("已取消：找不到這個專案。" if zh else
                                                    f"Cancelled: {value!r} is not one of {', '.join(names)}.")
        return {"coding": c}

    async def fail(c: dict[str, Any], e: Exception) -> dict[str, Any]:
        cancel = isinstance(e, JobCancelled)
        c["status"] = "cancelled" if cancel else "error"
        c["reply"] = ("⏹ Cancelled; the sandbox was removed." if cancel else f"⚠️ Coding sandbox failed: {e}")
        if c.get("task_id") and client is not None:
            try:
                await client.delete(c["task_id"])
            except Exception:
                pass
        return {"coding": c}

    async def run_copilot(s: CodingState) -> dict[str, Any]:
        c = dict(s["coding"])
        repo = repos[c["repo"]]
        try:
            if not c.get("task_id"):
                c["task_id"] = f"t{(s.get('run_id') or uuid.uuid4().hex)[:12]}-{uuid.uuid4().hex[:6]}"
                await client.create(c["task_id"], c["repo"])
                c["branch"] = branch_name(c["request"], c["task_id"])
            await say(("⏳ Copilot 正在修改 " if _zh(c) else "⏳ Copilot is working on ") + f"{c['repo']}…\n")
            res = await client.run_job(c["task_id"], "copilot", build_prompt(c, repo), poll_s=poll_s,
                                       timeout_s=float(sb.get("copilot_timeout_s", 1200)), cancelled=cancelled)
        except (SandboxError, JobCancelled) as e:
            return await fail(c, e)
        u = res.get("usage") or {}
        c["usage"] = {"premium_requests": (c["usage"].get("premium_requests") or 0) + (u.get("premium_requests") or 0),
                      "ai_credits": round((c["usage"].get("ai_credits") or 0) + (u.get("ai_credits") or 0), 3)}
        c["copilot"] = {"final": res.get("final") or "", "tools": res.get("tools") or [],
                        "exit_code": res.get("exit_code")}
        if res.get("exit_code") and not res.get("final"):
            return await fail(c, SandboxError(f"copilot exited {res.get('exit_code')}: "
                                              f"{(res.get('stderr') or '').strip()[-500:]}"))
        c["feedback"] = None
        c["status"] = "coded"
        return {"coding": c}

    async def run_tests(s: CodingState) -> dict[str, Any]:
        c = dict(s["coding"])
        if not repos[c["repo"]].get("test"):
            c["tests"] = {"skipped": True, "passed": True}
            return {"coding": c}
        await say("🧪 " + ("執行測試…\n" if _zh(c) else "Running the tests…\n"))
        try:
            res = await client.run_job(c["task_id"], "test", poll_s=poll_s,
                                       timeout_s=float(sb.get("test_timeout_s", 900)), cancelled=cancelled)
        except (SandboxError, JobCancelled) as e:
            return await fail(c, e)
        passed = res.get("exit_code") == 0
        c["tests"] = {"passed": passed, "exit_code": res.get("exit_code"), "output": (res.get("output") or "")[-4000:]}
        if not passed and c["attempt"] < max_retries:
            c["attempt"] += 1
            c["feedback"] = ("The harness ran the tests after your change and they FAILED. Fix the cause (keep the "
                             "intended change). Test output (tail):\n" + c["tests"]["output"][-3000:])
            await say("🔁 " + ("測試失敗，請 Copilot 修正…\n" if _zh(c) else "Tests failed; asking Copilot to fix…\n"))
        return {"coding": c}

    async def review(s: CodingState) -> dict[str, Any]:
        c = dict(s["coding"])
        try:
            d = await client.diff(c["task_id"], max_chars=max(diff_chars * 4, 20_000))
        except SandboxError as e:
            return await fail(c, e)
        if not d.get("files"):
            await fail(c, SandboxError("no changes"))
            c["status"] = "no_changes"
            c["reply"] = (("Copilot 沒有修改任何檔案。" if _zh(c) else "Copilot made no changes.")
                          + ("\n\n" + c["copilot"]["final"] if (c.get("copilot") or {}).get("final") else ""))
            return {"coding": c}
        c["diff"] = d
        c["status"] = "awaiting_approval"
        return {"coding": c}

    async def approve(s: CodingState) -> dict[str, Any]:
        c = dict(s["coding"])
        zh = _zh(c)
        value = interrupt({"type": "clarify", "kind": "approval", "round": c["revisions"] + 1,
                           "question": approval_question(c, diff_chars),
                           "options": ["Approve", "Reject"] if not zh else ["核准", "拒絕"],
                           "private": bool(s.get("private"))})
        value = "" if value is None else str(value).strip()
        if _APPROVE.match(value) or value in ("核准",):
            c["status"] = "approved"
        elif not value or value == SKIP or _REJECT.match(value) or value in ("拒絕",):
            c["status"] = "rejected"
        elif c["revisions"] >= max_revisions:
            c["status"] = "awaiting_approval"
            c["revisions"] += 1
        else:
            c["revisions"] += 1
            c["attempt"] = 0
            c["status"] = "revise"
            c["feedback"] = f"The owner reviewed your change and asked for this revision:\n{value}"
        return {"coding": c}

    async def apply(s: CodingState) -> dict[str, Any]:
        c = dict(s["coding"])
        zh = _zh(c)
        t = c.get("tests") or {}
        msg = (f"agent: {c['request'].splitlines()[0][:60]}\n\n{(c.get('copilot') or {}).get('final', '')[:2000]}\n\n"
               f"Task {c['task_id']}; tests {'passed' if t.get('passed') else 'FAILED'}. "
               "Generated by Copilot CLI in an agent-harness sandbox; approved by the owner.")
        try:
            res = await client.apply(c["task_id"], c["branch"], msg)
        except SandboxError as e:
            c["status"], c["reply"] = "error", f"⚠️ Apply failed: {e}"
            return {"coding": c}
        try:
            await client.delete(c["task_id"])
        except Exception:
            pass
        c["status"], c["applied"] = "applied", res
        short, base, path = res["commit"][:8], res.get("base_branch", "HEAD"), res.get("path", c["repo"])
        c["reply"] = (f"✅ 已提交到 {c['repo']} 的本地分支 `{c['branch']}`（commit {short}，基於 {base}），沒有 push。\n"
                      f"檢查：`git -C {path} log -p {base}..{c['branch']}`\n合併：`git -C {path} merge {c['branch']}`"
                      if zh else
                      f"✅ Committed to local branch `{c['branch']}` in {c['repo']} (commit {short}, based on {base}). "
                      f"Nothing was pushed.\nReview: `git -C {path} log -p {base}..{c['branch']}`\n"
                      f"Merge: `git -C {path} merge {c['branch']}`")
        return {"coding": c}

    async def discard(s: CodingState) -> dict[str, Any]:
        c = dict(s["coding"])
        try:
            await client.delete(c["task_id"])
        except Exception:
            pass
        c["reply"] = "🗑 已丟棄這次修改。" if _zh(c) else "🗑 Discarded; nothing was committed."
        return {"coding": c}

    async def finish(s: CodingState) -> dict[str, Any]:
        c = s.get("coding") or {}
        return {"messages": [AIMessage(c.get("reply") or "Done.")]}

    def after_start(s: CodingState) -> str:
        c = s["coding"]
        return "finish" if c.get("reply") else ("run_copilot" if c.get("repo") else "ask_repo")

    def after_ask(s: CodingState) -> str:
        c = s["coding"]
        return "finish" if c.get("reply") else ("run_copilot" if c.get("repo") else "ask_repo")

    def after_copilot(s: CodingState) -> str:
        return "finish" if s["coding"].get("reply") else "run_tests"

    def after_tests(s: CodingState) -> str:
        c = s["coding"]
        if c.get("reply"):
            return "finish"
        return "run_copilot" if c.get("feedback") else "review"

    def after_review(s: CodingState) -> str:
        return "finish" if s["coding"].get("reply") else "approve"

    def after_approve(s: CodingState) -> str:
        return {"approved": "apply", "rejected": "discard", "revise": "run_copilot"}.get(s["coding"]["status"],
                                                                                         "approve")

    g = StateGraph(CodingState)
    for name, fn in [("start", start), ("ask_repo", ask_repo), ("run_copilot", run_copilot),
                     ("run_tests", run_tests), ("review", review), ("approve", approve), ("apply", apply),
                     ("discard", discard), ("finish", finish)]:
        g.add_node(name, fn)
    g.add_edge(START, "start")
    g.add_conditional_edges("start", after_start, ["finish", "run_copilot", "ask_repo"])
    g.add_conditional_edges("ask_repo", after_ask, ["finish", "run_copilot", "ask_repo"])
    g.add_conditional_edges("run_copilot", after_copilot, ["finish", "run_tests"])
    g.add_conditional_edges("run_tests", after_tests, ["finish", "run_copilot", "review"])
    g.add_conditional_edges("review", after_review, ["finish", "approve"])
    g.add_conditional_edges("approve", after_approve, ["apply", "discard", "run_copilot", "approve"])
    g.add_edge("apply", "finish")
    g.add_edge("discard", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)
