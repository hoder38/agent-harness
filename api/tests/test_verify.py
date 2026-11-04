"""L9 verify checks (§12, phase 10b)."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import verify


def det(body, messages=(), **kw):
    c = {"simplified_min_chars": 3, "require_citations": True}
    return verify.deterministic(body, list(messages), grounding=kw.get("grounding"),
                                required_tool=kw.get("required_tool"), c=c)


def test_deterministic_checks():
    assert det("  ") == ["empty answer"]
    assert det("台北今天會下雨,這裡的氣溫約攝氏二十度,後天放晴,記得帶傘。") == []  # Traditional passes
    assert det("这个问题应该这样说")[0].startswith("Simplified Chinese")
    assert det("这个问题应该", grounding={"verdict": "products"}) == []  # store titles may be Simplified
    assert det("ok", grounding={"verdict": "uncited"}) == ["sources were read but none is cited"]
    turn = [HumanMessage("old", id="r0"), ToolMessage("boom", tool_call_id="x", name="web_search", status="error"),
            HumanMessage("add a todo", id="r1"), AIMessage("Done")]
    assert det("Done", turn, required_tool="add_todo") == ["required tool add_todo not called"]  # old turn ignored
    turn.insert(3, ToolMessage("boom", tool_call_id="y", name="add_todo", status="error"))
    assert det("Done", turn, required_tool="add_todo") == ["tool error: add_todo"]
    assert verify.wrote_something(turn)


def test_private_never_reaches_cloud():
    assert verify.checker_for(private=False, gpu="FREE", killswitch="off") == "jev"
    assert verify.checker_for(private=True, gpu="FREE", killswitch="off") == "qwen"
    assert verify.checker_for(private=False, gpu="FREE", killswitch="cloud") == "qwen"
    assert verify.checker_for(private=True, gpu="GAMING", killswitch="off") == "none"
    assert verify.next_model("local", private=False, killswitch="off") == "cloud_fast"
    assert verify.next_model("cloud_fast", private=False, killswitch="off") == "cloud_strong"
    assert verify.next_model("cloud_strong", private=False, killswitch="off") is None
    assert verify.next_model("local", private=True, killswitch="off") is None
    assert verify.next_model("local", private=False, killswitch="cloud") is None


async def test_model_score_threshold_and_checker_errors_pass():
    async def low(*a):
        return 0.1

    async def broken(*a):
        raise TimeoutError

    kw = dict(request="q", answer="a", body="a", messages=[], private=False, gpu="FREE", killswitch="off")
    res = await verify.check(**kw, jev=low)
    assert not res.passed and res.checker == "jev" and res.fails[0].startswith("jev score 0.10")
    res = await verify.check(**kw, jev=broken)
    assert res.passed and res.error
    assert not (await verify.check(**{**kw, "body": ""}, jev=broken)).passed  # deterministic first, no model call


def test_trivial_chitchat_and_clarify_are_not_checked():
    c = {"enabled": True, "skip_task_types": ["chitchat"], "trivial_chars": 300}
    assert not verify.should_check(route="chat", task_type="chitchat", body="Hi!", c=c)
    assert verify.should_check(route="chat", task_type="chitchat", body="x" * 400, c=c)
    assert not verify.should_check(route="clarify", task_type=None, body="Which one?", c=c)
    assert not verify.should_check(route="chat", task_type=None, body="x", c={**c, "enabled": False})
