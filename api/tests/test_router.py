"""Router fallback chain (§6)."""
import pytest

from app.gates.jev_router import route, rule_route
from app.models import RouterResult


def ok(source):
    async def f(text, *a):
        f.calls.append((text, a))
        return RouterResult(intent="chat", intent_confidence=0.9, tier="local", source=source)
    f.calls = []
    return f


def boom():
    async def f(text, *a):
        f.calls.append((text, a))
        raise TimeoutError("down")
    f.calls = []
    return f


def llm_by_alias(**fns):
    async def f(text, alias, timeout, source):
        return await fns[alias](text, alias)
    return f


POL = {"router": {"rule_confidence": 0.6}}


async def _route(jev, llm, private=False, gpu="FREE", ks="off"):
    return await route(text="secret text", redacted_text="[REDACTED]", private=private, gpu_state=gpu,
                       killswitch=ks, policy=POL, jev=jev, llm=llm)


async def test_jev_used_with_redacted_text():
    jev = ok("jev")
    r = await _route(jev, llm_by_alias(local=boom()))
    assert r.source == "jev" and jev.calls[0][0] == "[REDACTED]"


async def test_jev_outage_falls_back_to_qwen():
    r = await _route(boom(), llm_by_alias(local=ok("qwen")))
    assert r.source == "qwen" and r.errors and r.errors[0].startswith("jev")


async def test_private_never_calls_jev():
    jev = ok("jev")
    r = await _route(jev, llm_by_alias(local=ok("qwen")), private=True)
    assert r.source == "qwen" and not jev.calls


async def test_private_gaming_uses_rules_without_model_calls():
    jev, local = ok("jev"), ok("qwen")
    r = await _route(jev, llm_by_alias(local=local), private=True, gpu="GAMING")
    assert r.source == "rules" and r.intent_confidence == 0.6 and not jev.calls and not local.calls


async def test_gaming_jev_outage_uses_cloud_router_not_qwen():
    local, cloud = ok("qwen"), ok("cloud_fast")
    r = await _route(boom(), llm_by_alias(local=local, cloud_fast=cloud), gpu="GAMING")
    assert r.source == "cloud_fast" and not local.calls and cloud.calls[0][0] == "[REDACTED]"


async def test_killswitch_skips_jev_and_cloud_router():
    jev, cloud = ok("jev"), ok("cloud_fast")
    r = await _route(jev, llm_by_alias(local=boom(), cloud_fast=cloud), gpu="GAMING", ks="cloud")
    assert r.source == "rules" and not jev.calls and not cloud.calls


async def test_everything_down_uses_rules():
    r = await _route(boom(), llm_by_alias(local=boom()))
    assert r.source == "rules" and len(r.errors) == 2


@pytest.mark.parametrize("text,intent", [
    ("/code fix it", "coding"),
    ("Traceback (most recent call last):\n  File x", "coding"),
    ("please refactor src/app/main.py", "coding"),
    ("remind me to call mom", "assistant"),
    ("新增待辦：買牛奶", "assistant"),
    ("why is the sky blue", "chat"),
])
def test_rule_router(text, intent):
    assert rule_route(text).intent == intent


@pytest.mark.parametrize("text,task", [
    ("想買調味料罐 500以下", "shopping"),
    ("推薦適合冷凍且富含纖維素的食材", "advice"),   # 推薦 alone is not a purchase
    ("如何設定 VPN", "howto"),
])
def test_rule_router_task_type(text, task):
    r = rule_route(text)
    assert r.task_type == task and r.task_probs == {task: 0.6}
