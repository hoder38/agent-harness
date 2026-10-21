"""decide() — one or more cases per rule in §7."""
import pytest

from app.decision import decide
from app.models import AttachmentRef, Budget, Gate0Result, Job, RouterResult

NO_BUDGET = Budget(spend_usd={"openai": 0, "anthropic": 0}, limits_usd={"openai": 5, "anthropic": 5})


def job(**kw):
    return Job(**{"channel": "web", "user_id": "owner", "thread_id": "web:t", "text": "hi", "dedupe_key": "k", **kw})


def g0(private=False, override=None):
    return Gate0Result(text="hi", redacted_text="hi", override=override, privacy_tag="PRIVATE" if private else "NORMAL")


def rr(**kw):
    return RouterResult(**{"intent": "chat", "intent_confidence": 0.95, "tier": "local", "source": "jev", **kw})


def d(policy, j=None, g=None, r=None, gpu="FREE", budget=NO_BUDGET, ks="off"):
    return decide(j or job(), g or g0(), r or rr(), gpu, budget, policy, killswitch=ks)


def test_default_follows_router(policy):
    out = d(policy, r=rr(intent="assistant", tier="cloud_fast"))
    assert (out.route, out.model, out.queue, out.risky) == ("assistant", "cloud_fast", "interactive", False)


# 1. override
def test_override_local_beats_router_tier(policy):
    assert d(policy, g=g0(override="local"), r=rr(tier="cloud_strong", code_multi=0.9)).model == "local"


def test_override_cloud_upgrades_local(policy):
    assert d(policy, j=job(override="cloud")).model == "cloud_fast"


def test_override_code_forces_coding_route(policy):
    out = d(policy, g=g0(override="code"), r=rr(intent_confidence=0.2))
    assert out.route == "coding" and out.queue == "coding"


# 2. PRIVATE
def test_private_forces_local_even_with_cloud_override(policy):
    out = d(policy, j=job(override="cloud"), g=g0(private=True), r=rr(tier="cloud_strong"))
    assert out.model == "local" and out.private


def test_private_gaming_interactive_uses_cpu(policy):
    assert d(policy, g=g0(private=True), gpu="GAMING").model == "local_cpu"


def test_private_gaming_background_defers(policy):
    out = d(policy, j=job(mode="background"), g=g0(private=True), gpu="GAMING")
    assert out.route == "defer" and out.queue == "background"


# 3. confidence
def test_low_confidence_clarifies(policy):
    assert d(policy, r=rr(intent_confidence=0.3)).route == "clarify"


def test_mid_confidence_bumps_tier(policy):
    assert d(policy, r=rr(intent_confidence=0.6, tier="local")).model == "cloud_fast"
    assert d(policy, r=rr(intent_confidence=0.6, tier="cloud_strong")).model == "cloud_strong"


# 4. heavy
@pytest.mark.parametrize("field", ["code_multi", "long_ctx"])
def test_heavy_goes_strong(policy, field):
    assert d(policy, r=rr(**{field: 0.9})).model == "cloud_strong"


def test_heavy_with_strong_budget_exceeded_goes_fast(policy):
    b = Budget(spend_usd={"anthropic": 9}, limits_usd={"openai": 5, "anthropic": 5})
    assert d(policy, r=rr(code_multi=0.9), budget=b).model == "cloud_fast"


# 5. GPU busy
@pytest.mark.parametrize("gpu", ["BUSY", "GAMING"])
def test_gpu_busy_offloads_local_to_cloud(policy, gpu):
    assert d(policy, gpu=gpu).model == "cloud_fast"


def test_local_override_while_gaming_uses_cpu(policy):
    assert d(policy, g=g0(override="local"), gpu="GAMING").model == "local_cpu"


# 6. LINE
def test_line_avoids_cloud_strong(policy):
    assert d(policy, j=job(channel="line"), r=rr(tier="cloud_strong")).model == "cloud_fast"


def test_line_coding_goes_background(policy):
    out = d(policy, j=job(channel="line"), r=rr(intent="coding"))
    assert out.route == "coding" and out.queue == "background"


# 7. background budget
def test_background_strict_budget(policy):
    b = Budget(spend_usd={"anthropic": 3}, limits_usd={"openai": 5, "anthropic": 5})  # 60% ≥ 50%
    assert d(policy, j=job(mode="background"), r=rr(tier="cloud_strong"), budget=b).model == "cloud_fast"
    assert d(policy, j=job(mode="interactive"), r=rr(tier="cloud_strong"), budget=b).model == "cloud_strong"


# 8. budget / kill switch
def test_budget_exceeded_downgrades(policy):
    b = Budget(spend_usd={"openai": 6}, limits_usd={"openai": 5, "anthropic": 5})
    out = d(policy, r=rr(tier="cloud_fast"), budget=b)
    assert out.model == "local" and any("budget exceeded" in x for x in out.reasons)


def test_strong_budget_exceeded_falls_to_fast(policy):
    b = Budget(spend_usd={"anthropic": 6}, limits_usd={"openai": 5, "anthropic": 5})
    assert d(policy, r=rr(tier="cloud_strong"), budget=b).model == "cloud_fast"


@pytest.mark.parametrize("ks", ["cloud", "all"])
def test_killswitch_blocks_cloud_even_with_override(policy, ks):
    assert d(policy, j=job(override="cloud"), r=rr(tier="cloud_strong"), ks=ks).model == "local"
    assert d(policy, r=rr(tier="cloud_fast"), ks=ks, gpu="GAMING").model == "local_cpu"


# 9. heavy media
def test_video_goes_background(policy):
    out = d(policy, j=job(attachments=[AttachmentRef(id="a", kind="video")]))
    assert out.queue == "background"


# 10. risky
def test_risky_flag(policy):
    assert d(policy, r=rr(risky=0.7)).risky
    assert not d(policy, r=rr(risky=0.3)).risky
