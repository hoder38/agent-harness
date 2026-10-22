"""L6 decision layer — pure function, every rule unit-tested (§7).

Rules are evaluated in order. A rule that sets a field "locks" it, so later soft rules
do not change it. Hard constraints (privacy, kill switch, budget) always apply.
"""
from typing import Any

from .models import (
    LOCAL_MODELS, Budget, Decision, Gate0Result, GpuState, Job, KillSwitch, ModelAlias, Queue, Route,
    RouterResult,
)

def decide(
    job: Job,
    gate0: Gate0Result,
    router: RouterResult,
    gpu_state: GpuState,
    budget: Budget,
    policy: dict[str, Any],
    killswitch: KillSwitch = "off",
    clarify_rounds: int = 0,
) -> Decision:
    th = policy.get("thresholds", {})
    providers: dict[str, str] = policy.get("providers", {"cloud_fast": "openai", "cloud_strong": "anthropic"})
    reasons: list[str] = []
    route: Route = router.intent
    model: ModelAlias = router.tier
    queue: Queue = "background" if job.mode == "background" else "interactive"
    risky = False
    private = gate0.privacy_tag == "PRIVATE"
    locked: set[str] = set()
    reasons.append(f"router[{router.source}]: intent={router.intent}({router.intent_confidence:.2f}) tier={router.tier}")

    # 1. user override
    override = job.override or gate0.override
    if override == "local":
        model = "local"
        locked.add("model")
        reasons.append("1: /local override → local")
    elif override == "cloud":
        model = model if model != "local" else "cloud_fast"
        locked.add("model")
        reasons.append(f"1: /cloud override → {model}")
    elif override == "code":
        route = "coding"
        locked.add("route")
        reasons.append("1: /code override → coding")

    # 2. PRIVATE → local only (hard constraint, beats /cloud)
    if private:
        locked.add("model")
        if gpu_state == "GAMING":
            if job.mode == "interactive":
                model = "local_cpu"
                reasons.append("2: PRIVATE + GAMING + interactive → local_cpu (short answers)")
            else:
                model = "local"
                route = "defer"
                locked.add("route")
                reasons.append("2: PRIVATE + GAMING + background → defer until GPU FREE")
        else:
            if model != "local":
                reasons.append(f"2: PRIVATE → local (was {model})")
            model = "local"

    # 3. ambiguous intent → ask on a local model; the tier is decided once the intent is clear
    conf_clarify = th.get("clarify_below", 0.8)
    amb_th = th.get("ambiguous", 0.7)
    max_rounds = int((policy.get("clarify") or {}).get("max_rounds", 5))
    why = []
    if router.intent_confidence < conf_clarify:
        why.append(f"confidence {router.intent_confidence:.2f} < {conf_clarify}")
    if router.ambiguous > amb_th:
        why.append(f"ambiguous {router.ambiguous:.2f} > {amb_th}")
    if why and "route" not in locked:
        if clarify_rounds >= max_rounds:
            reasons.append(f"3: {' & '.join(why)} but {clarify_rounds} clarify rounds used "
                           "→ answer with stated assumptions")
        else:
            route = "clarify"
            locked.add("route")
            if "model" not in locked:
                model = "local_cpu" if gpu_state in ("BUSY", "GAMING") else "local"
                locked.add("model")
            reasons.append(f"3: {' & '.join(why)} → clarify on {model}")

    # 4. heavy tasks
    heavy = th.get("heavy", 0.6)
    if (router.code_multi > heavy or router.long_ctx > heavy) and "model" not in locked:
        strong_ok = not budget.exceeded(providers.get("cloud_strong", "anthropic"))
        model = "cloud_strong" if strong_ok else "cloud_fast"
        locked.add("model")
        reasons.append(f"4: code_multi={router.code_multi:.2f} long_ctx={router.long_ctx:.2f} → {model}")

    # 5. GPU busy → offload non-private local work to cloud
    if gpu_state in ("BUSY", "GAMING") and model == "local" and not private:
        if "model" in locked:
            if gpu_state == "GAMING":
                model = "local_cpu"
                reasons.append("5: /local while GAMING → local_cpu")
        else:
            model = "cloud_fast"
            reasons.append(f"5: GPU {gpu_state} → cloud_fast")

    # 6. LINE channel
    if job.channel == "line":
        if model == "cloud_strong" and "model" not in locked:
            model = "cloud_fast"
            reasons.append("6: LINE → cloud_fast (short answer)")
        if route == "coding":
            queue = "background"
            reasons.append("6: LINE coding → background, approvals via Inbox")

    # 7. background → stricter budget
    if job.mode == "background" and model == "cloud_strong":
        frac = budget.fraction(providers.get("cloud_strong", "anthropic"))
        if frac >= policy.get("background_budget_fraction", 0.5):
            model = "cloud_fast"
            reasons.append(f"7: background + cloud_strong spend {frac:.0%} → cloud_fast")

    # 8. budget exceeded / kill switch → downgrade
    def _downgrade(why: str) -> None:
        nonlocal model
        if model == "cloud_strong" and not budget.exceeded(providers.get("cloud_fast", "openai")) and killswitch == "off":
            model = "cloud_fast"
        else:
            model = "local_cpu" if gpu_state == "GAMING" else "local"
        reasons.append(f"8: {why} → {model} (notify)")

    if model not in LOCAL_MODELS:
        if killswitch in ("cloud", "all"):
            _downgrade(f"killswitch={killswitch}")
        elif budget.exceeded(providers.get(model, model)):
            _downgrade(f"{providers.get(model, model)} budget exceeded")

    # 9. heavy media → background
    heavy_kinds = set(policy.get("heavy_media_kinds", ["video", "large_doc"]))
    if any(a.kind in heavy_kinds for a in job.attachments):
        queue = "background"
        reasons.append("9: video/large doc → background")

    # 10. risky
    if router.risky > th.get("risky", 0.5):
        risky = True
        reasons.append(f"10: risky={router.risky:.2f} → approval required")

    if route == "coding" and queue != "background":
        queue = "coding"
    if route == "defer":
        queue = "background"

    return Decision(route=route, model=model, queue=queue, risky=risky, private=private, reasons=reasons)
