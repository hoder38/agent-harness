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
    resuming: bool = False,
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
        if resuming:
            reasons.append(f"3: {' & '.join(why)} but this answers the graph's question → the graph decides")
        elif clarify_rounds >= max_rounds:
            reasons.append(f"3: {' & '.join(why)} but {clarify_rounds} clarify rounds used "
                           "→ answer with stated assumptions")
        else:
            route = "clarify"
            locked.add("route")
            if "model" not in locked:
                model = "local_cpu" if gpu_state in ("BUSY", "GAMING") else "local"
                locked.add("model")
            reasons.append(f"3: {' & '.join(why)} → clarify on {model}")

    # 3b. unsure what kind of answer is wanted → the graph asks the owner to pick (options from task labels)
    confirm_task = _confirm_task(router, th, policy, reasons)
    if confirm_task and (route not in ("chat", "clarify") or private or resuming or clarify_rounds >= max_rounds):
        reasons.append("3b: not asked (" + ("resuming" if resuming else "private" if private
                       else "clarify cap" if clarify_rounds >= max_rounds else f"route={route}") + ")")
        confirm_task = []

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

    return Decision(route=route, model=model, queue=queue, risky=risky, private=private,
                    task_type=router.task_type, confirm_task=confirm_task, reasons=reasons)


def _confirm_task(router: RouterResult, th: dict[str, Any], policy: dict[str, Any], reasons: list[str]) -> list[str]:
    """Top-2 task types when the top one is unsure (< task_confirm_below) or barely ahead (margin < task_margin),
    and only if one of them changes the pipeline (clarify.confirm_task_types). Single-probability routers
    (LLM/rules) only confirm their own pick when it is one of those types."""
    if not router.task_type:
        return []
    types = set((policy.get("clarify") or {}).get("confirm_task_types", ["shopping"]))
    below, margin = th.get("task_confirm_below", 0.7), th.get("task_margin", 0.3)
    probs = dict(router.task_probs) or {router.task_type: router.intent_confidence}
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    top, p1 = ranked[0]
    if len(ranked) > 1:
        second, p2 = ranked[1]
    elif top in types:
        second, p2 = ("advice" if top != "advice" else "factual"), 1.0 - p1
    else:
        return []
    if not types & {top, second} or (p1 >= below and p1 - p2 >= margin):
        return []
    reasons.append(f"3b: task_type {top}({p1:.2f}) vs {second}({p2:.2f}) → confirm with the owner")
    return [top, second]
