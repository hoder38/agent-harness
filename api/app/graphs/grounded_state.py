"""State for the grounded-answering graph (§12a). Per-turn fields are reset by `ingest` on each new user turn."""
from typing import Any, Literal, TypedDict

from langgraph.graph import MessagesState

ClaimKind = Literal["factual", "greeting", "creative", "opinion", "reasoning", "math", "instruction"]
ClaimStatus = Literal["pending", "verified", "unverified", "contradicted", "exempt"]
Tier = Literal["T1_primary", "T2_secondary", "T3_reject", "unknown"]


class Clarification(TypedDict):
    round: int
    question: str
    answer: str | None
    slot: str            # referent|scope_time|scope_region|version|goal|conflict


class Claim(TypedDict, total=False):
    id: str
    text: str
    kind: ClaimKind
    status: ClaimStatus
    entity: str | None   # subject of the claim (decides which site is "official")
    time_sensitive: bool
    source_id: str | None  # exactly one
    quote: str | None    # verbatim, ≤ 300 chars
    reason: str | None   # shown to the user when unverified/contradicted
    notes: list[str]     # internal: why research failed so far (pdf, secondary_only, ...)
    tries: int           # internal: research rounds used
    supports: list[dict[str, Any]]  # internal: qualifying (source_id, quote) pairs before one is picked


class Candidate(TypedDict, total=False):
    url: str
    title: str
    snippet: str
    domain: str
    age: str | None
    tier: Tier
    tier_reason: str
    for_claims: list[str]
    fetched: bool


class Source(TypedDict, total=False):
    id: str
    url: str
    final_url: str
    domain: str
    title: str
    tier: Tier
    tier_reason: str
    fetched_at: str
    published_at: str | None
    text_sha256: str
    excerpt: str         # relevant windows only, ≤ 8000 chars
    for_claims: list[str]
    verified: bool       # internal: verify_support has processed it


class Budget(TypedDict):
    search_used: int
    fetch_used: int
    clarify_used: int
    usd: float
    active_ms: int       # waiting for the user is not counted
    exhausted: str | None  # search|fetch|clarify|usd|time


class Answer(TypedDict):
    text: str
    verdict: Literal["complete", "partial", "exempt_only"]
    citations: list[dict]
    unverified: list[str]
    assumptions: list[str]


class GroundedState(MessagesState, total=False):
    run_id: str
    turn_id: str
    thread_id: str
    channel: str
    private: bool
    model: str
    force_clarify: bool
    task_type: str | None               # router's (or the owner's confirmed) jev_router.TASK_TYPES key
    confirm_task: list[str]             # ask the owner to pick between these task types first
    answer_task_type: str | None        # router's reading of a free-text answer to that question
    intent_answered: bool               # internal: the last wait_answer answered the task-type question
    correction: str | None              # previous + current message when this turn corrects the last reply
    question: str                       # turn text + clarification answers
    clarifications: list[Clarification]
    assumptions: list[str]
    pending_ask: dict[str, Any] | None  # the clarifying question about to be asked
    skip_clarify: bool
    draft: str                          # internal, never streamed
    claims: list[Claim]
    candidates: list[Candidate]         # unique by url
    judged: dict[str, str]              # "domain|entity" -> tier from the LLM authority judge
    sources: dict[str, Source]
    queries: list[dict]
    searched: list[str]
    loops: int
    conflicts: list[dict[str, Any]]
    budget: Budget
    turn_started_ms: int
    clock_ms: int                       # start of the current active segment (reset after waiting for the user)
    answer: Answer | None
    answer_body: str
    compose_round: int
    check_failures: list[str]
    shopping: bool                      # this turn runs the shopping branch (§12c)
    shop: dict[str, Any] | None         # kept across turns: {slots, products, picks, terms, at_ms, fetched}


def new_budget() -> Budget:
    return {"search_used": 0, "fetch_used": 0, "clarify_used": 0, "usd": 0.0, "active_ms": 0, "exhausted": None}
