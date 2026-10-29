from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

Channel = Literal["web", "line", "cron", "event", "copilot_ingest"]
Mode = Literal["interactive", "background"]
Override = Literal["local", "cloud", "code"]
Intent = Literal["coding", "assistant", "chat"]
Tier = Literal["local", "cloud_fast", "cloud_strong"]
ModelAlias = Literal["local", "local_cpu", "cloud_fast", "cloud_strong"]
Route = Literal["chat", "assistant", "coding", "clarify", "defer"]
Queue = Literal["interactive", "background", "coding", "media"]
GpuState = Literal["FREE", "BUSY", "GAMING"]
KillSwitch = Literal["off", "cloud", "all"]

LOCAL_MODELS = ("local", "local_cpu")


class AttachmentRef(BaseModel):
    id: str
    kind: str  # voice | image | video | doc | large_doc
    mime: str | None = None


class Job(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    channel: Channel
    mode: Mode = "interactive"
    user_id: str
    thread_id: str
    text: str | None = None
    attachments: list[AttachmentRef] = []
    override: Override | None = None
    source: str | None = None  # origin tag for Gate 0 (mail, calendar, finance, ...)
    dedupe_key: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Gate0Result(BaseModel):
    text: str                       # text with override prefix stripped
    redacted_text: str
    override: Override | None = None
    findings: list[str] = []        # names of matched secret/PII patterns
    privacy_tag: Literal["PRIVATE", "NORMAL"] | None = None  # None = inconclusive
    reasons: list[str] = []


class RouterResult(BaseModel):
    intent: Intent
    intent_confidence: float
    tier: Tier
    difficulty: float = 0.0   # normalised 0..1
    risky: float = 0.0
    long_ctx: float = 0.0
    code_multi: float = 0.0
    ambiguous: float = 0.0    # needs a clarifying question even given the previous message
    task_type: str | None = None              # jev_router.TASK_TYPES key: what kind of answer is wanted
    task_probs: dict[str, float] = {}         # probability per task type (best effort for LLM/rule routers)
    correction: float = 0.0   # the message corrects how the previous one was understood
    source: Literal["jev", "qwen", "cloud_fast", "rules"]
    latency_ms: int = 0
    errors: list[str] = []
    raw: dict[str, Any] | None = None


class Budget(BaseModel):
    spend_usd: dict[str, float] = {}     # provider -> today's spend
    limits_usd: dict[str, float] = {}

    def fraction(self, provider: str) -> float:
        limit = self.limits_usd.get(provider, 0)
        return self.spend_usd.get(provider, 0) / limit if limit else 0.0

    def exceeded(self, provider: str) -> bool:
        limit = self.limits_usd.get(provider)
        return limit is not None and self.spend_usd.get(provider, 0) >= limit

    def total_fraction(self) -> float:
        total_limit = sum(self.limits_usd.values())
        return sum(self.spend_usd.values()) / total_limit if total_limit else 0.0


class Decision(BaseModel):
    route: Route
    model: ModelAlias
    queue: Queue
    risky: bool
    private: bool = False
    task_type: str | None = None
    confirm_task: list[str] = []    # ≥2 task types to ask the owner to choose between (low router confidence)
    reasons: list[str]
