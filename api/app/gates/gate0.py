"""Gate 0 — deterministic privacy/secret gate. Pure code, must stay < 5 ms."""
import re
from dataclasses import dataclass
from typing import Any

from ..models import Gate0Result, Job


@dataclass(frozen=True)
class Pattern:
    name: str
    kind: str  # secret | pii
    regex: re.Pattern[str]
    private: bool = False
    validator: Any = None


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


BUILTIN_PATTERNS: list[Pattern] = [
    Pattern("private_key", "secret", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----|$)")),
    Pattern("aws_access_key", "secret", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    Pattern("github_token", "secret", re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    Pattern("openai_key", "secret", re.compile(r"\bsk-(proj-|ant-)?[A-Za-z0-9_\-]{20,}\b")),
    Pattern("slack_token", "secret", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    Pattern("google_api_key", "secret", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    Pattern("jwt", "secret", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    Pattern("assigned_secret", "secret", re.compile(
        r"(?i)\b(api[_-]?key|secret|token|passw(or)?d|pwd)\b(\s*[:=]\s*|\s+is\s+)['\"]?[^\s'\"]{6,}")),
    Pattern("tw_national_id", "pii", re.compile(r"\b[A-Z][12]\d{8}\b")),
    Pattern("card_number", "pii", re.compile(r"\b(?:\d[ -]?){13,19}\b"), validator=_luhn),
    Pattern("email", "pii", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    Pattern("phone", "pii", re.compile(r"(?<![\d-])(\+?886[- ]?9\d{2}|09\d{2})[- ]?\d{3}[- ]?\d{3}(?![\d-])|\+\d{1,3}[- ]?\(?\d{1,4}\)?[- ]?\d{3,4}[- ]?\d{3,4}\b")),
]

_OVERRIDE_RE = re.compile(r"^\s*/(local|cloud|code)\b\s*", re.IGNORECASE)


def _compile_extra(rules: dict[str, Any]) -> list[Pattern]:
    out = []
    for p in rules.get("extra_patterns") or []:
        out.append(Pattern(p["name"], p.get("kind", "pii"), re.compile(p["regex"]), bool(p.get("private", False))))
    return out


def redact(text: str, rules: dict[str, Any] | None = None) -> tuple[str, list[str], list[str]]:
    """(redacted text, finding names, names of matched `private` patterns)."""
    findings: list[str] = []
    private_hits: list[str] = []
    redacted = text
    for pat in BUILTIN_PATTERNS + _compile_extra(rules or {}):
        def _sub(mm: re.Match[str], pat: Pattern = pat) -> str:
            if pat.validator and not pat.validator(mm.group(0)):
                return mm.group(0)
            if pat.name not in findings:
                findings.append(pat.name)
            return f"[REDACTED:{pat.name}]"

        new = pat.regex.sub(_sub, redacted)
        if new != redacted and pat.private:
            private_hits.append(pat.name)
        redacted = new
    return redacted, findings, private_hits


def gate0(job: Job, rules: dict[str, Any] | None = None, *, thread_private: bool = False, channel: str | None = None) -> Gate0Result:
    rules = rules or {}
    text = job.text or ""
    reasons: list[str] = []

    override = job.override
    m = _OVERRIDE_RE.match(text)
    if m:
        override = m.group(1).lower()  # type: ignore[assignment]
        text = text[m.end():]
    if override:
        reasons.append(f"override=/{override}")

    redacted, findings, private_hits = redact(text, rules)
    private = bool(private_hits)
    reasons += [f"pattern {n} marks PRIVATE" for n in private_hits]
    if findings:
        reasons.append("redacted: " + ",".join(findings))

    # LINE has no private tools or content, so only secret patterns lock it.
    topical = channel != "line"
    private_sources = set(rules.get("private_sources") or [])
    if topical and job.source and job.source in private_sources:
        private = True
        reasons.append(f"source={job.source} is PRIVATE")

    lowered = text.lower()
    for kw in (rules.get("private_keywords") or []) if topical else []:
        if kw.lower() in lowered:
            private = True
            reasons.append(f"keyword '{kw}' is PRIVATE")
            break

    if thread_private:
        private = True
        reasons.append("thread already contains PRIVATE content")

    if channel == "line":
        private = False  # LINE is never PRIVATE; secrets are still redacted above

    return Gate0Result(
        text=text,
        redacted_text=redacted,
        override=override,
        findings=findings,
        privacy_tag="PRIVATE" if private else None,
        reasons=reasons,
    )
