"""Source authority tiers (§12a): rules from config/authority.yaml; `unknown` is left for the LLM judge."""
from typing import Any
from urllib.parse import urlsplit

from .. import config

T1, T2, T3, UNKNOWN = "T1_primary", "T2_secondary", "T3_reject", "unknown"
TIER_RANK = {T1: 0, UNKNOWN: 1, T2: 2, T3: 3}
_GITHUB_SITE_PAGES = {"", "pricing", "features", "about", "enterprise", "security", "copilot"}


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


def _matches(host: str, domain: str) -> bool:
    domain = domain.lower().removeprefix("www.")
    return host == domain or host.endswith("." + domain)


def _entity_key(entity: str | None, table: dict[str, Any]) -> str | None:
    if not entity:
        return None
    low = entity.strip().lower()
    for k in table:
        if k.lower() == low:
            return k
    return None


def classify(url: str, entity: str | None = None, rules: dict[str, Any] | None = None) -> tuple[str, str]:
    """(tier, reason) of `url` as a source for a claim about `entity`."""
    rules = config.authority() if rules is None else rules
    host = domain_of(url)
    if not host:
        return T3, "no host"
    for pat in rules.get("reject_patterns") or []:
        p = pat.lower()
        if (host.endswith(p[1:]) if p.startswith("*.") else _matches(host, p)):
            return T3, f"rejected source ({pat})"

    entities: dict[str, list[str]] = rules.get("entity_domains") or {}
    orgs: dict[str, list[str]] = rules.get("github_orgs") or {}
    key = _entity_key(entity, entities) or _entity_key(entity, orgs)

    if host == "github.com":
        org = urlsplit(url).path.strip("/").split("/")[0].lower()
        if key and org in {o.lower() for o in orgs.get(key, [])}:
            return T1, f"official GitHub org of {key}"
        if key == "GitHub" and org in _GITHUB_SITE_PAGES:
            return T1, "official site of GitHub"
        return T2, "GitHub repository not owned by the claim subject"

    if key and any(_matches(host, d) for d in entities.get(key, [])):
        return T1, f"official site of {key}"
    owner = next((e for e, ds in entities.items() if any(_matches(host, d) for d in ds)), None)
    if owner and key:
        return T2, f"official site of {owner}, not of {key}"
    if owner:  # subject not in the rules (e.g. a product name): the LLM judge decides
        return UNKNOWN, f"official site of {owner}; claim subject {entity!r} not in rules"

    if any(_matches(host, d) for d in rules.get("primary_domains") or []):
        return T1, "primary-source domain"
    if any(host.endswith(s.lower()) for s in rules.get("primary_suffixes") or []):
        return T1, "government/standards/institution domain"
    if any(_matches(host, d) for d in rules.get("secondary_domains") or []):
        return T2, "secondary source (news/encyclopedia)"
    return UNKNOWN, "domain not in authority rules"
