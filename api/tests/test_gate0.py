from app.gates.gate0 import gate0
from app.models import Job


def job(text, **kw):
    return Job(**{"channel": "web", "user_id": "owner", "thread_id": "web:t", "text": text, "dedupe_key": "k", **kw})


RULES = {"private_sources": ["mail", "finance"]}


def test_override_prefix_is_parsed_and_stripped():
    r = gate0(job("/cloud   explain monads"), RULES)
    assert r.override == "cloud" and r.text == "explain monads"


def test_secrets_and_pii_are_redacted():
    text = ("key sk-proj-abcdefghijklmnopqrstuvwx and ghp_" + "a" * 36 +
            " mail me@example.com call 0912-345-678 id A123456789 card 4111 1111 1111 1111")
    r = gate0(job(text), RULES)
    for secret in ("sk-proj-abc", "ghp_aaa", "me@example.com", "0912-345-678", "A123456789", "4111 1111"):
        assert secret not in r.redacted_text
    assert {"openai_key", "github_token", "email", "phone", "tw_national_id", "card_number"} <= set(r.findings)
    assert r.privacy_tag is None  # redaction alone does not make it PRIVATE
    assert "me@example.com" in r.text  # original kept for local use


def test_non_luhn_digits_not_redacted_as_card():
    r = gate0(job("order 1234 5678 9012 3456"), RULES)
    assert "card_number" not in r.findings


def test_private_key_block_redacted():
    r = gate0(job("-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----"), RULES)
    assert "abc" not in r.redacted_text


def test_only_an_owner_marked_thread_is_private():
    src = gate0(job("hello", source="mail"), RULES)
    assert src.privacy_tag is None and src.local_only
    assert gate0(job("hello"), RULES, thread_private=True).privacy_tag == "PRIVATE"
    assert not gate0(job("hello"), RULES, thread_private=True).local_only
    # calendar words are not private any more (§6, 2026-11-05)
    r = gate0(job("幫我規劃東京行程，順便看行事曆"), {"private_keywords": ["行程"]})
    assert r.privacy_tag is None and not r.local_only


def test_extra_private_pattern():
    rules = {"extra_patterns": [{"name": "acct", "regex": r"ACCT-\d{4}", "private": True}]}
    r = gate0(job("pay ACCT-1234"), rules)
    assert r.local_only and r.privacy_tag is None and "ACCT-1234" not in r.redacted_text


def test_line_is_never_private_but_still_redacts():
    rules = {"private_sources": ["line"]}
    j = job("行程 key sk-abcdefghijklmnopqrstuvwx", channel="line", source="line", thread_id="line:u:2")
    r = gate0(j, rules, thread_private=True, channel="line")
    assert r.privacy_tag is None and not r.local_only and "sk-abcdefgh" not in r.redacted_text
