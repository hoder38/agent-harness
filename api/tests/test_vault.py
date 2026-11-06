"""Phase 10a: Obsidian vault parsing, link resolution, writer restrictions, daily note, Notion clean-up."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from app import vault_writer as vw
from app.rag import store, vault
from app.settings import get_settings


@pytest.fixture
def vdir(tmp_path, monkeypatch):
    v = tmp_path / "vault"
    for d in ("Inbox", "AI", "Notes", "Notion"):
        (v / d).mkdir(parents=True)
    s = get_settings()
    for k in ("vault_root", "vault_rw_root", "vault_edit_root"):
        monkeypatch.setattr(s, k, str(v))
    return v


def test_parse_links_tags_aliases_and_ignores_code():
    raw = """---
tags: [proj, "area/home"]
aliases: Harness
---
# Plan #headtag
See [[Docker Setup#Compose|compose]] and ![[diagram.png]] and [x](../Sub%20Page.md#Intro).
Web [site](https://example.com) and [[#Local heading]].
Tagged #todo/later and #42 and a#b.
```
[[Not A Link]] #nottag
```
`[[inline code]]`
"""
    n = vault.parse(raw)
    got = [(l.kind, l.target, l.frag, l.alias, l.md) for l in n.links]
    assert got == [("link", "Docker Setup", "Compose", "compose", False), ("embed", "diagram.png", "", "", False),
                   ("link", "../Sub Page.md", "Intro", "x", True)]
    assert n.tags == ["proj", "area/home", "headtag", "todo/later"]
    assert n.aliases == ["Harness"]
    assert vault.display(n.body, n.links).splitlines()[1] == "See compose and 📎 diagram.png and x."


def test_resolver_obsidian_rules():
    r = vault.Resolver(["Notes/Docker Setup.md", "Archive/Old/Docker Setup.md", "Projects/A/Plan.md",
                        "Projects/B/Plan.md", "Attachments/diagram.png", "Notion/Sub Page.md", "Notion/x/y.md"])
    L = lambda t, md=False: vault.Link("link", t, md=md)
    assert r.resolve(L("docker setup"), "x.md") == "Notes/Docker Setup.md"         # shallowest wins
    assert r.resolve(L("B/Plan"), "x.md") == "Projects/B/Plan.md"                   # path suffix
    assert r.resolve(L("diagram.png"), "x.md") == "Attachments/diagram.png"
    assert r.resolve(L("../Sub Page.md", md=True), "Notion/x/y.md") == "Notion/Sub Page.md"
    assert r.resolve(L("Missing"), "x.md") is None
    assert r.key(L("Missing"), "x.md") == "missing"
    assert r.key(L("B/Plan"), "x.md") == "projects/b/plan"


def test_vault_cite_label_and_private_paths():
    assert store.cite_label({"source": "vault", "path": "Notes/Docker.md", "heading": "Setup › Compose"}) \
        == "[[Notes/Docker#Compose]]"
    assert store.cite_label({"source": "vault", "path": "N.md", "heading": "N › 📎 scan.pdf"}) == "[[scan.pdf]]"
    assert store.cite_label({"source": "docs", "path": "a.md", "heading": "H"}) == "a.md#H"
    rules = {"private_paths": ["Areas/Health"]}
    assert store.doc_privacy("Areas/Health/2026.md", {}, "x", rules)[0] == "PRIVATE"
    assert store.doc_privacy("Areas/Healthy.md", {}, "x", rules)[0] == "NORMAL"


def test_writer_only_inbox_and_ai(vdir):
    assert vw.writable_path("Inbox/a").name == "a.md"
    assert vw.writable_path("AI/Daily/x.md").parent.name == "Daily"
    for bad in ("Notes/a.md", "Inbox", "../Inbox/a.md", "Inbox/../Notes/a.md", "Inbox/.obsidian/a.md", "AI.md"):
        with pytest.raises(vw.VaultWriteError):
            vw.writable_path(bad)
    (vdir / "Inbox" / "link").symlink_to(vdir / "Notes")
    with pytest.raises(vw.VaultWriteError):
        vw.writable_path("Inbox/link/a.md")


def test_capture_and_daily_note(vdir):
    when = datetime(2026, 11, 2, 9, 5, tzinfo=vw.tz())
    rel = vw.capture("買濾心\n第二行", private=False, when=when)
    assert rel == "Inbox/2026-11-02 0905 買濾心.md"
    assert vw.capture("買濾心", when=when).endswith(" 2.md")
    meta, body = vault.chunking.split_frontmatter((vdir / rel).read_text())
    assert meta["privacy"] == "normal" and meta["tags"] == ["inbox"] and body.strip().startswith("買濾心")
    secret = vw.capture("note about token ghp_" + "a" * 36, when=when)
    assert "privacy: private" in (vdir / secret).read_text()

    vw.append_daily("first", when=when)
    p = vdir / "AI/Daily/2026-11-02.md"
    p.write_text(p.read_text() + "\n## Owner\nmine\n")
    vw.write_daily_summary(when, "## Summary\nday went well")
    vw.append_daily("second", when=when.replace(hour=10))
    vw.write_daily_summary(when, "## Summary\nupdated")
    text = p.read_text()
    assert "privacy: normal" in text and "day went well" not in text and text.count(vw.SUMMARY_START) == 1
    log = text.split("## Log", 1)[1]
    assert log.index("09:05 first") < log.index("10:05 second") < log.index("## Owner")
    assert "mine" in text and "updated" in text


async def test_edit_outside_writable_folders_needs_approval(vdir, monkeypatch):
    from app import inbox

    (vdir / "Notes" / "n.md").write_text("hello world\n")
    added = []

    async def add(kind, title, body="", **kw):
        added.append((kind, title, body, kw))
        return 7

    async def list_(*a, **kw):
        return []

    monkeypatch.setattr(inbox, "add", add)
    monkeypatch.setattr(inbox, "list_", list_)
    out = await vw.propose_edit("Notes/n", "world", "vault")
    assert "NOT been changed" in out and (vdir / "Notes" / "n.md").read_text() == "hello world\n"
    kind, _, body, kw = added[0]
    assert kind == "approval" and kw["private"] and "-hello world" in body and "+hello vault" in body
    detail = kw["detail"]
    # applied only on approval, and only if the note did not change meanwhile
    (vdir / "Notes" / "n.md").write_text("hello world!\n")
    with pytest.raises(vw.VaultWriteError, match="changed"):
        vw.apply_approved(detail)
    (vdir / "Notes" / "n.md").write_text("hello world\n")
    assert vw.apply_approved(detail) == "applied"
    assert (vdir / "Notes" / "n.md").read_text() == "hello vault\n"
    # Inbox/ and AI/ change at once
    (vdir / "Inbox" / "i.md").write_text("a b\n")
    assert (await vw.propose_edit("Inbox/i.md", "b", "c")).startswith("Saved")
    assert (vdir / "Inbox" / "i.md").read_text() == "a c\n" and len(added) == 1


async def test_read_note_hides_private_from_cloud(vdir, monkeypatch):
    async def tag(source, path):
        return None

    monkeypatch.setattr(store, "doc_privacy_tag", tag)
    (vdir / "Notes" / "pub.md").write_text("# Pub\nmail me at someone@example.com\n")
    (vdir / "Notes" / "sec.md").write_text("---\nprivacy: private\n---\nsecret\n")
    text, err = await vault.read_note("pub", private=False)
    assert not err and "someone@example.com" not in text and text.startswith("# Notes/pub.md")
    assert (await vault.read_note("Notes/sec.md", private=False))[1] == "this note is private"
    assert "secret" in (await vault.read_note("sec", private=True))[0]
    assert (await vault.read_note("../etc/passwd", private=True))[1] == "invalid path"


async def test_exclude_paths_hide_folder_from_agents(vdir, monkeypatch):
    from app.rag import watcher

    monkeypatch.setattr(vault, "cfg", lambda: {"exclude_paths": ["notion"]})
    (vdir / "Notion" / "sub").mkdir()
    (vdir / "Notion" / "sub" / "raw.md").write_text("raw import\n")
    (vdir / "Notion" / "top.md").write_text("raw\n")
    (vdir / "Notes" / "Notion-ish.md").write_text("kept\n")
    assert sorted(vault.scan_all(vdir)) == ["Notes/Notion-ish.md"]
    assert len(vault.scan_all(vdir, include_excluded=True)) == 3
    assert watcher._rel(vdir, str(vdir / "Notion" / "top.md")) is None
    assert (await vault.read_note("Notion/top", private=True))[1] == "no note 'Notion/top'"
    assert (await vault.read_note("raw", private=True))[1] == "no note 'raw'"
    assert (await vault.index_changed({"Notion/top.md"}, vdir))["added"] == 0


def test_notion_fix_rewrites_links_and_strips_ids(vdir):
    from app.tools import vault_tool

    nid = "0123456789abcdef0123456789abcdef"
    (vdir / "Notion" / f"Sub Page {nid}.md").write_text("# Sub\n")
    (vdir / "Notion" / "Home.md").write_text(f"See [the sub](Sub%20Page%20{nid}.md) and [web](https://x.y) "
                                             "and [gone](Missing.md).\n")
    assert len(vault_tool.report(vdir, "Notion")["broken"]) == 1
    st = vault_tool.notion_fix(vdir, "Notion", strip_ids=True)
    assert st["renamed"] == 1 and st["links_rewritten"] == 1
    home = (vdir / "Notion" / "Home.md").read_text()
    assert "[[Notion/Sub Page|the sub]]" in home and "source: notion" in home
    assert f"notion_id: {nid}" in (vdir / "Notion" / "Sub Page.md").read_text()
    assert [b["target"] for b in vault_tool.report(vdir, "Notion")["broken"]] == ["Missing.md"]


def test_daily_summary_build():
    from app import vault_daily

    day = datetime(2026, 11, 2, 23, 50, tzinfo=vw.tz())
    md = vault_daily.build(day, episodes=[{"text": "[2026-11-02] Talked about NAS"}], coding=[],
                           done=[{"title": "pay bill"}], captured=["Inbox/2026-11-02 0905 x.md"])
    assert "- Talked about NAS" in md and "- [x] pay bill" in md and "[[Inbox/2026-11-02 0905 x]]" in md
    assert "Nothing recorded" in vault_daily.build(day, episodes=[], coding=[], done=[], captured=[])


def test_vault_tools_by_route():
    from app.tools import tools_for

    names = lambda **kw: {t.name for t in tools_for(**kw)}
    assert {"capture_note", "append_daily_note", "edit_note", "read_note"} <= names(route="assistant")
    assert "read_note" in names(route="chat") and not {"capture_note", "edit_note"} & names(route="chat")


async def test_note_command(vdir, redis):
    from app import pipeline
    from app.models import Job

    job = Job(channel="line", mode="interactive", user_id="owner", source="line", thread_id="line:u",
              text="/note 買新的濾水器濾心", dedupe_key="d1")
    res = await pipeline.submit(job, SimpleNamespace())
    assert res.command_reply.startswith("📝 Saved: Inbox/")
    assert len(list((vdir / "Inbox").glob("*.md"))) == 1


def test_exact_title_or_alias_keeps_low_scoring_hit():
    cands = [{"path": "a.md", "heading": "", "vec_sim": 0.2, "named": True}, {"path": "b.md", "heading": "", "vec_sim": 0.2}]
    assert [h["path"] for h in store.select_hits(cands, [0.0, 0.0], k=5, min_rerank=0.01, min_vec=0.45)] == ["a.md"]
