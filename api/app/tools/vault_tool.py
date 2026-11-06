"""Vault maintenance (§10a).

    docker compose exec agent-api python -m app.tools.vault_tool links [--folder Notion] [--all]
    docker compose exec agent-api python -m app.tools.vault_tool notion-fix [--folder Notion] [--strip-ids] [--dry-run]
    docker compose exec agent-api python -m app.tools.vault_tool index [--force-links]
    docker compose exec agent-api python -m app.tools.vault_tool daily [YYYY-MM-DD|yesterday]

`links`: notes, links and every link/embed whose target does not exist (exit code 1 if any; `--all` also lists
  wikilinks to notes not written yet, which Obsidian allows on purpose).
`notion-fix` (after the Obsidian Importer, on the read-write mount of agent-api): relative markdown links → wikilinks,
  frontmatter `source: notion` (+ `notion_id` from a "Title <32 hex>.md" name), `--strip-ids` renames those files to
  "Title.md" (when free) and rewrites the links to them. Then re-run `links`.
"""
import argparse
import asyncio
import json
import posixpath
import re
import sys
from pathlib import Path
from typing import Any

from ..rag import vault
from ..settings import get_settings

_NOTION_ID = re.compile(r"^(?P<title>.*?)\s+(?P<id>[0-9a-f]{32})$")


def _in(rel: str, folder: str) -> bool:
    return not folder or rel == folder or rel.startswith(folder.strip("/") + "/")


def report(base: Path, folder: str = "", *, all_unresolved: bool = False) -> dict[str, Any]:
    files = vault.scan_all(base, include_excluded=True)
    res = vault.Resolver(files)
    notes = [r for r in files if vault.is_note_path(r) and _in(r, folder)]
    broken, links = [], 0
    for rel in notes:
        note = vault.parse(files[rel].read_text(encoding="utf-8", errors="replace"))
        for l in note.links:
            links += 1
            if res.resolve(l, rel):
                continue
            # a wikilink to a note that does not exist yet is normal in Obsidian; files and markdown links are not
            if all_unresolved or l.md or l.is_file or l.kind == "embed":
                broken.append({"note": rel, "kind": l.kind, "target": l.target, "markdown": l.md})
    return {"notes": len(notes), "files": len(files), "links": links, "broken": broken}


def wikilink(path: str, *, frag: str = "", alias: str = "", embed: bool = False) -> str:
    key = re.sub(r"\.(md|markdown)$", "", path, flags=re.I) if vault.is_note_path(path) else path
    name = key.rsplit("/", 1)[-1]
    a = alias if alias and alias != name and not embed else ""
    return f"{'!' if embed else ''}[[{key}{'#' + frag if frag else ''}{'|' + a if a else ''}]]"


def rewrite_links(text: str, src: str, resolver: vault.Resolver, renames: dict[str, str]) -> tuple[str, int]:
    """Markdown links that resolve inside the vault → wikilinks; links to renamed files → their new path."""
    meta_end = len(text) - len(vault.chunking.split_frontmatter(text)[1])
    body = text[meta_end:]
    out, pos, n = [], 0, 0
    for l in vault.parse_links(vault.strip_code(body)):
        target = resolver.resolve(l, src)
        if not target or (not l.md and target not in renames):
            continue
        new = renames.get(target, target)
        a, b = l.span
        out += [body[pos:a], wikilink(new, frag=l.frag, alias=l.alias, embed=l.kind == "embed")]
        pos, n = b, n + 1
    out.append(body[pos:])
    return text[:meta_end] + "".join(out), n


def notion_meta(text: str, notion_id: str | None) -> str:
    from ..vault_writer import set_meta

    meta, _ = vault.chunking.split_frontmatter(text)
    upd: dict[str, Any] = {}
    if not meta.get("source"):
        upd["source"] = "notion"
    if notion_id and not meta.get("notion_id"):
        upd["notion_id"] = notion_id
    return set_meta(text, **upd) if upd else text


def notion_fix(base: Path, folder: str = "Notion", *, strip_ids: bool = False, dry_run: bool = False) -> dict[str, Any]:
    from ..vault_writer import atomic_write

    files = vault.scan_all(base, include_excluded=True)
    resolver = vault.Resolver(files)
    scope = [r for r in files if _in(r, folder)]
    renames: dict[str, str] = {}
    ids: dict[str, str] = {}
    taken = set(files)
    for rel in scope:
        stem, ext = posixpath.splitext(rel)
        m = _NOTION_ID.match(stem.rsplit("/", 1)[-1])
        if not m:
            continue
        if vault.is_note_path(rel):
            ids[rel] = m.group("id")
        if strip_ids:
            new = posixpath.join(posixpath.dirname(rel), m.group("title").strip() + ext)
            if new not in taken:
                renames[rel] = new
                taken.add(new)
    stats = {"notes": 0, "links_rewritten": 0, "meta_added": 0, "renamed": len(renames), "dry_run": dry_run}
    for rel in [r for r in scope if vault.is_note_path(r)]:
        p = files[rel]
        before = p.read_text(encoding="utf-8", errors="replace")
        text, n = rewrite_links(before, rel, resolver, renames)
        text2 = notion_meta(text, ids.get(rel))
        stats["notes"] += 1
        stats["links_rewritten"] += n
        stats["meta_added"] += text2 != text
        if text2 != before and not dry_run:
            atomic_write(p, text2)
    # notes outside the folder may link to renamed files too
    if renames:
        for rel in [r for r in files if vault.is_note_path(r) and not _in(r, folder)]:
            before = files[rel].read_text(encoding="utf-8", errors="replace")
            text, n = rewrite_links(before, rel, resolver, renames)
            if n:
                stats["links_rewritten"] += n
                if not dry_run:
                    atomic_write(files[rel], text)
    if not dry_run:
        for old, new in renames.items():
            (base / old).rename(base / new)
    return stats


async def _with_db(coro_fn):
    from .. import db

    await db.open_db()
    try:
        return await coro_fn()
    finally:
        await db.close_db()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vault_tool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("links")
    a.add_argument("--folder", default="")
    a.add_argument("--all", action="store_true")
    a.add_argument("--root", default="")
    a = sub.add_parser("notion-fix")
    a.add_argument("--folder", default="Notion")
    a.add_argument("--strip-ids", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--root", default="")
    a = sub.add_parser("index")
    a.add_argument("--force-links", action="store_true")
    a = sub.add_parser("daily")
    a.add_argument("date", nargs="?", default="")
    args = ap.parse_args(argv)
    s = get_settings()
    if args.cmd == "links":
        r = report(Path(args.root or s.vault_root), args.folder, all_unresolved=args.all)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 1 if r["broken"] else 0
    if args.cmd == "notion-fix":
        base = Path(args.root or s.vault_edit_root)
        if not base.is_dir():
            print(f"{base} is not mounted (run in agent-api, which mounts the vault read-write)", file=sys.stderr)
            return 2
        print(json.dumps(notion_fix(base, args.folder, strip_ids=args.strip_ids, dry_run=args.dry_run), indent=2))
        print(json.dumps(report(base, args.folder), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "index":
        print(json.dumps(asyncio.run(_with_db(lambda: vault.index_vault(force_links=args.force_links, wait_s=600))),
                         ensure_ascii=False, indent=2, default=str))
        return 0
    from .. import vault_daily

    print(asyncio.run(_with_db(lambda: vault_daily.run_for(args.date) if args.date else vault_daily.run())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
