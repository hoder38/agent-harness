"""vault-indexer service (§10a): full vault scan at start, then re-index files as they change.

Syncthing writes phone edits into the vault; `watchfiles` (inotify) sees them and the changed notes are re-indexed
after `vault.debounce_ms` of quiet, so an edit is searchable within seconds. The scheduler cron still runs a full
scan every `rag.index_interval_min` (missed events, attachments whose extraction finished later).

    docker compose up -d vault-indexer        # command: python -m app.rag.watcher
"""
import asyncio
import logging
from pathlib import Path

from watchfiles import Change, awatch

from .. import config, db, state
from . import vault

log = logging.getLogger(__name__)
HEARTBEAT_KEY = "vault_indexer:heartbeat"
_IGNORE = (".~harness", ".syncthing.", "~syncthing~", ".stversions", ".stfolder", ".git", ".obsidian", ".trash")


def _rel(base: Path, path: str) -> str | None:
    try:
        rel = Path(path).relative_to(base).as_posix()
    except ValueError:
        return None
    if any(seg.startswith(_IGNORE) or seg.startswith(".") for seg in rel.split("/")):
        return None   # temp files (Syncthing, our atomic writes) and hidden folders
    if vault.is_excluded(rel):
        return None   # vault.exclude_paths: no re-index (or full-scan fallback) while e.g. an import lands there
    return rel


async def _heartbeat() -> None:
    r = state.get_redis()
    while True:
        try:
            await r.set(HEARTBEAT_KEY, "1", ex=90)
        except Exception as e:
            log.warning("heartbeat: %s", e)
        await asyncio.sleep(30)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    await db.open_db()
    hb = asyncio.create_task(_heartbeat())
    base = vault.root()
    try:
        while not base.is_dir():
            log.warning("%s is not mounted yet; retrying", base)
            await asyncio.sleep(30)
        if (config.policy().get("vault") or {}).get("enabled"):
            log.info("initial vault scan: %s", await vault.index_vault(base, force_links=True, wait_s=1800))
        debounce = int((config.policy().get("vault") or {}).get("debounce_ms", 2000))
        async for changes in awatch(base, debounce=debounce, step=200, recursive=True, force_polling=False):
            if not (config.policy().get("vault") or {}).get("enabled"):
                continue
            rels = {rel for ch, p in changes if (rel := _rel(base, p))}
            if not rels:
                continue
            deleted_dirs = any(ch == Change.deleted and not vault.is_note_path(r) and "." not in r.rsplit("/", 1)[-1]
                               for ch, p in changes if (r := _rel(base, p)))
            try:
                if deleted_dirs:
                    await vault.index_vault(base, wait_s=600)
                else:
                    await vault.index_changed(rels, base)
            except Exception:
                log.exception("vault re-index failed for %s", sorted(rels)[:10])
    finally:
        hb.cancel()
        await db.close_db()


if __name__ == "__main__":
    asyncio.run(main())
