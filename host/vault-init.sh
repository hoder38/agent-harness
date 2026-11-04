#!/usr/bin/env bash
# One-time vault setup (§10a). Creates the Obsidian vault folder layout, ignore files and its git repo.
# Safe to re-run: existing files are kept. Run as hoder BEFORE `docker compose up` (the bind mounts of
# Inbox/ and AI/ must exist, or Docker creates them owned by root).
#   host/vault-init.sh [/home/hoder/vault]
set -euo pipefail
V=${1:-${VAULT_DIR:-/home/hoder/vault}}

mkdir -p "$V"/{Inbox,Notes,Notion,Projects,Areas,Resources,Archive,Attachments,AI/Daily}
cd "$V"

add_line() { grep -qxF -- "$2" "$1" 2>/dev/null || echo "$2" >> "$1"; }

# git: history of the vault (nightly commit, host/vault-commit.sh); device-local Obsidian state stays out
for l in '.obsidian/workspace*' '.obsidian/cache' '.trash/' '.stversions/' '.stfolder/' '.~harness-*' \
         '*.sync-conflict-*' '.DS_Store'; do add_line .gitignore "$l"; done
# Syncthing: never sync the git repo, device-local UI state, or the harness's atomic-write temp files
for l in '.git' '.obsidian/workspace*' '.obsidian/cache' '.trash' '.~harness-*' '.DS_Store'; do
  add_line .stignore "$l"
done

[[ -f Inbox/README.md ]] || cat > Inbox/README.md <<'MD'
---
privacy: normal
tags: [harness]
---
# Inbox

Captured notes land here (`/note`, the assistant, Copilot). Move them where they belong.
The harness only ever writes in `Inbox/` and `AI/`; edits to other notes need approval in the harness Inbox.
MD

if [[ ! -d .git ]]; then
  git init -q -b main
  git config user.name "${GIT_NAME:-vault}"
  git config user.email "${GIT_EMAIL:-vault@localhost}"
fi
git add -A
git diff --cached --quiet || git commit -q -m "vault: init"
echo "vault ready: $V"
