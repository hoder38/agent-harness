#!/usr/bin/env bash
# Nightly git commit of the vault (§10a): one commit per day with whatever changed (phone, PC, harness).
# Installed as agent-vault-commit.timer (host/systemd). History only; the backup (backup.sh) copies .git too.
set -euo pipefail
V=${VAULT_DIR:-/home/hoder/vault}
cd "$V"
[[ -d .git ]] || { echo "no git repo in $V (run host/vault-init.sh)" >&2; exit 1; }
git add -A
if git diff --cached --quiet; then
  echo "vault: nothing to commit"
  exit 0
fi
n=$(git diff --cached --name-only | wc -l)
git commit -q -m "vault: $(date +%F) ($n files)"
echo "vault: committed $n files"
