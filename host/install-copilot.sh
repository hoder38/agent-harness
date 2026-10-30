#!/usr/bin/env bash
# Phase 6: install the Copilot CLI hooks (session ingest, §9b) and register the harness-memory MCP server (§9c).
# Idempotent. Needs ~/.config/harness/token (written by host/setup.sh) and agent-api on 127.0.0.1:8100.
set -euo pipefail
cd "$(dirname "$0")/.."

home="${COPILOT_HOME:-$HOME/.copilot}"
token_file="$HOME/.config/harness/token"
[[ -s "$token_file" ]] || { echo "missing $token_file — run host/setup.sh first" >&2; exit 1; }

mkdir -p "$home/hooks"
install -m 755 host/copilot-hooks/harness.sh "$home/hooks/harness.sh"
install -m 644 host/copilot-hooks/harness.json "$home/hooks/harness.json"

HOME_DIR="$home" TOKEN_FILE="$token_file" python3 - <<'PY'
import json, os, pathlib

path = pathlib.Path(os.environ["HOME_DIR"]) / "mcp-config.json"
cfg = json.loads(path.read_text()) if path.exists() else {}
cfg.setdefault("mcpServers", {})["harness-memory"] = {
    "type": "http",
    "url": "http://127.0.0.1:8100/mcp",
    "headers": {"Authorization": "Bearer " + pathlib.Path(os.environ["TOKEN_FILE"]).read_text().strip()},
    "tools": ["*"],
}
path.write_text(json.dumps(cfg, indent=2) + "\n")
path.chmod(0o600)
PY

echo "Installed hooks in $home/hooks and harness-memory MCP in $home/mcp-config.json (restart Copilot CLI)."
