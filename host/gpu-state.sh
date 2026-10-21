#!/usr/bin/env bash
# Manually set gpu_state until gpu-governor (phase 4) exists:  host/gpu-state.sh FREE|BUSY|GAMING
set -euo pipefail
cd "$(dirname "$0")/.."
pw=$(grep '^REDIS_PASSWORD=' .env | cut -d= -f2-)
rc() { docker compose exec -T agent-redis redis-cli --no-auth-warning -a "$pw" "$@"; }
case "${1:-}" in
  FREE|BUSY|GAMING)
    rc SET gpu_state "$1" >/dev/null
    rc PUBLISH gpu_state_changed "$1" >/dev/null
    if [[ "$1" == GAMING ]]; then  # release VRAM now
      model=$(grep '^LOCAL_MODEL=' .env | cut -d= -f2- | sed 's|^ollama_chat/||')
      docker compose exec -T ollama ollama stop "$model" || true
    fi
    echo "gpu_state=$1" ;;
  "") echo "gpu_state=$(rc GET gpu_state)" ;;
  *) echo "usage: $0 [FREE|BUSY|GAMING]" >&2; exit 2 ;;
esac
