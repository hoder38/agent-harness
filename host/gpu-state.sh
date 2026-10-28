#!/usr/bin/env bash
# gpu_state is set by the gpu-governor service (phase 4, §11). This script shows it or pins a manual override:
#   host/gpu-state.sh                      show state, override and the governor's last sample
#   host/gpu-state.sh FREE|BUSY|GAMING     pin (the governor publishes it within poll_s; GAMING unloads Ollama VRAM)
#   host/gpu-state.sh auto                 clear the override (back to measured state)
set -euo pipefail
cd "$(dirname "$0")/.."
pw=$(grep '^REDIS_PASSWORD=' .env | cut -d= -f2-)
rc() { docker compose exec -T agent-redis redis-cli --no-auth-warning -a "$pw" "$@"; }
case "${1:-}" in
  FREE|BUSY|GAMING)
    rc SET gpu_state_override "$1" >/dev/null
    echo "override=$1 (run '$0 auto' to clear)" ;;
  auto)
    rc DEL gpu_state_override >/dev/null
    echo "override cleared" ;;
  "")
    echo "gpu_state=$(rc GET gpu_state) override=$(rc GET gpu_state_override)"
    echo "heartbeat=$(rc GET gpu_governor:heartbeat)" ;;
  *) echo "usage: $0 [FREE|BUSY|GAMING|auto]" >&2; exit 2 ;;
esac
