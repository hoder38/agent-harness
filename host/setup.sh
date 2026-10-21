#!/usr/bin/env bash
# One-time setup: create .env with random secrets, the host harness token, and pull the local model.
set -euo pipefail
cd "$(dirname "$0")/.."

rand() { head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 40; }

if [[ ! -f .env ]]; then
  cp .env.example .env
  for k in POSTGRES_PASSWORD REDIS_PASSWORD AGENT_JWT_SECRET HARNESS_TOKEN; do
    sed -i -E "s|^${k}=.*$|${k}=$(rand)|" .env
  done
  sed -i -E "s|^LITELLM_MASTER_KEY=.*$|LITELLM_MASTER_KEY=sk-$(rand)|" .env
  sed -i -E 's/[[:space:]]+#.*$//' .env   # strip inline comments
  chmod 600 .env
  echo "Created .env — fill in TYPESAFE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, LINE_* and copy AGENT_JWT_SECRET to nodeReact."
fi

mkdir -p ~/.config/harness
grep '^HARNESS_TOKEN=' .env | cut -d= -f2- > ~/.config/harness/token
chmod 600 ~/.config/harness/token

docker compose up -d --build
model=$(grep '^LOCAL_MODEL=' .env | cut -d= -f2- | sed 's|^ollama_chat/||')
echo "Pulling ${model} (via egress proxy)…"
docker compose exec -T ollama ollama pull "${model}"
docker compose ps
