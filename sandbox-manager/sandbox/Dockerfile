# Throwaway coding sandbox (§9a): Copilot CLI + git + node + python. Runs as uid 1000 (`node`).
FROM node:22-bookworm-slim

ARG COPILOT_VERSION=1.0.105
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates make g++ python3 python3-pip python3-venv ripgrep procps \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g "@github/copilot@${COPILOT_VERSION}" && npm cache clean --force

ENV COPILOT_AUTO_UPDATE=false NO_COLOR=1 TZ=Asia/Taipei \
    GIT_AUTHOR_NAME=agent-harness GIT_AUTHOR_EMAIL=agent@localhost \
    GIT_COMMITTER_NAME=agent-harness GIT_COMMITTER_EMAIL=agent@localhost
USER node
WORKDIR /workspace
CMD ["copilot", "--version"]
