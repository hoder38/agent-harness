from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    database_url: str = "postgresql://agent:agent@agent-postgres:5432/agent"
    redis_url: str = "redis://agent-redis:6379/0"

    litellm_base_url: str = "http://litellm:4000/v1"
    litellm_master_key: str = "sk-change-me"

    typesafe_api_key: str = ""
    jev_model: str = "jev-latest"

    brave_api_key: str = ""
    fetch_proxy: str = "http://egress-proxy:4751"   # open-egress squid port (private IPs blocked) for fetch_page
    browser_url: str = "http://browser:3000"         # §12b headless Chromium renderer (read-only fallback)

    owner_id: str = "owner"
    agent_jwt_secret: str = "change-me"
    jwt_issuer: str = "nodereact"
    jwt_audience: str = "agent-harness"
    event_secrets: dict[str, str] = {}   # phase 11: EVENT_SECRETS='{"source": "hmac secret"}'
    harness_token: str = ""          # host-only internal endpoints (backup alerts, ingest)

    line_channel_secret: str = ""
    line_channel_access_token: str = ""
    line_owner_user_id: str = ""
    line_api_base: str = "https://api.line.me"
    line_data_api_base: str = "https://api-data.line.me"   # message content (images, audio, files)

    telegram_bot_token: str = ""                       # §19 proactive notices (send-only)
    telegram_owner_chat_id: str = ""
    telegram_api_base: str = "https://api.telegram.org"

    docker_host: str = "http://socket-proxy-ro:2375"
    ollama_url: str = "http://ollama:11434"            # gpu-governor: /api/ps, unload on GAMING; RAG embeddings
    reranker_url: str = ""                             # §10 CPU cross-encoder; empty = RRF order only
    google_credentials_path: str = "/google/calendar.json"   # host-only OAuth grant (host/google_auth.py), mounted ro
    docs_root: str = "/docs"                           # §10 read-only docs folder indexed for RAG
    vault_root: str = "/vault"                         # §10a Obsidian vault, read-only (indexing, read_note)
    vault_rw_root: str = "/vault-rw"                   # only <vault>/Inbox and <vault>/AI, read-write (agent writes)
    vault_edit_root: str = "/vault-edit"               # whole vault read-write: agent-api only (approved edits)
    files_root: str = "/files"                         # §5 uploads: /srv/agent/files/<sha[:2]>/<sha> on the host
    copilot_sessions_dir: str = "/copilot-sessions"    # §9b host ~/.copilot/session-state, read-only
    compose_project: str = "agent-harness"

    policy_path: str = "/config/policy.yaml"
    privacy_rules_path: str = "/config/privacy_rules.yaml"
    authority_path: str = "/config/authority.yaml"
    search_sites_path: str = "/config/search_sites.yaml"

    phoenix_collector_endpoint: str = ""   # e.g. http://phoenix:6006/v1/traces
    phoenix_project: str = "agent-harness"


@lru_cache
def get_settings() -> Settings:
    return Settings()
