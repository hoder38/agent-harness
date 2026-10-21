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

    owner_id: str = "owner"
    agent_jwt_secret: str = "change-me"
    jwt_issuer: str = "nodereact"
    jwt_audience: str = "agent-harness"
    harness_token: str = ""          # host-only internal endpoints (backup alerts, ingest)

    line_channel_secret: str = ""
    line_channel_access_token: str = ""
    line_owner_user_id: str = ""
    line_api_base: str = "https://api.line.me"

    docker_host: str = "http://socket-proxy-ro:2375"
    compose_project: str = "agent-harness"

    policy_path: str = "/config/policy.yaml"
    privacy_rules_path: str = "/config/privacy_rules.yaml"

    phoenix_collector_endpoint: str = ""   # e.g. http://phoenix:6006/v1/traces
    phoenix_project: str = "agent-harness"


@lru_cache
def get_settings() -> Settings:
    return Settings()
