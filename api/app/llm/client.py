"""All model calls go through LiteLLM with a model alias (§7)."""
import httpx
from langchain_openai import ChatOpenAI

from ..models import LOCAL_MODELS, KillSwitch
from ..settings import get_settings


class CloudBlocked(RuntimeError):
    pass


def make_llm(alias: str, killswitch: KillSwitch, *, timeout_s: float = 300) -> ChatOpenAI:
    if alias not in LOCAL_MODELS and killswitch in ("cloud", "all"):
        raise CloudBlocked(f"killswitch={killswitch}: '{alias}' blocked")
    s = get_settings()
    return ChatOpenAI(
        model=alias,
        base_url=s.litellm_base_url,
        api_key=s.litellm_master_key,
        streaming=True,
        stream_usage=True,
        timeout=timeout_s,
        max_retries=1,
        # LiteLLM is on the internal network; never route it through the egress proxy.
        http_async_client=httpx.AsyncClient(trust_env=False, timeout=timeout_s),
    )
