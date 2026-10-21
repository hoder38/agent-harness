"""LiteLLM pre-call hook: kill switch backstop.

Blocks every non-local model call when Redis `killswitch` is `cloud` or `all`.
The harness enforces the same rule upstream; this guards against bugs.
"""
import os

import redis.asyncio as aioredis
from litellm.integrations.custom_logger import CustomLogger

LOCAL_ALIASES = {"local", "local_cpu"}


class KillSwitchHook(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._redis = aioredis.from_url(os.environ["REDIS_URL"], decode_responses=True)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        model = data.get("model")
        if model in LOCAL_ALIASES:
            return data
        try:
            state = await self._redis.get("killswitch")
        except Exception:
            state = "cloud"  # fail closed for cloud models if Redis is unreachable
        if state in ("cloud", "all"):
            raise Exception(f"killswitch={state}: cloud model '{model}' blocked")
        return data


proxy_handler_instance = KillSwitchHook()
