"""Hot-reloadable YAML config (policy.yaml, privacy_rules.yaml)."""
import os
import threading
from typing import Any

import yaml

from .settings import get_settings

_lock = threading.Lock()
_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def load_yaml(path: str) -> dict[str, Any]:
    mtime = os.path.getmtime(path)
    with _lock:
        hit = _cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    with _lock:
        _cache[path] = (mtime, data)
    return data


def policy() -> dict[str, Any]:
    return load_yaml(get_settings().policy_path)


def privacy_rules() -> dict[str, Any]:
    return load_yaml(get_settings().privacy_rules_path)


def authority() -> dict[str, Any]:
    return load_yaml(get_settings().authority_path)
