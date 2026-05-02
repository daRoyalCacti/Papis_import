"""Disk-backed JSON cache and cache-validity helper."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class Cache:
    """Simple SHA-1-keyed JSON cache stored under *root*."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha1(key.encode()).hexdigest()
        ns = self.root / namespace
        ns.mkdir(parents=True, exist_ok=True)
        return ns / f"{digest}.json"

    def load(self, namespace: str, key: str) -> Any | None:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save(self, namespace: str, key: str, data: Any) -> None:
        self._path(namespace, key).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def clear_errors(self) -> int:
        """Delete cached entries that look like errors (network failures,
        rate-limit responses, empty XML).  Returns the number of entries removed.
        """
        removed = 0
        for cache_file in self.root.rglob("*.json"):
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                cache_file.unlink(missing_ok=True)
                removed += 1
                continue
            should_delete = False
            if isinstance(data, dict):
                if "_error" in data:
                    should_delete = True
                elif "_text" in data and not data["_text"]:
                    should_delete = True
                elif "_http_error" in data and data["_http_error"] in (429, 500, 502, 503, 504):
                    should_delete = True
            if should_delete:
                cache_file.unlink(missing_ok=True)
                removed += 1
        return removed


def json_is_cacheable(data: Any) -> bool:
    """Return True if a JSON response is worth caching.

    We cache successful responses and permanent failures (404 Not Found).
    We do NOT cache transient failures: network errors, timeouts, or
    rate-limit responses (429).  Those should be retried on the next run.
    """
    if not isinstance(data, dict):
        return True   # Non-dict JSON (e.g. lists) is always valid
    if "_error" in data:
        return False  # Network/timeout error — always retry
    code = data.get("_http_error")
    if code is None:
        return True   # Successful response
    return code not in (429, 500, 502, 503, 504)  # Retry transient HTTP errors
