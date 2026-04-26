"""Disk-cached HTTP client used by all resolver functions."""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from papis_import.utils import USER_AGENT, eprint


_RESET_DURATION_RE = re.compile(r"([\d.]+)(ms|s|m|h)")


def _parse_reset_duration(value: str) -> float:
    """Parse a Groq-style rate-limit reset duration string to seconds.

    Handles formats like '6m2.5s', '30s', '500ms', '1h20m'.
    Returns 60.0 as a safe fallback if the string is unparseable.
    """
    total = 0.0
    for amount, unit in _RESET_DURATION_RE.findall(value):
        a = float(amount)
        if unit == "h":    total += a * 3600
        elif unit == "m":  total += a * 60
        elif unit == "s":  total += a
        elif unit == "ms": total += a / 1000
    return total if total > 0 else 60.0


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
        """Delete any cached entries that look like errors (network failures,
        rate-limit responses, empty XML).  Returns the number of entries removed.

        Run this after fixing a network issue or after upgrading the script to
        ensure stale failures get retried on the next scan.
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


class HttpClient:
    """Throttled, cached HTTP helper for all external API calls."""

    def __init__(
        self,
        cache: Cache,
        mailto: str = "",
        verbose: bool = False,
    ) -> None:
        self.cache   = cache
        self.mailto  = mailto.strip()   # used for both Crossref & OpenAlex polite pool
        self.verbose = verbose
        self._last: dict[str, float] = {}
        # Maps bucket → (remaining_tokens, reset_at_monotonic).
        # Populated from x-ratelimit-remaining-tokens / x-ratelimit-reset-tokens
        # response headers so we can pace proactively instead of reacting to 429s.
        self._token_budget: dict[str, tuple[int, float]] = {}
        # Cumulative seconds slept in _wait_for_token_budget per bucket.
        # Read and reset with take_pacing_s() after each call site.
        self._pacing_sleep: dict[str, float] = {}

    def _throttle(self, bucket: str, min_interval: float) -> None:
        now   = time.time()
        delay = min_interval - (now - self._last.get(bucket, 0.0))
        if delay > 0:
            time.sleep(delay)
        self._last[bucket] = time.time()

    def _update_token_budget(self, bucket: str, headers: Any) -> None:
        """Record remaining-token count from x-ratelimit-* response headers.

        Safe to call on both successful responses and HTTPError responses —
        the 429 case is especially useful because it tells us exactly when
        the window resets.
        """
        try:
            remaining = headers.get("x-ratelimit-remaining-tokens")
            reset_str  = headers.get("x-ratelimit-reset-tokens")
            if remaining is None:
                return
            reset_s = _parse_reset_duration(reset_str) if reset_str else 60.0
            self._token_budget[bucket] = (int(remaining), time.monotonic() + reset_s)
        except Exception:
            pass

    def _wait_for_token_budget(self, bucket: str, min_remaining: int) -> None:
        """Sleep proactively if the token budget is too low for another API call.

        Called before each attempt in the retry loop so that after a capped
        429-sleep we recheck and wait out any remaining reset time rather than
        immediately firing another request into an exhausted quota.
        """
        entry = self._token_budget.get(bucket)
        if entry is None:
            return
        remaining, reset_at = entry
        if remaining >= min_remaining:
            return
        wait = reset_at - time.monotonic()
        if wait <= 0:
            return
        eprint(
            f"[pacing] {bucket}: {remaining} tokens remaining "
            f"(need ≥{min_remaining}), sleeping {wait:.1f}s for quota reset"
        )
        actual = wait + 1.0  # +1 s buffer so we don't race the window edge
        time.sleep(actual)
        self._pacing_sleep[bucket] = self._pacing_sleep.get(bucket, 0.0) + actual

    def take_pacing_s(self, bucket: str) -> float:
        """Return and reset the accumulated proactive-pacing sleep for *bucket*.

        Call this once after a post_json call site to capture how long that
        call spent waiting on quota, then store the result in the timing
        breakdown before making the next call.
        """
        return self._pacing_sleep.pop(bucket, 0.0)

    def remaining_tokens(self, bucket: str) -> int | None:
        """Return the last-known remaining-token count for *bucket*, or None."""
        entry = self._token_budget.get(bucket)
        return entry[0] if entry is not None else None

    def _agent(self) -> str:
        if self.mailto:
            return f"{USER_AGENT} mailto:{self.mailto}"
        return USER_AGENT

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cache validity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json_is_cacheable(data: Any) -> bool:
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

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_json(
        self,
        url: str,
        namespace: str,
        cache_key: str,
        *,
        headers: dict[str, str] | None = None,
        bucket: str = "default",
        min_interval: float = 0.2,
        max_retries: int = 3,
    ) -> Any | None:
        cached = self.cache.load(namespace, cache_key)
        if cached is not None and self._json_is_cacheable(cached):
            return cached
        req_headers = {"Accept": "application/json", "User-Agent": self._agent()}
        if headers:
            req_headers.update(headers)

        data: Any = None
        for attempt in range(max_retries):
            self._throttle(bucket, min_interval)
            req = urllib.request.Request(url, headers=req_headers)
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                break  # success
            except urllib.error.HTTPError as exc:
                retry_after = int(exc.headers.get("Retry-After", 0)) if exc.headers else 0
                if exc.code in (429, 503) and attempt < max_retries - 1:
                    wait = max(retry_after, 2 ** (attempt + 2))
                    if self.verbose:
                        eprint(f"[retry] GET {url} HTTP {exc.code}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                data = {"_http_error": exc.code, "_url": url}
                if self.verbose:
                    eprint(f"[warn] GET {url} HTTP {exc.code}")
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    if self.verbose:
                        eprint(f"[retry] GET failed {url}: {exc}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] GET failed {url}: {exc}")
                return None

        if data is not None and self._json_is_cacheable(data):
            self.cache.save(namespace, cache_key, data)
        return data

    def get_xml(
        self,
        url: str,
        namespace: str,
        cache_key: str,
        *,
        bucket: str = "default",
        min_interval: float = 0.2,
        max_retries: int = 3,
    ) -> str:
        cached = self.cache.load(namespace, cache_key)
        if isinstance(cached, dict) and "_text" in cached and cached["_text"]:
            return str(cached["_text"])  # Only return non-empty cached text

        raw = ""
        for attempt in range(max_retries):
            self._throttle(bucket, min_interval)
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.8",
                    "User-Agent": self._agent(),
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                break  # success
            except urllib.error.HTTPError as exc:
                retry_after = int(exc.headers.get("Retry-After", 0)) if exc.headers else 0
                if exc.code in (429, 503) and attempt < max_retries - 1:
                    wait = max(retry_after, 2 ** (attempt + 2))
                    if self.verbose:
                        eprint(f"[retry] GET xml {url} HTTP {exc.code}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] GET xml {url} HTTP {exc.code}")
                return ""
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    if self.verbose:
                        eprint(f"[retry] GET xml failed {url}: {exc}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] GET xml failed {url}: {exc}")
                return ""

        if raw.strip():
            self.cache.save(namespace, cache_key, {"_text": raw})
        return raw

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        namespace: str,
        cache_key: str,
        *,
        extra_headers: dict[str, str] | None = None,
        bucket: str = "default",
        min_interval: float = 0.0,
        max_retries: int = 3,
        max_retry_wait: float = 0.0,
        min_remaining_tokens: int = 0,
    ) -> Any | None:
        cached = self.cache.load(namespace, cache_key)
        if cached is not None and self._json_is_cacheable(cached):
            return cached
        body = json.dumps(payload).encode("utf-8")
        hdrs: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self._agent(),
        }
        if extra_headers:
            hdrs.update(extra_headers)

        data: Any = None
        for attempt in range(max_retries):
            self._throttle(bucket, min_interval)
            if min_remaining_tokens > 0:
                self._wait_for_token_budget(bucket, min_remaining_tokens)
            req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    self._update_token_budget(bucket, resp.headers)
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                break  # success
            except urllib.error.HTTPError as exc:
                retry_after = int(exc.headers.get("Retry-After", 0)) if exc.headers else 0
                if exc.headers:
                    self._update_token_budget(bucket, exc.headers)
                if exc.code in (429, 503) and attempt < max_retries - 1:
                    wait = max(retry_after, 2 ** (attempt + 2))
                    if max_retry_wait > 0:
                        wait = min(wait, max_retry_wait)
                    if self.verbose:
                        eprint(f"[retry] POST {url} HTTP {exc.code}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                try:
                    body_text = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body_text = ""
                data = {"_http_error": exc.code, "_url": url, "_body": body_text}
                if self.verbose:
                    eprint(f"[warn] POST {url} HTTP {exc.code}: {body_text[:200]}")
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    if self.verbose:
                        eprint(f"[retry] POST failed {url}: {exc}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] POST failed {url}: {exc}")
                return None

        if data is not None and self._json_is_cacheable(data):
            self.cache.save(namespace, cache_key, data)
        return data

    def post_multipart(
        self,
        url: str,
        fields: dict[str, str],
        file_field: str,
        file_path: Path,
        namespace: str,
        cache_key: str,
        *,
        bucket: str = "default",
        min_interval: float = 0.0,
        max_retries: int = 3,
    ) -> str:
        """Upload a PDF file to a multipart endpoint (used by GROBID)."""
        cached = self.cache.load(namespace, cache_key)
        if isinstance(cached, dict) and "_text" in cached and cached["_text"]:
            return str(cached["_text"])
        boundary = f"----papis-{hashlib.md5(str(file_path).encode()).hexdigest()}"
        body = bytearray()
        for k, v in fields.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
            body.extend(v.encode("utf-8") + b"\r\n")
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode()
        )
        body.extend(b"Content-Type: application/pdf\r\n\r\n")
        body.extend(file_path.read_bytes())
        body.extend(f"\r\n--{boundary}--\r\n".encode())

        raw = ""
        for attempt in range(max_retries):
            self._throttle(bucket, min_interval)
            req = urllib.request.Request(
                url,
                data=bytes(body),
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Accept": "application/xml, text/xml;q=0.9, */*;q=0.8",
                    "User-Agent": self._agent(),
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                break
            except urllib.error.HTTPError as exc:
                retry_after = int(exc.headers.get("Retry-After", 0)) if exc.headers else 0
                if exc.code in (429, 503) and attempt < max_retries - 1:
                    wait = max(retry_after, 2 ** (attempt + 1))
                    if self.verbose:
                        eprint(f"[retry] multipart POST {url} HTTP {exc.code}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] multipart POST {url} HTTP {exc.code}")
                raw = ""
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    if self.verbose:
                        eprint(f"[retry] multipart POST failed {url}: {exc}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                raw = ""
                if self.verbose:
                    eprint(f"[warn] multipart POST failed {url}: {exc}")
                break
        if raw.strip():
            self.cache.save(namespace, cache_key, {"_text": raw})
        return raw
