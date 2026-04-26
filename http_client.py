"""Disk-cached HTTP client used by all resolver functions."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from papis_import.utils import USER_AGENT, eprint


_RESET_DURATION_RE = re.compile(r"([\d.]+)(ms|s|m|h)")
_TRY_AGAIN_RE = re.compile(r"please\s+try\s+again\s+in\s+([0-9a-zA-Z. ]+?)(?:[.,]|$)", re.I)


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
        # Last HTTP trace per bucket.  This is intentionally compact and
        # excludes query strings/headers so API keys are not persisted.
        self._last_request_trace: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _redacted_url(url: str) -> str:
        parts = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    @staticmethod
    def _now_iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())

    @staticmethod
    def _retry_after(headers: Any) -> float:
        if not headers:
            return 0.0
        try:
            return float(headers.get("Retry-After", 0) or 0)
        except Exception:
            return 0.0

    @staticmethod
    def _rate_limit_headers(headers: Any) -> dict[str, str]:
        if not headers:
            return {}
        out: dict[str, str] = {}
        try:
            items = headers.items()
        except Exception:
            items = []
        for key, value in items:
            key_s = str(key)
            if key_s.lower().startswith("x-ratelimit-") and value is not None:
                out[key_s.lower()] = str(value)
        return out

    @staticmethod
    def _safe_body_excerpt(body: str, limit: int = 1000) -> str:
        """Return a compact HTTP error body excerpt suitable for TSV debug."""
        if not body:
            return ""
        # Bodies should not contain auth headers, but redact common key/token
        # spellings defensively before persisting the excerpt.
        redacted = re.sub(
            r'(?i)("?(?:api[_-]?key|authorization|token|secret)"?\s*[:=]\s*")([^"]+)(")',
            r"\1[redacted]\3",
            body,
        )
        redacted = re.sub(r"\s+", " ", redacted).strip()
        return redacted[:limit]

    @staticmethod
    def _parse_rate_limit_body(body: str) -> dict[str, Any]:
        if not body:
            return {}
        message = body
        code = ""
        err_type = ""
        try:
            data = json.loads(body)
            err = data.get("error") if isinstance(data, dict) else None
            if isinstance(err, dict):
                message = str(err.get("message") or message)
                code = str(err.get("code") or "")
                err_type = str(err.get("type") or "")
            elif isinstance(data, dict):
                message = str(data.get("message") or message)
                code = str(data.get("code") or "")
                err_type = str(data.get("type") or "")
        except Exception:
            pass
        info: dict[str, Any] = {
            "message": HttpClient._safe_body_excerpt(message, limit=500),
            "code": code,
            "type": err_type,
        }
        msg_l = message.lower()
        if "tokens per day" in msg_l or "(tpd)" in msg_l:
            info["quota"] = "tokens_per_day"
        elif "tokens per minute" in msg_l or "(tpm)" in msg_l:
            info["quota"] = "tokens_per_minute"
        elif "requests per minute" in msg_l or "(rpm)" in msg_l:
            info["quota"] = "requests_per_minute"
        elif "rate limit" in msg_l or "too many requests" in msg_l:
            info["quota"] = "rate_limit"

        m = _TRY_AGAIN_RE.search(message)
        if m:
            wait_s = _parse_reset_duration(m.group(1))
            if wait_s > 0:
                info["try_again_s"] = wait_s
        for label, key in (("Limit", "limit"), ("Used", "used"), ("Requested", "requested")):
            m2 = re.search(rf"\b{label}\s+(\d+)", message)
            if m2:
                info[key] = int(m2.group(1))
        return {k: v for k, v in info.items() if v not in ("", None)}

    @staticmethod
    def _http_error_body(exc: urllib.error.HTTPError) -> str:
        try:
            return exc.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _response_headers(headers: Any) -> dict[str, str]:
        if not headers:
            return {}
        out: dict[str, str] = {}
        try:
            items = headers.items()
        except Exception:
            items = []
        for key, value in items:
            key_l = str(key).lower()
            if value is None:
                continue
            if key_l in {"retry-after"} or key_l.startswith("x-ratelimit-"):
                out[key_l] = str(value)
        return out

    @staticmethod
    def _http_error_data(exc: urllib.error.HTTPError) -> tuple[str, str]:
        body = HttpClient._http_error_body(exc)
        return body, HttpClient._safe_body_excerpt(body)

    @staticmethod
    def _fill_http_error_trace(attempt_trace: dict[str, Any], exc: urllib.error.HTTPError) -> str:
        body, excerpt = HttpClient._http_error_data(exc)
        attempt_trace["status"] = exc.code
        attempt_trace["retry_after_s"] = HttpClient._retry_after(exc.headers)
        attempt_trace["rate_limit"] = HttpClient._rate_limit_headers(exc.headers)
        attempt_trace["headers"] = HttpClient._response_headers(exc.headers)
        attempt_trace["body_excerpt"] = excerpt
        parsed = HttpClient._parse_rate_limit_body(body)
        if parsed:
            attempt_trace["rate_limit_error"] = parsed
        return body

    @staticmethod
    def _fill_success_trace(attempt_trace: dict[str, Any], resp: Any) -> None:
        attempt_trace["status"] = getattr(resp, "status", 200)
        attempt_trace["rate_limit"] = HttpClient._rate_limit_headers(resp.headers)
        attempt_trace["headers"] = HttpClient._response_headers(resp.headers)

    @staticmethod
    def _empty_attempt_trace(attempt: int) -> dict[str, Any]:
        return {
            "attempt": attempt + 1,
            "started_at": HttpClient._now_iso(),
            "elapsed_s": 0.0,
            "status": "",
            "error": "",
            "retry_after_s": 0.0,
            "sleep_s": 0.0,
            "sleep_reason": "",
            "pacing_sleep_s": 0.0,
            "rate_limit": {},
            "headers": {},
            "body_excerpt": "",
            "rate_limit_error": {},
        }

    @staticmethod
    def _empty_get_attempt_trace(attempt: int) -> dict[str, Any]:
        trace = HttpClient._empty_attempt_trace(attempt)
        trace.pop("pacing_sleep_s", None)
        return trace

    @staticmethod
    def _header_float(headers: Any, key: str) -> float | None:
        try:
            value = headers.get(key)
        except Exception:
            value = None
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _header_duration(headers: Any, key: str) -> float:
        try:
            value = headers.get(key)
        except Exception:
            value = None
        return _parse_reset_duration(str(value)) if value else 0.0

    @staticmethod
    def _retry_wait_s(
        exc: urllib.error.HTTPError,
        body: str,
        attempt: int,
        *,
        min_remaining_tokens: int = 0,
        max_retry_wait: float = 0.0,
    ) -> tuple[float, str]:
        parsed = HttpClient._parse_rate_limit_body(body)
        try_again = float(parsed.get("try_again_s") or 0.0)
        if try_again > 0:
            # Provider body has the most specific reset for daily/model limits.
            return math.ceil(try_again), str(parsed.get("quota") or "provider_retry")

        headers = exc.headers
        remaining_requests = HttpClient._header_float(headers, "x-ratelimit-remaining-requests")
        if remaining_requests is not None and remaining_requests <= 0:
            reset_s = HttpClient._header_duration(headers, "x-ratelimit-reset-requests")
            if reset_s > 0:
                return math.ceil(reset_s) + 1, "requests_reset"

        remaining_tokens = HttpClient._header_float(headers, "x-ratelimit-remaining-tokens")
        if (
            min_remaining_tokens > 0
            and remaining_tokens is not None
            and remaining_tokens < min_remaining_tokens
        ):
            reset_s = HttpClient._header_duration(headers, "x-ratelimit-reset-tokens")
            if reset_s > 0:
                return math.ceil(reset_s) + 1, "tokens_reset"

        retry_after = HttpClient._retry_after(headers)
        if retry_after > 0:
            return math.ceil(retry_after), "retry_after"

        wait = 2 ** (attempt + 2)
        if max_retry_wait > 0:
            wait = min(wait, max_retry_wait)
        return wait, "exponential_backoff"

    def _start_trace(self, method: str, url: str, namespace: str, bucket: str) -> dict[str, Any]:
        return {
            "method": method,
            "url": self._redacted_url(url),
            "namespace": namespace,
            "bucket": bucket,
            "cache_hit": False,
            "started_at": self._now_iso(),
            "elapsed_s": 0.0,
            "attempts": [],
            "attempt_count": 0,
            "retry_sleep_s": 0.0,
            "pacing_sleep_s": 0.0,
            "final_status": "",
            "final_error": "",
        }

    def _finish_trace(self, bucket: str, trace: dict[str, Any], started: float) -> None:
        trace["elapsed_s"] = max(0.0, time.monotonic() - started)
        trace["attempt_count"] = len(trace.get("attempts") or [])
        self._last_request_trace[bucket] = trace

    def take_last_request_trace(self, bucket: str) -> dict[str, Any]:
        """Return and clear the last compact HTTP trace for *bucket*."""
        return self._last_request_trace.pop(bucket, {})

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

    def _wait_for_token_budget(self, bucket: str, min_remaining: int) -> float:
        """Sleep proactively if the token budget is too low for another API call.

        Called before each attempt in the retry loop so that after a capped
        429-sleep we recheck and wait out any remaining reset time rather than
        immediately firing another request into an exhausted quota.
        """
        entry = self._token_budget.get(bucket)
        if entry is None:
            return 0.0
        remaining, reset_at = entry
        if remaining >= min_remaining:
            return 0.0
        wait = reset_at - time.monotonic()
        if wait <= 0:
            return 0.0
        eprint(
            f"[pacing] {bucket}: {remaining} tokens remaining "
            f"(need ≥{min_remaining}), sleeping {wait:.1f}s for quota reset"
        )
        actual = wait + 1.0  # +1 s buffer so we don't race the window edge
        time.sleep(actual)
        self._pacing_sleep[bucket] = self._pacing_sleep.get(bucket, 0.0) + actual
        return actual

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
        trace_started = time.monotonic()
        trace = self._start_trace("GET", url, namespace, bucket)
        cached = self.cache.load(namespace, cache_key)
        if cached is not None and self._json_is_cacheable(cached):
            trace["cache_hit"] = True
            trace["final_status"] = "cache_hit"
            self._finish_trace(bucket, trace, trace_started)
            return cached
        req_headers = {"Accept": "application/json", "User-Agent": self._agent()}
        if headers:
            req_headers.update(headers)

        data: Any = None
        for attempt in range(max_retries):
            self._throttle(bucket, min_interval)
            attempt_started = time.monotonic()
            attempt_trace = self._empty_get_attempt_trace(attempt)
            req = urllib.request.Request(url, headers=req_headers)
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    self._fill_success_trace(attempt_trace, resp)
                break  # success
            except urllib.error.HTTPError as exc:
                body_text = self._fill_http_error_trace(attempt_trace, exc)
                if exc.code in (429, 503) and attempt < max_retries - 1:
                    wait, wait_reason = self._retry_wait_s(exc, body_text, attempt)
                    attempt_trace["sleep_s"] = wait
                    attempt_trace["sleep_reason"] = wait_reason
                    trace["retry_sleep_s"] += wait
                    if self.verbose:
                        eprint(f"[retry] GET {url} HTTP {exc.code}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                data = {"_http_error": exc.code, "_url": url, "_body": body_text}
                if self.verbose:
                    eprint(f"[warn] GET {url} HTTP {exc.code}")
                break
            except Exception as exc:
                attempt_trace["error"] = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    attempt_trace["sleep_s"] = wait
                    trace["retry_sleep_s"] += wait
                    if self.verbose:
                        eprint(f"[retry] GET failed {url}: {exc}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] GET failed {url}: {exc}")
                trace["final_status"] = "error"
                trace["final_error"] = attempt_trace["error"]
                self._finish_trace(bucket, trace, trace_started)
                return None
            finally:
                attempt_trace["elapsed_s"] = max(0.0, time.monotonic() - attempt_started)
                if attempt_trace not in trace["attempts"]:
                    trace["attempts"].append(attempt_trace)

        if data is not None and self._json_is_cacheable(data):
            self.cache.save(namespace, cache_key, data)
        if isinstance(data, dict) and "_http_error" in data:
            trace["final_status"] = f"http_{data['_http_error']}"
        elif data is None:
            trace["final_status"] = "none"
        else:
            trace["final_status"] = "ok"
        self._finish_trace(bucket, trace, trace_started)
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
        trace_started = time.monotonic()
        trace = self._start_trace("GET", url, namespace, bucket)
        cached = self.cache.load(namespace, cache_key)
        if isinstance(cached, dict) and "_text" in cached and cached["_text"]:
            trace["cache_hit"] = True
            trace["final_status"] = "cache_hit"
            self._finish_trace(bucket, trace, trace_started)
            return str(cached["_text"])  # Only return non-empty cached text

        raw = ""
        for attempt in range(max_retries):
            self._throttle(bucket, min_interval)
            attempt_started = time.monotonic()
            attempt_trace = self._empty_get_attempt_trace(attempt)
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
                    self._fill_success_trace(attempt_trace, resp)
                break  # success
            except urllib.error.HTTPError as exc:
                body_text = self._fill_http_error_trace(attempt_trace, exc)
                if exc.code in (429, 503) and attempt < max_retries - 1:
                    wait, wait_reason = self._retry_wait_s(exc, body_text, attempt)
                    attempt_trace["sleep_s"] = wait
                    attempt_trace["sleep_reason"] = wait_reason
                    trace["retry_sleep_s"] += wait
                    if self.verbose:
                        eprint(f"[retry] GET xml {url} HTTP {exc.code}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] GET xml {url} HTTP {exc.code}")
                trace["final_status"] = f"http_{exc.code}"
                self._finish_trace(bucket, trace, trace_started)
                return ""
            except Exception as exc:
                attempt_trace["error"] = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    attempt_trace["sleep_s"] = wait
                    trace["retry_sleep_s"] += wait
                    if self.verbose:
                        eprint(f"[retry] GET xml failed {url}: {exc}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] GET xml failed {url}: {exc}")
                trace["final_status"] = "error"
                trace["final_error"] = attempt_trace["error"]
                self._finish_trace(bucket, trace, trace_started)
                return ""
            finally:
                attempt_trace["elapsed_s"] = max(0.0, time.monotonic() - attempt_started)
                trace["attempts"].append(attempt_trace)

        if raw.strip():
            self.cache.save(namespace, cache_key, {"_text": raw})
            trace["final_status"] = "ok"
        else:
            trace["final_status"] = "empty"
        self._finish_trace(bucket, trace, trace_started)
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
        trace_started = time.monotonic()
        trace = self._start_trace("POST", url, namespace, bucket)
        trace["min_remaining_tokens"] = min_remaining_tokens
        cached = self.cache.load(namespace, cache_key)
        if cached is not None and self._json_is_cacheable(cached):
            trace["cache_hit"] = True
            trace["final_status"] = "cache_hit"
            self._finish_trace(bucket, trace, trace_started)
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
            attempt_started = time.monotonic()
            attempt_trace = self._empty_attempt_trace(attempt)
            if min_remaining_tokens > 0:
                slept = self._wait_for_token_budget(bucket, min_remaining_tokens)
                attempt_trace["pacing_sleep_s"] = slept
                trace["pacing_sleep_s"] += slept
            req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    self._update_token_budget(bucket, resp.headers)
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    self._fill_success_trace(attempt_trace, resp)
                break  # success
            except urllib.error.HTTPError as exc:
                if exc.headers:
                    self._update_token_budget(bucket, exc.headers)
                body_text = self._fill_http_error_trace(attempt_trace, exc)
                if exc.code in (429, 503) and attempt < max_retries - 1:
                    wait, wait_reason = self._retry_wait_s(
                        exc,
                        body_text,
                        attempt,
                        min_remaining_tokens=min_remaining_tokens,
                        max_retry_wait=max_retry_wait,
                    )
                    attempt_trace["sleep_s"] = wait
                    attempt_trace["sleep_reason"] = wait_reason
                    trace["retry_sleep_s"] += wait
                    if self.verbose:
                        eprint(f"[retry] POST {url} HTTP {exc.code}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                data = {"_http_error": exc.code, "_url": url, "_body": body_text}
                if self.verbose:
                    eprint(f"[warn] POST {url} HTTP {exc.code}: {body_text[:200]}")
                break
            except Exception as exc:
                attempt_trace["error"] = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    attempt_trace["sleep_s"] = wait
                    trace["retry_sleep_s"] += wait
                    if self.verbose:
                        eprint(f"[retry] POST failed {url}: {exc}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] POST failed {url}: {exc}")
                trace["final_status"] = "error"
                trace["final_error"] = attempt_trace["error"]
                self._finish_trace(bucket, trace, trace_started)
                return None
            finally:
                attempt_trace["elapsed_s"] = max(0.0, time.monotonic() - attempt_started)
                trace["attempts"].append(attempt_trace)

        if data is not None and self._json_is_cacheable(data):
            self.cache.save(namespace, cache_key, data)
        if isinstance(data, dict) and "_http_error" in data:
            trace["final_status"] = f"http_{data['_http_error']}"
        elif data is None:
            trace["final_status"] = "none"
        else:
            trace["final_status"] = "ok"
        self._finish_trace(bucket, trace, trace_started)
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
