"""Throttled, cached HTTP client used by all resolver functions."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from papis_import.http.cache import Cache, json_is_cacheable
from papis_import.http.throttle import _parse_reset_duration, _retry_wait_s
from papis_import.http.trace import (
    _empty_attempt_trace,
    _fill_http_error_trace,
    _fill_success_trace,
    _now_iso,
    _redacted_url,
)
from papis_import.output.writers import LiveStatusReporter
from papis_import.utils import USER_AGENT, eprint


def _phase_for_bucket(bucket: str) -> str:
    if bucket == "llm":
        return "text_llm"
    if bucket == "vision_llm":
        return "vision_llm"
    if bucket in {"crossref", "openalex", "openlibrary", "google_books", "semanticscholar", "arxiv"}:
        return "title_search"
    return "identifier_lookup"


class HttpClient:
    """Throttled, cached HTTP helper for all external API calls."""

    def __init__(
        self,
        cache: Cache,
        mailto: str = "",
        verbose: bool = False,
        live_reporter: LiveStatusReporter | None = None,
    ) -> None:
        self.cache   = cache
        self.mailto  = mailto.strip()   # used for both Crossref & OpenAlex polite pool
        self.verbose = verbose
        self.live_reporter = live_reporter
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

    def _start_trace(self, method: str, url: str, namespace: str, bucket: str) -> dict[str, Any]:
        return {
            "method": method,
            "url": _redacted_url(url),
            "namespace": namespace,
            "bucket": bucket,
            "cache_hit": False,
            "started_at": _now_iso(),
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
        if self.live_reporter is not None:
            self.live_reporter.update(
                status="sleeping",
                phase=_phase_for_bucket(bucket),
                bucket=bucket,
                event="pacing_sleep",
                message=(
                    f"{bucket} token pacing: {remaining} remaining, "
                    f"need {min_remaining}, sleeping {actual:.1f}s"
                ),
                wait_s=round(actual, 3),
                sleep_reason="tokens_reset",
                remaining_tokens=remaining,
                needed_tokens=min_remaining,
            )
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

    def _request_with_retry(
        self,
        method: str,
        url: str,
        namespace: str,
        cache_key: str,
        req_headers: dict[str, str],
        req_body: bytes | None,
        *,
        bucket: str,
        min_interval: float,
        max_retries: int,
        timeout_s: float,
        parse_response: Any,       # Callable[[bytes], Any]
        cache_valid: Any,          # Callable[[Any], bool] — is cached data usable?
        cache_extract: Any,        # Callable[[Any], Any] — pull value out of cache hit
        cache_wrap: Any,           # Callable[[Any], Any | None] — wrap result for storage
        failure_value: Any,
        http_error_as_data: bool,  # True: store HTTPError as {"_http_error":..} and break
        log_kind: str = "",        # human name for eprint messages
        track_tokens: bool = False,
        min_remaining_tokens: int = 0,
        max_retry_wait: float = 0.0,
    ) -> Any:
        trace_started = time.monotonic()
        trace = self._start_trace(method, url, namespace, bucket)
        if track_tokens:
            trace["min_remaining_tokens"] = min_remaining_tokens

        cached = self.cache.load(namespace, cache_key)
        if cached is not None and cache_valid(cached):
            trace["cache_hit"] = True
            trace["final_status"] = "cache_hit"
            self._finish_trace(bucket, trace, trace_started)
            return cache_extract(cached)

        result: Any = failure_value
        for attempt in range(max_retries):
            self._throttle(bucket, min_interval)
            attempt_started = time.monotonic()
            attempt_trace = _empty_attempt_trace(attempt, include_pacing=track_tokens)
            if track_tokens and min_remaining_tokens > 0:
                slept = self._wait_for_token_budget(bucket, min_remaining_tokens)
                attempt_trace["pacing_sleep_s"] = slept
                trace["pacing_sleep_s"] += slept
            if self.live_reporter is not None:
                self.live_reporter.update(
                    status="processing",
                    phase=_phase_for_bucket(bucket),
                    bucket=bucket,
                    event="request_started",
                    method=method,
                    url=_redacted_url(url),
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    message=f"{bucket} {method} request started",
                )
            req = urllib.request.Request(url, data=req_body, headers=req_headers,
                                         method=method)
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    if track_tokens:
                        self._update_token_budget(bucket, resp.headers)
                    result = parse_response(resp.read())
                    _fill_success_trace(attempt_trace, resp)
                break
            except urllib.error.HTTPError as exc:
                if track_tokens and exc.headers:
                    self._update_token_budget(bucket, exc.headers)
                body_text = _fill_http_error_trace(attempt_trace, exc)
                if exc.code in (429, 503) and attempt < max_retries - 1:
                    wait, wait_reason = _retry_wait_s(
                        exc, body_text, attempt,
                        min_remaining_tokens=min_remaining_tokens,
                        max_retry_wait=max_retry_wait,
                    )
                    attempt_trace["sleep_s"] = wait
                    attempt_trace["sleep_reason"] = wait_reason
                    trace["retry_sleep_s"] += wait
                    eprint(
                        f"[retry] {bucket} {method} HTTP {exc.code}, "
                        f"sleeping {wait}s reason={wait_reason} "
                        f"attempt {attempt+1}/{max_retries} url={_redacted_url(url)}"
                    )
                    if self.live_reporter is not None:
                        self.live_reporter.update(
                            status="sleeping",
                            phase=_phase_for_bucket(bucket),
                            bucket=bucket,
                            event="rate_limit_sleep",
                            method=method,
                            url=_redacted_url(url),
                            message=(
                                f"{bucket} {method} HTTP {exc.code}; "
                                f"sleeping {wait}s ({wait_reason})"
                            ),
                            wait_s=wait,
                            sleep_reason=wait_reason,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            http_status=exc.code,
                            rate_limit=attempt_trace.get("rate_limit") or {},
                            rate_limit_error=attempt_trace.get("rate_limit_error") or {},
                        )
                    time.sleep(wait)
                    continue
                if http_error_as_data:
                    result = {"_http_error": exc.code, "_url": url, "_body": body_text}
                    if self.verbose:
                        eprint(f"[warn] {log_kind} {url} HTTP {exc.code}")
                    break
                else:
                    if self.verbose:
                        eprint(f"[warn] {log_kind} {url} HTTP {exc.code}")
                    trace["final_status"] = f"http_{exc.code}"
                    self._finish_trace(bucket, trace, trace_started)
                    return failure_value
            except Exception as exc:
                attempt_trace["error"] = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    attempt_trace["sleep_s"] = wait
                    trace["retry_sleep_s"] += wait
                    if self.verbose:
                        eprint(f"[retry] {log_kind} failed {url}: {exc},"
                               f" waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] {log_kind} failed {url}: {exc}")
                trace["final_status"] = "error"
                trace["final_error"] = attempt_trace["error"]
                self._finish_trace(bucket, trace, trace_started)
                return failure_value
            finally:
                attempt_trace["elapsed_s"] = max(0.0, time.monotonic() - attempt_started)
                if attempt_trace not in trace["attempts"]:
                    trace["attempts"].append(attempt_trace)

        to_store = cache_wrap(result)
        if to_store is not None:
            self.cache.save(namespace, cache_key, to_store)
        if http_error_as_data and isinstance(result, dict) and "_http_error" in result:
            trace["final_status"] = f"http_{result['_http_error']}"
        elif result is None or result == failure_value:
            trace["final_status"] = "none" if http_error_as_data else "empty"
        else:
            trace["final_status"] = "ok"
        self._finish_trace(bucket, trace, trace_started)
        return result

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
        req_headers: dict[str, str] = {"Accept": "application/json", "User-Agent": self._agent()}
        if headers:
            req_headers.update(headers)
        return self._request_with_retry(
            "GET", url, namespace, cache_key, req_headers, None,
            bucket=bucket, min_interval=min_interval, max_retries=max_retries,
            timeout_s=25, log_kind="GET",
            parse_response=lambda b: json.loads(b.decode("utf-8", errors="replace")),
            cache_valid=lambda c: c is not None and json_is_cacheable(c),
            cache_extract=lambda c: c,
            cache_wrap=lambda d: d if d is not None and json_is_cacheable(d) else None,
            failure_value=None,
            http_error_as_data=True,
        )

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
        req_headers = {
            "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.8",
            "User-Agent": self._agent(),
        }
        return self._request_with_retry(
            "GET", url, namespace, cache_key, req_headers, None,
            bucket=bucket, min_interval=min_interval, max_retries=max_retries,
            timeout_s=25, log_kind="GET xml",
            parse_response=lambda b: b.decode("utf-8", errors="replace"),
            cache_valid=lambda c: isinstance(c, dict) and "_text" in c and bool(c["_text"]),
            cache_extract=lambda c: str(c["_text"]),
            cache_wrap=lambda raw: {"_text": raw} if isinstance(raw, str) and raw.strip() else None,
            failure_value="",
            http_error_as_data=False,
        )

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
        timeout_s: float = 90.0,
        track_tokens: bool = True,
        response_cacheable: Any | None = None,
    ) -> Any | None:
        hdrs: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self._agent(),
        }
        if extra_headers:
            hdrs.update(extra_headers)
        return self._request_with_retry(
            "POST", url, namespace, cache_key, hdrs,
            json.dumps(payload).encode("utf-8"),
            bucket=bucket, min_interval=min_interval, max_retries=max_retries,
            timeout_s=timeout_s, log_kind="POST",
            parse_response=lambda b: json.loads(b.decode("utf-8", errors="replace")),
            cache_valid=lambda c: (
                c is not None
                and json_is_cacheable(c)
                and (response_cacheable(c) if response_cacheable is not None else True)
            ),
            cache_extract=lambda c: c,
            cache_wrap=lambda d: (
                d
                if (
                    d is not None
                    and json_is_cacheable(d)
                    and (response_cacheable(d) if response_cacheable is not None else True)
                )
                else None
            ),
            failure_value=None,
            http_error_as_data=True,
            track_tokens=track_tokens,
            min_remaining_tokens=min_remaining_tokens,
            max_retry_wait=max_retry_wait,
        )

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
        trace_started = time.monotonic()
        trace = self._start_trace("POST", url, namespace, bucket)
        cached = self.cache.load(namespace, cache_key)
        if isinstance(cached, dict) and "_text" in cached and cached["_text"]:
            trace["cache_hit"] = True
            trace["final_status"] = "cache_hit"
            self._finish_trace(bucket, trace, trace_started)
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
            attempt_started = time.monotonic()
            attempt_trace = _empty_attempt_trace(attempt, include_pacing=False)
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
                    _fill_success_trace(attempt_trace, resp)
                break
            except urllib.error.HTTPError as exc:
                body_text = _fill_http_error_trace(attempt_trace, exc)
                if exc.code in (429, 503) and attempt < max_retries - 1:
                    wait, wait_reason = _retry_wait_s(exc, body_text, attempt)
                    attempt_trace["sleep_s"] = wait
                    attempt_trace["sleep_reason"] = wait_reason
                    trace["retry_sleep_s"] += wait
                    if self.verbose:
                        eprint(f"[retry] multipart POST {url} HTTP {exc.code}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] multipart POST {url} HTTP {exc.code}")
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
                        eprint(f"[retry] multipart POST failed {url}: {exc}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if self.verbose:
                    eprint(f"[warn] multipart POST failed {url}: {exc}")
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
