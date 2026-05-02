"""Stateless per-attempt trace construction helpers."""
from __future__ import annotations

import time
import urllib.error
import urllib.parse
from typing import Any

from papis_import.http.throttle import (
    _parse_rate_limit_body,
    _rate_limit_headers,
    _response_headers,
    _retry_after,
    _safe_body_excerpt,
)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _redacted_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _http_error_data(exc: urllib.error.HTTPError) -> tuple[str, str]:
    body = _http_error_body(exc)
    return body, _safe_body_excerpt(body)


def _empty_attempt_trace(attempt: int, *, include_pacing: bool = True) -> dict[str, Any]:
    """Return a blank per-attempt trace dict.

    *include_pacing* controls whether ``pacing_sleep_s`` is present — True for
    POST requests that track token budgets, False for plain GET requests.
    """
    trace: dict[str, Any] = {
        "attempt": attempt + 1,
        "started_at": _now_iso(),
        "elapsed_s": 0.0,
        "status": "",
        "error": "",
        "retry_after_s": 0.0,
        "sleep_s": 0.0,
        "sleep_reason": "",
        "rate_limit": {},
        "headers": {},
        "body_excerpt": "",
        "rate_limit_error": {},
    }
    if include_pacing:
        trace["pacing_sleep_s"] = 0.0
    return trace


def _fill_http_error_trace(attempt_trace: dict[str, Any], exc: urllib.error.HTTPError) -> str:
    body, excerpt = _http_error_data(exc)
    attempt_trace["status"] = exc.code
    attempt_trace["retry_after_s"] = _retry_after(exc.headers)
    attempt_trace["rate_limit"] = _rate_limit_headers(exc.headers)
    attempt_trace["headers"] = _response_headers(exc.headers)
    attempt_trace["body_excerpt"] = excerpt
    parsed = _parse_rate_limit_body(body)
    if parsed:
        attempt_trace["rate_limit_error"] = parsed
    return body


def _fill_success_trace(attempt_trace: dict[str, Any], resp: Any) -> None:
    attempt_trace["status"] = getattr(resp, "status", 200)
    attempt_trace["rate_limit"] = _rate_limit_headers(resp.headers)
    attempt_trace["headers"] = _response_headers(resp.headers)
