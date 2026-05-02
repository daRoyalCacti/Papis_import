"""Stateless rate-limit parsing and retry-wait policy helpers."""
from __future__ import annotations

import json
import math
import re
import urllib.error
from typing import Any


_RESET_DURATION_RE = re.compile(r"([\d.]+)(ms|s|m|h)")
_TRY_AGAIN_RE = re.compile(r"please\s+try\s+again\s+in\s+((?:[\d.]+(?:ms|s|m|h)\s*)+)", re.I)


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


def _safe_body_excerpt(body: str, limit: int = 1000) -> str:
    """Return a compact HTTP error body excerpt suitable for TSV debug."""
    if not body:
        return ""
    redacted = re.sub(
        r'(?i)("?(?:api[_-]?key|authorization|token|secret)"?\s*[:=]\s*")([^"]+)(")',
        r"\1[redacted]\3",
        body,
    )
    redacted = re.sub(r"\s+", " ", redacted).strip()
    return redacted[:limit]


def _retry_after(headers: Any) -> float:
    if not headers:
        return 0.0
    try:
        return float(headers.get("Retry-After", 0) or 0)
    except Exception:
        return 0.0


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


def _header_duration(headers: Any, key: str) -> float:
    try:
        value = headers.get(key)
    except Exception:
        value = None
    return _parse_reset_duration(str(value)) if value else 0.0


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
        "message": _safe_body_excerpt(message, limit=500),
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


def _retry_wait_s(
    exc: urllib.error.HTTPError,
    body: str,
    attempt: int,
    *,
    min_remaining_tokens: int = 0,
    max_retry_wait: float = 0.0,
) -> tuple[float, str]:
    parsed = _parse_rate_limit_body(body)
    try_again = float(parsed.get("try_again_s") or 0.0)
    if try_again > 0:
        return math.ceil(try_again), str(parsed.get("quota") or "provider_retry")

    headers = exc.headers
    remaining_requests = _header_float(headers, "x-ratelimit-remaining-requests")
    if remaining_requests is not None and remaining_requests <= 0:
        reset_s = _header_duration(headers, "x-ratelimit-reset-requests")
        if reset_s > 0:
            return math.ceil(reset_s) + 1, "requests_reset"

    remaining_tokens = _header_float(headers, "x-ratelimit-remaining-tokens")
    if (
        min_remaining_tokens > 0
        and remaining_tokens is not None
        and remaining_tokens < min_remaining_tokens
    ):
        reset_s = _header_duration(headers, "x-ratelimit-reset-tokens")
        if reset_s > 0:
            return math.ceil(reset_s) + 1, "tokens_reset"

    retry_after_val = _retry_after(headers)
    if retry_after_val > 0:
        return math.ceil(retry_after_val), "retry_after"

    wait = 2 ** (attempt + 2)
    if max_retry_wait > 0:
        wait = min(wait, max_retry_wait)
    return wait, "exponential_backoff"
