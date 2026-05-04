"""Helpers for resolving remote vs local LLM settings."""
from __future__ import annotations

import argparse
from dataclasses import dataclass


DEFAULT_LOCAL_LLM_MODEL = "qwen3-vl:8b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class LlmRequestConfig:
    endpoint: str
    model: str
    api_key: str
    timeout_s: float
    track_tokens: bool
    min_interval: float
    min_remaining_tokens: int
    local: bool = False
    max_tokens: int = 0
    think: bool | str | None = None
    # Model cycling (remote path only; ignored when local=True)
    models: tuple[str, ...] = ()
    switch_threshold_s: float = 120.0
    soft_failure_cooldown_s: float = 60.0


def _resolve_llm_models(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the ordered model preference list from args.

    Prefers --llm-models (comma-separated); falls back to --llm-model as a
    one-element list for backward compatibility.
    """
    raw = str(getattr(args, "llm_models", "") or "").strip()
    if raw:
        return tuple(m.strip() for m in raw.split(",") if m.strip())
    single = str(getattr(args, "llm_model", "") or "").strip()
    return (single,) if single else ()


def local_llm_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "local_llm", False) or getattr(args, "ollama_model", ""))


def local_llm_model(args: argparse.Namespace) -> str:
    model = str(getattr(args, "local_llm_model", "") or "").strip()
    legacy = str(getattr(args, "ollama_model", "") or "").strip()
    return model or legacy or DEFAULT_LOCAL_LLM_MODEL


def ollama_host(args: argparse.Namespace) -> str:
    return (str(getattr(args, "ollama_host", "") or "").strip() or DEFAULT_OLLAMA_HOST).rstrip("/")


def ollama_openai_endpoint(args: argparse.Namespace) -> str:
    return ollama_host(args) + "/v1"


def local_llm_num_predict(args: argparse.Namespace) -> int:
    return int(getattr(args, "local_llm_num_predict", 2048) or 2048)


def text_llm_config(args: argparse.Namespace) -> LlmRequestConfig:
    if local_llm_enabled(args):
        return LlmRequestConfig(
            endpoint=ollama_host(args),
            model=local_llm_model(args),
            api_key="",
            timeout_s=float(getattr(args, "ollama_request_timeout", 300.0) or 300.0),
            track_tokens=False,
            min_interval=0.0,
            min_remaining_tokens=0,
            local=True,
            max_tokens=local_llm_num_predict(args),
            think=False,
        )
    models = _resolve_llm_models(args)
    # Space out calls per model so the aggregate endpoint rate stays reasonable
    # (one model active at a time in normal operation; cycling is the exception).
    min_interval = 0.5
    return LlmRequestConfig(
        endpoint=str(getattr(args, "llm_endpoint", "") or "").strip(),
        model=models[0] if models else "",
        api_key=str(getattr(args, "llm_api_key", "") or "").strip(),
        timeout_s=float(getattr(args, "llm_request_timeout", 90.0) or 90.0),
        track_tokens=True,
        min_interval=min_interval,
        min_remaining_tokens=-1,
        max_tokens=500,
        models=models,
        switch_threshold_s=float(getattr(args, "llm_switch_threshold", 120.0) or 120.0),
    )


def vision_llm_config(args: argparse.Namespace) -> LlmRequestConfig:
    if local_llm_enabled(args):
        return LlmRequestConfig(
            endpoint=ollama_host(args),
            model=local_llm_model(args),
            api_key="",
            timeout_s=float(getattr(args, "ollama_request_timeout", 300.0) or 300.0),
            track_tokens=False,
            min_interval=0.0,
            min_remaining_tokens=0,
            local=True,
            max_tokens=local_llm_num_predict(args),
            think=False,
        )

    endpoint = str(getattr(args, "vision_llm_endpoint", "") or "").strip()
    api_key = str(getattr(args, "vision_llm_api_key", "") or "").strip()
    if not endpoint:
        endpoint = str(getattr(args, "llm_endpoint", "") or "").strip()
    if not api_key:
        api_key = str(getattr(args, "llm_api_key", "") or "").strip()
    return LlmRequestConfig(
        endpoint=endpoint,
        model=str(getattr(args, "vision_llm_model", "") or "").strip(),
        api_key=api_key,
        timeout_s=float(getattr(args, "vision_llm_request_timeout", 90.0) or 90.0),
        track_tokens=True,
        min_interval=0.5,
        min_remaining_tokens=10_000,
        max_tokens=600,
    )


def local_llm_response_is_cacheable(data: object) -> bool:
    if not isinstance(data, dict):
        return True
    if str(data.get("done_reason", "")).lower() == "length":
        return False
    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict) and str(choice.get("finish_reason", "")).lower() == "length":
                return False
    return True
