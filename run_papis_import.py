#!/usr/bin/env python3
"""Thin wrapper that injects saved defaults from a JSON config file.

Config file location (default): ~/.config/papis-import/config.json

Example config.json:
{
    "mailto":               "you@example.com",
    "google_books_api_key": "AIza...",
    "grobid_url":           "http://localhost:8070",
    "start_local_grobid":   true,
    "grobid_image":         "grobid/grobid:0.9.0-full",
    "llm_endpoint":         "https://api.groq.com/openai/v1",
    "llm_api_key":          "gsk_...",
    "llm_model":            "llama-3.3-70b-versatile",
    "llm_models":           ["llama-3.3-70b-versatile", "openai/gpt-oss-120b",
                             "qwen/qwen3-32b", "meta-llama/llama-4-scout-17b-16e-instruct",
                             "openai/gpt-oss-20b", "llama-3.1-8b-instant"],
    "llm_switch_threshold": 120,
    "vision_llm_endpoint":  "https://api.groq.com/openai/v1",
    "vision_llm_api_key":   "gsk_...",
    "vision_llm_model":     "meta-llama/llama-4-scout-17b-16e-instruct",
    "local_llm":            false,
    "local_llm_model":      "qwen3-vl:8b",
    "local_llm_num_predict": 2048,
    "ollama_host":          "http://127.0.0.1:11434",
    "ollama_request_timeout": 300,
    "ollama_pull_missing":  false,
    "vision_only_if_hard":  true,
    "no_semantic_scholar":  false,
    "semantic_scholar_api_key": "",

    "tsv":                 "~/Documents/papis_import.tsv",
    "review_tsv":          "~/Documents/papis_import_review.tsv",
    "soft_tsv":            "~/Documents/papis_import_soft.tsv",
    "debug_jsonl":         "~/Documents/papis_import_debug.jsonl",
    "profile_tsv":         "~/Documents/papis_import_profile.tsv",
    "live_status_json":    "~/Documents/papis_import_current_status.json",
    "cache_dir":           "~/.cache/papis_import",
    "title_search_timeout": 12
}

Any value already supplied on the command line takes precedence over the config.

Usage:
    python run_papis_import.py --staging ~/Literature_pre_papis --dry-run
    python run_papis_import.py --staging ~/Literature_pre_papis --import --link
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from io_utils import DEFAULT_CONFIG_PATH, load_json_config

_CONFIG_PATH = DEFAULT_CONFIG_PATH

# Keys in config.json  →  CLI flag name (without leading --)
_CONFIG_MAP = {
    "mailto":               "mailto",
    "crossref_mailto":      "crossref-mailto",      # legacy alias
    "tsv":                  "tsv",
    "review_tsv":           "review-tsv",
    "soft_tsv":             "soft-tsv",
    "debug_jsonl":          "debug-jsonl",
    "profile_tsv":          "profile-tsv",
    "live_status_json":     "live-status-json",
    "cache_dir":            "cache-dir",
    "title_search_timeout": "title-search-timeout",
    "google_books_api_key": "google-books-api-key",
    "grobid_url":           "grobid-url",
    "start_local_grobid":   "start-local-grobid",
    "grobid_runtime":       "grobid-runtime",
    "grobid_image":         "grobid-image",
    "grobid_port":          "grobid-port",
    "grobid_start_timeout": "grobid-start-timeout",
    "llm_endpoint":         "llm-endpoint",
    "llm_api_key":          "llm-api-key",
    "llm_model":            "llm-model",
    "llm_models":           "llm-models",
    "llm_switch_threshold": "llm-switch-threshold",
    "llm_chars":            "llm-chars",
    "llm_request_timeout":  "llm-request-timeout",
    "local_llm":            "local-llm",
    "local_llm_model":      "local-llm-model",
    "local_llm_num_predict": "local-llm-num-predict",
    "ollama_host":          "ollama-host",
    "ollama_start_timeout": "ollama-start-timeout",
    "ollama_request_timeout": "ollama-request-timeout",
    "ollama_pull_missing":  "ollama-pull-missing",
    "vision_llm_endpoint":  "vision-llm-endpoint",
    "vision_llm_api_key":   "vision-llm-api-key",
    "vision_llm_model":     "vision-llm-model",
    "vision_llm_request_timeout": "vision-llm-request-timeout",
    "vision_pages":         "vision-pages",
    "vision_pages_escalate": "vision-pages-escalate",
    "vision_dpi":           "vision-dpi",
    "vision_only_if_hard":  "vision-only-if-hard",
    "text_llm_pages_escalated": "text-llm-pages-escalated",
    "no_semantic_scholar":  "no-semantic-scholar",
    "semantic_scholar_api_key": "semantic-scholar-api-key",
    "accept_mode":          "accept-mode",
    "ollama_model":         "ollama-model",         # legacy
}


def _has_flag(argv: list[str], flag: str) -> bool:
    """Check if *flag* (e.g. '--mailto') already appears in *argv*."""
    return flag in argv or any(a.startswith(flag + "=") for a in argv)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run papis_import with defaults from a config file",
        add_help=False,
    )
    p.add_argument("--config", default=_CONFIG_PATH)
    p.add_argument("--script", default="",
                   help="Path to papis_import package dir (default: same dir as this script)")
    p.add_argument("--debug-config", action="store_true",
                   help="Print config loading and injected defaults")
    known, rest = p.parse_known_args()

    try:
        cfg = load_json_config(known.config)
    except FileNotFoundError as exc:
        if known.debug_config:
            print(f"[config] {exc}", file=sys.stderr)
        cfg = {}
    except Exception as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    # Build the command: python -m papis_import <injected defaults> <rest>
    script_dir = Path(known.script).resolve() if known.script else Path(__file__).resolve().parent
    cmd = [sys.executable, "-m", "papis_import"]

    injected: list[str] = []
    for cfg_key, flag_name in _CONFIG_MAP.items():
        flag = f"--{flag_name}"
        if _has_flag(rest, flag):
            continue
        value = cfg.get(cfg_key)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
                injected.append(flag)
        else:
            # llm_models may be a JSON list ["a","b",...] or a comma-separated string.
            if cfg_key == "llm_models" and isinstance(value, list):
                value = ",".join(str(v) for v in value)
            cmd.extend([flag, str(value)])
            injected.append(f"{flag}={value}")

    cmd.extend(rest)

    # Make sure the package is importable
    env = os.environ.copy()
    pythonpath_root = script_dir.parent if (script_dir / "__init__.py").exists() else script_dir
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(pythonpath_root) + (os.pathsep + python_path if python_path else "")

    if known.debug_config:
        print(f"[config] path={Path(known.config).expanduser()}")
        print(f"[config] keys={sorted(cfg.keys())}")
        print(f"[config] injected={injected}")
    print("Using PYTHONPATH root:", pythonpath_root)
    print("Running:", subprocess.list2cmdline(cmd))
    return subprocess.call(cmd, env=env, cwd=str(pythonpath_root))


if __name__ == "__main__":
    raise SystemExit(main())
