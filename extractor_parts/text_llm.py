from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from papis_import.core.identifiers import validate_isbn
from papis_import.core.text import clean_text
from papis_import.extractor_parts.common import coerce_str
from papis_import.extractor_parts.llm_pool import ModelPool
from papis_import.http_client import HttpClient
from papis_import.llm_config import LlmRequestConfig, local_llm_response_is_cacheable, remote_llm_response_is_cacheable, text_llm_config
from papis_import.models import Candidate
from papis_import.utils import eprint

# Give up waiting for any model if all are cooling longer than this.
_MAX_ALL_COOLING_WAIT_S = 1800.0


class TextLlmExtractor:
    def __init__(self, args: argparse.Namespace, http: HttpClient) -> None:
        self.args = args
        self.http = http
        self.last_debug: dict[str, Any] = {}
        self._pool: ModelPool | None = None
        self._pool_cfg_models: tuple[str, ...] = ()

    def _get_or_build_pool(self, cfg: LlmRequestConfig) -> ModelPool:
        if self._pool is None or self._pool_cfg_models != cfg.models:
            self._pool = ModelPool(cfg.models, cfg.switch_threshold_s)
            self._pool_cfg_models = cfg.models
        return self._pool

    def candidate(
        self,
        path: Path,
        text: str,
        filename_cand: Candidate | None,
    ) -> list[Candidate]:
        self.last_debug = {
            "text_llm_used": "no",
            "text_llm_status": "not_attempted",
            "text_llm_error": "",
        }
        cfg = text_llm_config(self.args)
        endpoint = cfg.endpoint
        api_key = cfg.api_key
        chars = int(getattr(self.args, "llm_chars", 4000))
        pdf_sha1 = hashlib.sha1(path.read_bytes()).hexdigest()

        if getattr(self.args, "ollama_model", "").strip() and not getattr(self.args, "local_llm", False):
            chars = int(getattr(self.args, "ollama_chars", chars))

        # Determine models list and whether to use the cycling pool.
        if cfg.local:
            models_seq: tuple[str, ...] = (cfg.model,) if cfg.model else ()
            pool: ModelPool | None = None
        else:
            models_seq = cfg.models
            pool = self._get_or_build_pool(cfg) if len(models_seq) > 1 else None

        if not endpoint or not models_seq:
            return []

        self.last_debug.update({
            "text_llm_used": "yes",
            "text_llm_status": "requesting",
            "text_llm_model": models_seq[0],
            "text_llm_chars": str(chars),
        })

        snippet = text[:chars]
        prompt_data = {
            "task": (
                "Extract bibliographic metadata from the PDF text and filename below. "
                "Return only what you can directly observe - do not guess or hallucinate."
            ),
            "rules": [
                "Return empty strings for unknown fields.",
                "Strip 'Author(s):' prefixes.",
                "Ignore download watermarks, page numbers, hashes, and JSTOR stable IDs.",
                "Do not include publisher/series names in the title unless they are part of the real title.",
                "authors must be a JSON array of individual name strings.",
            ],
            "filename": path.name,
            "filename_guess": dataclasses.asdict(filename_cand) if filename_cand else {},
            "text": snippet,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a bibliographic metadata extractor. "
                    "Respond ONLY with a valid JSON object matching the schema: "
                    '{"title": string, "authors": [string], "year": string, '
                    '"doi": string, "isbn": string, "arxiv": string, "notes": string}.'
                ),
            },
            {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=False)},
        ]
        payload: dict[str, Any]
        if cfg.local:
            payload = {
                "model": cfg.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "think": cfg.think,
                "options": {"num_predict": cfg.max_tokens},
            }
        else:
            payload = {
                "model": "",  # filled per iteration below
                "messages": messages,
                "max_tokens": cfg.max_tokens,
                "response_format": {"type": "json_object"},
            }
        estimated_tokens = max(1_500, int(len(json.dumps(messages, ensure_ascii=False)) / 4) + 500)
        min_remaining_tokens = estimated_tokens if cfg.min_remaining_tokens < 0 else cfg.min_remaining_tokens

        extra_headers: dict[str, str] = {}
        if api_key:
            extra_headers["Authorization"] = f"Bearer {api_key}"

        url = endpoint.rstrip("/") + ("/api/chat" if cfg.local else "/chat/completions")

        # --- Model-cycling loop ---
        attempts_record: list[str] = []
        total_elapsed_s = 0.0
        total_retry_sleep_s = 0.0
        total_pacing_s = 0.0
        total_attempt_count = 0
        final_data: dict[str, Any] | None = None
        final_resp: dict[str, Any] | None = None
        chosen = ""
        if pool is not None:
            pool.reset_call_state()

        while True:
            # Pick the next model to try.
            if pool is not None:
                chosen = pool.pick() or ""
                if not chosen:
                    recovery = pool.all_cooling()
                    if recovery is None:
                        break
                    _, wait = recovery
                    if wait > _MAX_ALL_COOLING_WAIT_S:
                        self.last_debug["text_llm_status"] = "all_models_cooling"
                        break
                    eprint(f"[llm-pool] all models cooling; sleeping {wait:.1f}s until {recovery[0]} recovers")
                    time.sleep(wait + 1.0)
                    continue
            else:
                chosen = models_seq[0]

            attempts_record.append(chosen)

            if not cfg.local:
                payload["model"] = chosen

            bucket = f"llm:{chosen}"
            cache_key = f"{pdf_sha1}::{endpoint}::any::{chars}"
            if cfg.local:
                cache_key += f"::max{cfg.max_tokens}::think{cfg.think}"

            resp = self.http.post_json(
                url,
                payload,
                "llm",
                cache_key,
                extra_headers=extra_headers,
                bucket=bucket,
                min_interval=cfg.min_interval,
                min_remaining_tokens=min_remaining_tokens,
                timeout_s=cfg.timeout_s,
                track_tokens=cfg.track_tokens,
                response_cacheable=local_llm_response_is_cacheable if cfg.local else remote_llm_response_is_cacheable,
                max_retry_wait=cfg.switch_threshold_s if pool is not None else 0.0,
                signal_long_wait=pool is not None,
            )

            # Accumulate per-attempt trace stats.
            http_trace = self.http.take_last_request_trace(bucket)
            if http_trace:
                total_elapsed_s += float(http_trace.get("elapsed_s") or 0.0)
                total_retry_sleep_s += float(http_trace.get("retry_sleep_s") or 0.0)
                total_pacing_s += float(http_trace.get("pacing_sleep_s") or 0.0)
                total_attempt_count += int(http_trace.get("attempt_count") or 0)
                if not self.last_debug.get("text_llm_cache_hit") or http_trace.get("cache_hit"):
                    self.last_debug["text_llm_cache_hit"] = "yes" if http_trace.get("cache_hit") else "no"
                self.last_debug["text_llm_http_json"] = json.dumps(
                    http_trace, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                )
                self.last_debug["text_llm_status"] = str(http_trace.get("final_status", "")) or "unknown"
                self.last_debug["text_llm_error"] = str(http_trace.get("final_error", ""))

            # 429 with a long wait — http layer signals us to switch models.
            if (
                pool is not None
                and isinstance(resp, dict)
                and resp.get("_http_error") == 429
                and "_retry_wait_s" in resp
            ):
                wait_s = float(resp.get("_retry_wait_s") or 0.0)
                pool.mark_rate_limited(chosen, wait_s, str(resp.get("_wait_reason") or ""))
                continue

            # Parse the response.
            content: str | None = None
            if isinstance(resp, dict) and "choices" in resp:
                try:
                    content = resp["choices"][0]["message"]["content"]
                except Exception:
                    pass
            elif isinstance(resp, dict) and "message" in resp:
                try:
                    content = resp["message"]["content"]
                except Exception:
                    pass

            if content is None:
                if pool is not None:
                    pool.mark_soft_failure(chosen, "bad_shape")
                    if pool.soft_failures_this_call() < 3:
                        continue
                self.last_debug["text_llm_status"] = "bad_response_shape"
                break

            try:
                data = json.loads(content) if isinstance(content, str) else content
            except Exception as exc:
                err_msg = clean_text(str(exc))
                if pool is not None:
                    pool.mark_soft_failure(chosen, "parse_error")
                    if pool.soft_failures_this_call() < 3:
                        continue
                self.last_debug["text_llm_status"] = "parse_error"
                self.last_debug["text_llm_error"] = err_msg
                break

            title_raw = clean_text(coerce_str(data.get("title")))
            authors_raw = [clean_text(a) for a in (data.get("authors") or []) if clean_text(str(a))]

            if not title_raw and not authors_raw:
                if pool is not None:
                    pool.mark_soft_failure(chosen, "empty")
                    if pool.soft_failures_this_call() < 3:
                        continue
                self.last_debug["text_llm_status"] = "empty_result"
                break

            # Success.
            if pool is not None:
                pool.mark_success(chosen)
            final_data = data
            final_resp = resp
            break

        # --- Post-loop: fill debug fields ---
        self.last_debug["text_llm_model"] = chosen
        self.last_debug["text_llm_models_attempted"] = ",".join(attempts_record)
        self.last_debug["text_llm_model_switches"] = str(max(0, len(attempts_record) - 1))
        self.last_debug["text_llm_http_s"] = f"{total_elapsed_s:.6f}"
        self.last_debug["text_llm_retry_sleep_s"] = f"{total_retry_sleep_s:.6f}"
        self.last_debug["text_llm_pacing_s"] = f"{total_pacing_s:.6f}"
        self.last_debug["text_llm_attempts"] = str(total_attempt_count)

        if pool is not None:
            snap = pool.snapshot()
            if snap:
                self.last_debug["text_llm_pool_state"] = json.dumps(
                    snap, ensure_ascii=True, separators=(",", ":"),
                )

        rem = self.http.remaining_tokens(f"llm:{chosen}") if chosen else None
        if rem is not None:
            self.last_debug["text_llm_tokens_remaining"] = str(rem)

        if final_resp is not None and isinstance(final_resp, dict):
            usage = final_resp.get("usage")
            if isinstance(usage, dict):
                for src, dst in (
                    ("prompt_tokens", "text_llm_prompt_tokens"),
                    ("completion_tokens", "text_llm_completion_tokens"),
                    ("total_tokens", "text_llm_total_tokens"),
                    ("total_time", "text_llm_provider_total_s"),
                    ("queue_time", "text_llm_provider_queue_s"),
                ):
                    if src in usage and usage[src] is not None:
                        self.last_debug[dst] = str(usage[src])

        if final_data is None:
            return []

        self.last_debug["text_llm_status"] = "ok"
        data = final_data
        title = clean_text(coerce_str(data.get("title")))
        authors = [clean_text(a) for a in (data.get("authors") or []) if clean_text(str(a))]
        year = clean_text(coerce_str(data.get("year")))
        doi = clean_text(coerce_str(data.get("doi")))
        isbn = validate_isbn(coerce_str(data.get("isbn")))
        arxiv = clean_text(coerce_str(data.get("arxiv")))
        notes_s = clean_text(coerce_str(data.get("notes")))
        notes = [notes_s] if notes_s else []

        self.last_debug["text_llm_title"] = title
        self.last_debug["text_llm_authors"] = "; ".join(authors)
        self.last_debug["text_llm_year"] = year
        self.last_debug["text_llm_doi"] = doi
        self.last_debug["text_llm_isbn"] = isbn
        self.last_debug["text_llm_arxiv"] = arxiv
        return [
            Candidate(
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                isbn=isbn,
                arxiv=arxiv,
                source=f"llm:{chosen}",
                priority=30,
                notes=notes,
            )
        ]
