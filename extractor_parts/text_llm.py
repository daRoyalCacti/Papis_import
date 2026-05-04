from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from papis_import.core.identifiers import validate_isbn
from papis_import.core.text import clean_text
from papis_import.extractor_parts.common import coerce_str
from papis_import.http_client import HttpClient
from papis_import.llm_config import local_llm_response_is_cacheable, text_llm_config
from papis_import.models import Candidate


class TextLlmExtractor:
    def __init__(self, args: argparse.Namespace, http: HttpClient) -> None:
        self.args = args
        self.http = http
        self.last_debug: dict[str, Any] = {}

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
        model = cfg.model
        api_key = cfg.api_key
        chars = int(getattr(self.args, "llm_chars", 4000))

        if getattr(self.args, "ollama_model", "").strip() and not getattr(self.args, "local_llm", False):
            chars = int(getattr(self.args, "ollama_chars", chars))

        if not endpoint or not model:
            return []
        self.last_debug.update({
            "text_llm_used": "yes",
            "text_llm_status": "requesting",
            "text_llm_model": model,
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
                "model": model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "think": cfg.think,
                "options": {"num_predict": cfg.max_tokens},
            }
        else:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": cfg.max_tokens,
                "response_format": {"type": "json_object"},
            }
        estimated_tokens = max(1_500, int(len(json.dumps(messages, ensure_ascii=False)) / 4) + 500)
        min_remaining_tokens = estimated_tokens if cfg.min_remaining_tokens < 0 else cfg.min_remaining_tokens
        cache_key = f"{path.resolve()}::{path.stat().st_mtime_ns}::{endpoint}::{model}::{chars}"
        if cfg.local:
            cache_key += f"::max{cfg.max_tokens}::think{cfg.think}"

        extra_headers: dict[str, str] = {}
        if api_key:
            extra_headers["Authorization"] = f"Bearer {api_key}"

        url = endpoint.rstrip("/") + ("/api/chat" if cfg.local else "/chat/completions")
        resp = self.http.post_json(
            url,
            payload,
            "llm",
            cache_key,
            extra_headers=extra_headers,
            bucket="llm",
            min_interval=cfg.min_interval,
            min_remaining_tokens=min_remaining_tokens,
            timeout_s=cfg.timeout_s,
            track_tokens=cfg.track_tokens,
            response_cacheable=local_llm_response_is_cacheable if cfg.local else None,
        )
        http_trace = self.http.take_last_request_trace("llm")
        if http_trace:
            self.last_debug["text_llm_http_json"] = json.dumps(
                http_trace,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            self.last_debug["text_llm_cache_hit"] = "yes" if http_trace.get("cache_hit") else "no"
            self.last_debug["text_llm_attempts"] = str(http_trace.get("attempt_count", ""))
            self.last_debug["text_llm_http_s"] = f"{float(http_trace.get('elapsed_s') or 0.0):.6f}"
            self.last_debug["text_llm_retry_sleep_s"] = f"{float(http_trace.get('retry_sleep_s') or 0.0):.6f}"
            self.last_debug["text_llm_pacing_s"] = f"{float(http_trace.get('pacing_sleep_s') or 0.0):.6f}"
            self.last_debug["text_llm_status"] = str(http_trace.get("final_status", "")) or "unknown"
            self.last_debug["text_llm_error"] = str(http_trace.get("final_error", ""))
        rem = self.http.remaining_tokens("llm")
        if rem is not None:
            self.last_debug["text_llm_tokens_remaining"] = str(rem)

        try:
            usage = resp.get("usage") if isinstance(resp, dict) else None
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
            if isinstance(resp, dict) and "choices" in resp:
                content = resp["choices"][0]["message"]["content"]
            elif isinstance(resp, dict) and "message" in resp:
                content = resp["message"]["content"]
            else:
                self.last_debug["text_llm_status"] = "bad_response_shape"
                return []
            data = json.loads(content) if isinstance(content, str) else content
        except Exception as exc:
            self.last_debug["text_llm_status"] = "parse_error"
            self.last_debug["text_llm_error"] = clean_text(str(exc))
            return []

        title = clean_text(coerce_str(data.get("title")))
        authors = [clean_text(a) for a in (data.get("authors") or []) if clean_text(str(a))]
        year = clean_text(coerce_str(data.get("year")))
        doi = clean_text(coerce_str(data.get("doi")))
        isbn = validate_isbn(coerce_str(data.get("isbn")))
        arxiv = clean_text(coerce_str(data.get("arxiv")))
        notes_s = clean_text(coerce_str(data.get("notes")))
        notes = [notes_s] if notes_s else []

        if not title and not authors:
            self.last_debug["text_llm_status"] = "empty_result"
            return []
        self.last_debug["text_llm_status"] = "ok"
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
                source=f"llm:{model}",
                priority=30,
                notes=notes,
            )
        ]
