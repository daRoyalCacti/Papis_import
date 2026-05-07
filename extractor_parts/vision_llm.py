from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from papis_import.core.identifiers import validate_isbn
from papis_import.core.text import clean_text
from papis_import.extractor_parts.common import coerce_str
from papis_import.http_client import HttpClient
from papis_import.llm_config import local_llm_response_is_cacheable, vision_llm_config
from papis_import.models import Candidate


class VisionLlmExtractor:
    def __init__(self, args: argparse.Namespace, http: HttpClient) -> None:
        self.args = args
        self.http = http

    def render_pages_to_b64(
        self,
        path: Path,
        pages: int,
        dpi: int,
        debug: dict[str, str],
    ) -> list[str]:
        """Render the first *pages* pages of *path* to JPEG and return base64."""
        debug["vision_status"] = "rendering"
        try:
            tmpdir_ctx = tempfile.TemporaryDirectory(prefix="papis_import_vis_")
        except Exception as exc:
            debug["vision_status"] = "tempdir_error"
            debug["vision_error"] = clean_text(str(exc))
            return []

        image_b64s: list[str] = []
        try:
            with tmpdir_ctx as td:
                prefix = Path(td) / "page"
                try:
                    cp = subprocess.run(
                        [
                            "pdftoppm",
                            "-jpeg",
                            "-r",
                            str(dpi),
                            "-f",
                            "1",
                            "-l",
                            str(pages),
                            str(path),
                            str(prefix),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                except FileNotFoundError:
                    debug["vision_status"] = "pdftoppm_missing"
                    return []
                except subprocess.TimeoutExpired:
                    debug["vision_status"] = "render_timeout"
                    return []
                if cp.returncode != 0:
                    debug["vision_status"] = "render_failed"
                    stderr = clean_text(cp.stderr or "")
                    if stderr:
                        debug["vision_error"] = stderr
                    return []
                imgs = sorted(Path(td).glob("page-*.jpg"))[:pages]
                if not imgs:
                    debug["vision_status"] = "no_rendered_images"
                    return []
                for img in imgs:
                    try:
                        image_b64s.append(base64.b64encode(img.read_bytes()).decode("ascii"))
                    except Exception:
                        continue
        except Exception as exc:
            debug["vision_status"] = "render_exception"
            debug["vision_error"] = clean_text(str(exc))
            return []

        if not image_b64s:
            debug["vision_status"] = "no_image_bytes"
        return image_b64s

    def candidate(
        self,
        path: Path,
        filename_cand: Candidate | None,
        is_book: bool = False,
        pages_override: int | None = None,
    ) -> tuple[list[Candidate], dict[str, str]]:
        cfg = vision_llm_config(self.args)
        endpoint = cfg.endpoint
        model = cfg.model
        api_key = cfg.api_key
        pages = pages_override if pages_override is not None else max(1, int(getattr(self.args, "vision_pages", 4)))
        dpi = max(72, int(getattr(self.args, "vision_dpi", 120)))

        debug: dict[str, str] = {
            "vision_status": "not_configured",
            "vision_model": model,
            "vision_pages": str(pages),
            "vision_dpi": str(dpi),
            "vision_escalated": "no",
        }

        if not endpoint or not model:
            return [], debug

        if is_book:
            tiers = [(pages, dpi)]
        else:
            cheap_dpi = min(100, dpi)
            tiers = [(1, cheap_dpi), (pages, dpi)] if pages > 1 else [(pages, dpi)]

        url = endpoint.rstrip("/") + ("/api/chat" if cfg.local else "/chat/completions")
        extra_headers: dict[str, str] = {}
        if api_key:
            extra_headers["Authorization"] = f"Bearer {api_key}"
        filename_guess = dataclasses.asdict(filename_cand) if filename_cand else {}

        for tier_idx, (tier_pages, tier_dpi) in enumerate(tiers):
            is_last_tier = tier_idx == len(tiers) - 1

            image_b64s = self.render_pages_to_b64(path, tier_pages, tier_dpi, debug)
            if not image_b64s:
                return [], debug

            debug["vision_pages"] = str(tier_pages)
            debug["vision_dpi"] = str(tier_dpi)
            debug["vision_images"] = str(len(image_b64s))
            debug["vision_status"] = "requesting"

            user_text = (
                "Extract bibliographic metadata from these PDF pages (typically "
                "the cover, title page, and copyright page of a book or paper). "
                "Return ONLY what you can directly read - do not guess, infer, "
                "or hallucinate. "
                f"Filename (for context only, may be unhelpful): {path.name}. "
                f"Filename-based guess (may be wrong): "
                f"{json.dumps(filename_guess, ensure_ascii=False)}"
            )
            if cfg.local:
                user_content: str | list[dict[str, Any]] = user_text
            else:
                openai_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
                for b64 in image_b64s:
                    openai_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                user_content = openai_content

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a bibliographic metadata extractor for an "
                        "academic reference manager. Respond ONLY with a valid "
                        "JSON object matching the schema: "
                        '{"title": string, "authors": [string], "year": string, '
                        '"doi": string, "isbn": string, "arxiv": string, '
                        '"publisher": string, "notes": string}. '
                        "Return empty strings for fields you cannot read directly. "
                        "The title is the main title of the work, not the series "
                        "or imprint name. Authors should be individual people "
                        "listed on the title page, not editors (unless there is "
                        "no author). ISBNs and DOIs typically appear on the "
                        "copyright page."
                    ),
                },
                {"role": "user", "content": user_content},
            ]
            if cfg.local:
                messages[-1]["images"] = image_b64s
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
            cache_key = (
                f"{path.resolve()}::{path.stat().st_mtime_ns}::"
                f"{endpoint}::{model}::vision::p{tier_pages}::r{tier_dpi}"
            )
            if cfg.local:
                cache_key += f"::max{cfg.max_tokens}::think{cfg.think}"

            try:
                resp = self.http.post_json(
                    url,
                    payload,
                    "vision_llm",
                    cache_key,
                    extra_headers=extra_headers,
                    bucket="vision_llm",
                    min_interval=cfg.min_interval,
                    max_retry_wait=60.0,
                    min_remaining_tokens=cfg.min_remaining_tokens,
                    timeout_s=cfg.timeout_s,
                    track_tokens=cfg.track_tokens,
                    response_cacheable=local_llm_response_is_cacheable if cfg.local else None,
                )
                http_trace = self.http.take_last_request_trace("vision_llm")
                if http_trace:
                    debug["vision_http_json"] = json.dumps(
                        http_trace,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    debug["vision_cache_hit"] = "yes" if http_trace.get("cache_hit") else "no"
                    debug["vision_attempts"] = str(http_trace.get("attempt_count", ""))
                    debug["vision_http_s"] = f"{float(http_trace.get('elapsed_s') or 0.0):.6f}"
                    debug["vision_retry_sleep_s"] = f"{float(http_trace.get('retry_sleep_s') or 0.0):.6f}"
            except Exception as exc:
                debug["vision_status"] = "request_error"
                debug["vision_error"] = clean_text(str(exc))
                return [], debug

            try:
                usage = resp.get("usage") if isinstance(resp, dict) else None
                if isinstance(usage, dict):
                    for src, dst in (
                        ("prompt_tokens", "vision_prompt_tokens"),
                        ("completion_tokens", "vision_completion_tokens"),
                        ("total_tokens", "vision_total_tokens"),
                        ("total_time", "vision_provider_total_s"),
                        ("queue_time", "vision_provider_queue_s"),
                    ):
                        if src in usage and usage[src] is not None:
                            debug[dst] = str(usage[src])
                if isinstance(resp, dict) and "choices" in resp:
                    content = resp["choices"][0]["message"]["content"]
                elif isinstance(resp, dict) and "message" in resp:
                    content = resp["message"]["content"]
                else:
                    debug["vision_status"] = "bad_response_shape"
                    return [], debug
                data = json.loads(content) if isinstance(content, str) else content
            except Exception as exc:
                debug["vision_status"] = "parse_error"
                debug["vision_error"] = clean_text(str(exc))
                return [], debug

            title = clean_text(coerce_str(data.get("title")))
            authors = [clean_text(a) for a in (data.get("authors") or []) if clean_text(str(a))]
            year = clean_text(coerce_str(data.get("year")))
            doi = clean_text(coerce_str(data.get("doi")))
            isbn = validate_isbn(coerce_str(data.get("isbn")))
            arxiv = clean_text(coerce_str(data.get("arxiv")))
            notes_s = clean_text(coerce_str(data.get("notes")))

            if not title and not authors and not (doi or isbn or arxiv):
                if not is_last_tier:
                    debug["vision_escalated"] = "yes"
                    continue
                debug["vision_status"] = "empty_result"
                return [], debug

            notes = [f"vision-extracted from {len(image_b64s)} page(s)"]
            if notes_s:
                notes.append(notes_s)

            debug["vision_status"] = "ok"
            debug["vision_title"] = title
            debug["vision_authors"] = "; ".join(authors)
            debug["vision_year"] = year
            debug["vision_doi"] = doi
            debug["vision_isbn"] = isbn
            debug["vision_arxiv"] = arxiv
            return [
                Candidate(
                    title=title,
                    authors=authors,
                    year=year,
                    doi=doi,
                    isbn=isbn,
                    arxiv=arxiv,
                    source=f"vision_llm:{model}",
                    priority=25,
                    notes=notes,
                )
            ], debug

        debug["vision_status"] = "empty_result"
        return [], debug
