"""Typed wrapper around the per-file debug dict produced by the resolution pipeline."""
from __future__ import annotations

import argparse


class PipelineDebug(dict):
    """A plain dict with a typed constructor that initialises all expected keys.

    Behaves identically to ``dict[str, str]`` at runtime — all existing
    ``debug["key"] = value`` call sites continue to work unchanged.
    The class boundary makes it easy to grep for debug-write sites and
    will carry typed setters once the phase classes are extracted (step 2.5).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            vision_used="no",
            vision_trigger="",
            vision_status="not_attempted",
            vision_error="",
            final_source="",
            grobid_used="yes" if bool(getattr(args, "grobid_url", "")) else "no",
            grobid_title="",
            grobid_authors="",
            grobid_year="",
            vision_model=getattr(args, "vision_llm_model", "") or "",
            vision_pages=str(getattr(args, "vision_pages", "") or ""),
            vision_dpi=str(getattr(args, "vision_dpi", "") or ""),
            vision_title="",
            vision_authors="",
            vision_year="",
            text_llm_used="no",
            text_llm_status="not_attempted",
            text_llm_error="",
            text_llm_model=getattr(args, "llm_model", "") or "",
            text_llm_http_json="",
            text_llm_title="",
            text_llm_authors="",
            text_llm_year="",
            local_best_source="",
            candidate_sources="",
            identifier_dois="",
            identifier_isbns="",
            identifier_arxivs="",
            candidates_json="",
            title_search_queries_json="",
        )
