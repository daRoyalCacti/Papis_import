"""Command-line interface, output formatting, and main loop."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from papis_import.extractors import Extractor
from papis_import.http_client import Cache, HttpClient
from papis_import.models import Metadata, Record, TimingBreakdown
from papis_import.grobid_service import local_grobid_session
from papis_import.pipeline import choose_best_local, resolve
from papis_import.utils import (
    DEFAULT_CACHE_DIR,
    build_tags,
    clean_text,
    command_exists,
    confidence_rank,
    eprint,
    quote_shell,
    should_import,
)

# Short local alias so the OCR retry code reads cleanly.
_confidence_rank = confidence_rank


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Bulk import PDFs into Papis with verified metadata.\n\n"
            "Typical usage:\n"
            "  python -m papis_import --staging ~/Literature_pre_papis --dry-run\n"
            "  python -m papis_import --staging ~/Literature_pre_papis --import --link\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    p.add_argument("--staging", required=True,
                   help="Root directory containing PDFs to import")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",  action="store_true",
                      help="Scan PDFs and write TSV; do not import")
    mode.add_argument("--import",   dest="do_import", action="store_true",
                      help="Actually import into Papis")
    p.add_argument("--link", action="store_true",
                   help="Use 'papis add --link' (keeps PDFs in place)")
    p.add_argument("--tsv",
                   default=os.path.expanduser("~/Documents/papis_import.tsv"),
                   help="Output TSV path for dry-run results")
    p.add_argument("--min-confidence", choices=["low", "medium", "high"], default="high",
                   help="Only import entries at or above this confidence (default: high)")
    p.add_argument("--limit",  type=int, default=0, help="Process at most N PDFs (0 = all)")
    p.add_argument("--offset", type=int, default=0, help="Skip first N PDFs")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--retry-unverified", action="store_true",
                   help="Only re-process PDFs that were not verified in the previous TSV. "
                        "Reads the existing TSV (--tsv path), skips already-verified entries, "
                        "and writes the combined results back. Much faster for iterating.")

    # Extraction
    p.add_argument("--text-pages", type=int, default=2,
                   help="Leading PDF pages to extract for heuristic parsing (default: 2)")
    p.add_argument("--max-search-candidates", type=int, default=6,
                   help="How many local candidates to attempt title-search with")
    p.add_argument("--title-search-timeout", type=float, default=12.0,
                   help="Approximate per-file wall-time budget in seconds for "
                        "title-search fallback (default: 12). In-flight HTTP "
                        "requests are allowed to finish, but no new candidate "
                        "queries are started after the budget is exhausted.")
    p.add_argument("--ocr", action="store_true",
                   help="When a PDF fails verification AND its text looks unreadable "
                        "(low English-word ratio), run ocrmypdf on a temp copy and "
                        "re-resolve once.  Requires ocrmypdf on PATH.")
    p.add_argument("--ocr-text-threshold", type=int, default=150,
                   help=argparse.SUPPRESS)  # kept for backward compat, no longer used
    p.add_argument("--review-tsv", default="",
                   help="Output path for the review TSV (rows that need human "
                        "attention).  Default: auto-derived from --tsv by "
                        "appending _review before the extension.")
    p.add_argument("--debug-tsv", default="",
                   help="Optional path for a verbose debug TSV with per-file "
                        "pipeline diagnostics (vision/GROBID/raw candidates).")
    p.add_argument("--profile-tsv", default="",
                   help="Optional path for a timing TSV. One row is appended "
                        "as each file finishes, with elapsed wall time broken "
                        "down by pipeline phase and per-identifier lookup timings.")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                   help="Directory to cache API responses")
    p.add_argument("--clear-cache-errors", action="store_true",
                   help="Delete stale failure entries from the cache before scanning. "
                        "Run this after a network problem or after upgrading the script "
                        "so that previously-failed arXiv / LLM lookups are retried.")

    # External APIs
    p.add_argument("--mailto", default="",
                   help="Your email — used for the Crossref and OpenAlex polite pools "
                        "(higher rate limits). Sent in the User-Agent header, not stored.")
    p.add_argument("--no-semantic-scholar", action="store_true",
                   help="Disable Semantic Scholar lookups")
    p.add_argument("--semantic-scholar-api-key", default="",
                   help="Optional Semantic Scholar API key (sent as x-api-key)")
    p.add_argument("--google-books-api-key", default="",
                   help="Optional Google Books API key (enables Google Books search)")
    p.add_argument("--grobid-url", default="",
                   help="Base URL of a running GROBID service, e.g. http://localhost:8070. "
                        "GROBID is optional — the script works without it.")
    p.add_argument("--start-local-grobid", action="store_true",
                   help="Start a temporary local GROBID container for this run and stop it at the end. "
                        "If --grobid-url is omitted, uses http://127.0.0.1:<grobid-port>.")
    p.add_argument("--grobid-runtime", choices=["auto", "docker", "podman"], default="auto",
                   help="Container runtime to use for --start-local-grobid (default: auto)")
    p.add_argument("--grobid-image", default="grobid/grobid:0.9.0-full",
                   help="Container image to use for --start-local-grobid")
    p.add_argument("--grobid-port", type=int, default=8070,
                   help="Local port to bind the temporary GROBID container to (default: 8070)")
    p.add_argument("--grobid-start-timeout", type=int, default=180,
                   help="Seconds to wait for a temporary GROBID container to become ready (default: 180)")

    # LLM fallback
    llm = p.add_argument_group(
        "LLM fallback (optional)",
        description=(
            "When title-search fails, an LLM can extract metadata from the PDF text.\n"
            "Supports any OpenAI-compatible endpoint — no local download required.\n\n"
            "Recommended free option (Groq):\n"
            "  1. Get a free API key at https://console.groq.com\n"
            "  2. Run with:\n"
            "       --llm-endpoint https://api.groq.com/openai/v1\n"
            "       --llm-api-key  YOUR_KEY\n"
            "       --llm-model    llama-3.1-8b-instant\n\n"
            "Alternative (local Ollama, no key needed):\n"
            "  --llm-endpoint http://localhost:11434/v1  --llm-model llama3.2\n\n"
            "Alternative (OpenRouter, has free models):\n"
            "  --llm-endpoint https://openrouter.ai/api/v1\n"
            "  --llm-model    meta-llama/llama-3.1-8b-instruct:free\n"
        ),
    )
    llm.add_argument("--llm-endpoint", default="",
                     help="Base URL for an OpenAI-compatible chat completions API")
    llm.add_argument("--llm-api-key",  default="",
                     help="Bearer token for the LLM endpoint")
    llm.add_argument("--llm-model",    default="",
                     help="Model identifier to request")
    llm.add_argument("--llm-chars",    type=int, default=4000,
                     help="Characters of PDF text to send to the LLM (default: 4000)")
    # Backward-compat aliases for people who had --ollama-model in their config
    llm.add_argument("--ollama-model", default="", help=argparse.SUPPRESS)
    llm.add_argument("--ollama-chars", type=int,   default=0,  help=argparse.SUPPRESS)

    # Vision LLM — separate from the text LLM because you typically want a
    # smaller/cheaper text model running always and a vision model running
    # only when you care about hard cases (scanned PDFs, bad filenames).
    vis = p.add_argument_group(
        "Vision LLM (optional, for hard cases)",
        description=(
            "Renders the first few pages of each PDF as JPEGs and sends them\n"
            "to a multimodal LLM for metadata extraction. Essential for:\n"
            "  • Scanned PDFs where pdftotext / OCR produces garbage.\n"
            "  • PDFs with pathological filenames (ass.pdf, BOOK.pdf, 2_4.pdf)\n"
            "    where filename heuristics can't help.\n\n"
            "Requires 'pdftoppm' on PATH (part of poppler-utils, which you\n"
            "already have for pdftotext).\n\n"
            "Recommended (Groq, same API key as the text LLM):\n"
            "  --vision-llm-endpoint https://api.groq.com/openai/v1\n"
            "  --vision-llm-api-key  YOUR_GROQ_KEY\n"
            "  --vision-llm-model    meta-llama/llama-4-scout-17b-16e-instruct\n\n"
            "Alternatives:\n"
            "  Local Ollama (no key):\n"
            "    --vision-llm-endpoint http://localhost:11434/v1\n"
            "    --vision-llm-model    llama3.2-vision:11b  (or qwen2.5vl:7b)\n"
            "  OpenAI:\n"
            "    --vision-llm-endpoint https://api.openai.com/v1\n"
            "    --vision-llm-model    gpt-4o-mini\n\n"
            "Cost note: on Groq, Llama 4 Scout is roughly $0.001 per PDF\n"
            "(≈1500 input + 150 output tokens at 3 pages @ 120 DPI).\n"
            "For 300 PDFs ≈ $0.30. For 3000 PDFs ≈ $3. Negligible.\n"
        ),
    )
    vis.add_argument("--vision-llm-endpoint", default="",
                     help="Base URL for an OpenAI-compatible vision endpoint")
    vis.add_argument("--vision-llm-api-key",  default="",
                     help="Bearer token for the vision endpoint")
    vis.add_argument("--vision-llm-model",    default="",
                     help="Multimodal model identifier "
                          "(e.g. meta-llama/llama-4-scout-17b-16e-instruct)")
    vis.add_argument("--vision-pages",        type=int, default=4,
                     help="Number of leading pages to render and send for books "
                          "and as the escalation ceiling for non-books "
                          "(default: 4 — covers cover + title + copyright + one more)")
    vis.add_argument("--vision-dpi",          type=int, default=120,
                     help="Render DPI for the page images (default: 120). "
                          "Higher = sharper but more tokens. 96 is fine for "
                          "most models; 150 for dense text or small fonts.")
    vis.add_argument("--vision-only-if-hard", action="store_true",
                     help="Only invoke the vision LLM when the text-based "
                          "candidates look weak (no DOI/ISBN/arXiv in text "
                          "or filename, or filename stem is very short / "
                          "obviously generic). Saves ~90%% of vision calls "
                          "on a well-named corpus at the cost of missing a "
                          "few borderline cases.")

    # Backward-compat flag kept from v4 (no-op; OpenAlex is always free now)
    p.add_argument("--openalex-api-key", default="", help=argparse.SUPPRESS)
    # Kept for run_papis_import.py compat
    p.add_argument("--crossref-mailto", default="", help=argparse.SUPPRESS)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def build_papis_command(path: Path, tags: list[str], meta: Metadata, link: bool) -> list[str]:
    cmd = ["papis", "add"]
    if link:
        cmd.append("--link")
    cmd.append(str(path))
    if meta.doi:
        cmd.extend(["--from", "doi", meta.doi])
    elif meta.isbn:
        cmd.extend(["--from", "isbn", meta.isbn])
    elif meta.arxiv:
        cmd.extend(["--from", "arxiv", meta.arxiv])
    if meta.title:
        cmd.extend(["--set", "title", meta.title])
    if meta.authors:
        cmd.extend(["--set", "author", " and ".join(meta.authors)])
    if meta.year:
        cmd.extend(["--set", "year", meta.year])
    if meta.doi:
        cmd.extend(["--set", "doi", meta.doi])
    if meta.isbn:
        cmd.extend(["--set", "isbn", meta.isbn])
    if meta.arxiv:
        cmd.extend(["--set", "arxiv", meta.arxiv])
    if meta.publisher:
        cmd.extend(["--set", "publisher", meta.publisher])
    if tags:
        cmd.extend(["--set", "tags", " ".join(tags)])
    cmd.extend(["--batch", "--no-confirm"])
    return cmd


_TSV_HEADER = [
    "File Path", "Tags", "Source", "Confidence", "Verified",
    "Title", "Authors", "Year", "DOI", "ISBN", "arXiv",
    # New columns introduced alongside the sanity-check pipeline.
    # Added at the end to remain backward-compatible with existing
    # TSVs (review_tsv.py reads by column name via DictReader).
    "Sanity Passed", "Sanity Score", "Auto Safe", "Needs OCR",
    "Notes", "Imported", "Error", "Suggested Command",
    "Vision Used", "Vision Trigger", "Vision Status", "Vision Error", "Final Source",
]

_DEBUG_TSV_HEADER = [
    "File Path", "Tags", "Final Source", "Confidence", "Verified",
    "Sanity Passed", "Sanity Score", "Auto Safe", "Needs OCR",
    "Text LLM Used", "Text LLM Model", "Text LLM Status", "Text LLM Error",
    "Text LLM Cache Hit", "Text LLM Attempts", "Text LLM HTTP s",
    "Text LLM Retry Sleep s", "Text LLM Pacing s",
    "Text LLM Tokens Remaining", "Text LLM Prompt Tokens",
    "Text LLM Completion Tokens", "Text LLM Total Tokens",
    "Text LLM Provider Total s", "Text LLM Provider Queue s",
    "Text LLM Title", "Text LLM Authors", "Text LLM Year",
    "Text LLM DOI", "Text LLM ISBN", "Text LLM arXiv",
    "Text LLM HTTP JSON",
    "Vision Used", "Vision Trigger", "Vision Status", "Vision Error",
    "Vision Cache Hit", "Vision Attempts", "Vision HTTP s",
    "Vision Retry Sleep s", "Vision HTTP JSON",
    "Vision Prompt Tokens", "Vision Completion Tokens", "Vision Total Tokens",
    "Vision Provider Total s", "Vision Provider Queue s",
    "Vision Pacing s", "Vision Tokens Remaining", "Vision Escalated",
    "Vision Model", "Vision Pages", "Vision DPI",
    "Vision Title", "Vision Authors", "Vision Year",
    "GROBID Used", "GROBID Title", "GROBID Authors", "GROBID Year",
    "Local Best Source", "Identifier DOIs", "Identifier ISBNs", "Identifier arXivs",
    "Candidate Sources", "Candidates JSON", "Title Search Queries JSON",
    "Title", "Authors", "Year", "DOI", "ISBN", "arXiv",
    "Notes", "Imported", "Error", "Suggested Command",
]

_PROFILE_TSV_HEADER = [
    "File Path", "Status", "Error", "Final Source", "Confidence", "Verified",
    "File Wall s", "Resolve Total s",
    "Text Extract s", "Embedded Metadata s", "GROBID s",
    "Text LLM s", "Text LLM HTTP s", "Text LLM Retry Sleep s",
    "Text LLM Pacing s", "Text LLM Provider Total s",
    "Text LLM Provider Queue s", "Text LLM Attempts", "Text LLM Cache Hit",
    "Vision LLM s", "Vision HTTP s", "Vision Retry Sleep s",
    "Vision Provider Total s", "Vision Provider Queue s",
    "Vision Attempts", "Vision Cache Hit",
    "Vision Pacing s", "Identifier Lookups s",
    "Title Search s", "OCR Retry s", "OCRMyPDF s", "OCR Reresolve s",
    "Best Local s", "Header Candidate s",
    "Identifier Lookup Count", "Identifier Lookups JSON",
    "Crossref Search s", "Crossref Search Candidates Tried",
    "Crossref Search Matches", "Crossref Search Errors",
    "OpenAlex Search s", "OpenAlex Search Candidates Tried",
    "OpenAlex Search Matches", "OpenAlex Search Errors",
    "Semantic Scholar Search s", "Semantic Scholar Search Candidates Tried",
    "Semantic Scholar Search Matches", "Semantic Scholar Search Errors",
    "OpenLibrary Search s", "OpenLibrary Search Candidates Tried",
    "OpenLibrary Search Matches", "OpenLibrary Search Errors",
    "Google Books Search s", "Google Books Search Candidates Tried",
    "Google Books Search Matches", "Google Books Search Errors",
    "Title Searches JSON",
]


def _fmt_seconds(value: float) -> str:
    return f"{max(0.0, value):.6f}"


def _identifier_timings_json(timing: TimingBreakdown) -> str:
    return json.dumps(
        [asdict(item) for item in timing.identifier_lookups],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _title_search_timings_json(timing: TimingBreakdown) -> str:
    return json.dumps(
        [
            {k: v for k, v in asdict(item).items() if k != "query_traces"}
            for item in timing.title_searches
        ],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _title_search_by_source(timing: TimingBreakdown) -> dict[str, object]:
    return {item.source: item for item in timing.title_searches}


def init_profile_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(_PROFILE_TSV_HEADER)


def append_profile_tsv(path: Path, rec: Record, status: str = "processed") -> None:
    m = rec.result
    t = rec.timing
    d = rec.debug
    title_searches = _title_search_by_source(t)

    def title_cols(source: str) -> list[str]:
        item = title_searches.get(source)
        if item is None:
            return ["0.000000", "0", "0", "0"]
        return [
            _fmt_seconds(item.elapsed_s),
            str(item.candidates_tried),
            str(item.matches_returned),
            str(item.errors),
        ]

    row = [
        str(rec.path),
        status,
        rec.error,
        rec.debug.get("final_source", m.source),
        m.confidence,
        "yes" if m.verified else "no",
        _fmt_seconds(t.file_wall_s),
        _fmt_seconds(t.resolve_total_s),
        _fmt_seconds(t.text_extract_s),
        _fmt_seconds(t.embedded_metadata_s),
        _fmt_seconds(t.grobid_s),
        _fmt_seconds(t.text_llm_s),
        d.get("text_llm_http_s", "0.000000"),
        d.get("text_llm_retry_sleep_s", "0.000000"),
        d.get("text_llm_pacing_s", "0.000000"),
        d.get("text_llm_provider_total_s", ""),
        d.get("text_llm_provider_queue_s", ""),
        d.get("text_llm_attempts", ""),
        d.get("text_llm_cache_hit", ""),
        _fmt_seconds(t.vision_llm_s),
        d.get("vision_http_s", "0.000000"),
        d.get("vision_retry_sleep_s", "0.000000"),
        d.get("vision_provider_total_s", ""),
        d.get("vision_provider_queue_s", ""),
        d.get("vision_attempts", ""),
        d.get("vision_cache_hit", ""),
        _fmt_seconds(t.vision_pacing_s),
        _fmt_seconds(t.identifier_lookups_s),
        _fmt_seconds(t.title_search_s),
        _fmt_seconds(t.ocr_retry_s),
        _fmt_seconds(t.ocrmypdf_s),
        _fmt_seconds(t.ocr_reresolve_s),
        _fmt_seconds(t.best_local_s),
        _fmt_seconds(t.header_candidate_s),
        str(len(t.identifier_lookups)),
        _identifier_timings_json(t),
        *title_cols("crossref"),
        *title_cols("openalex"),
        *title_cols("semanticscholar"),
        *title_cols("openlibrary"),
        *title_cols("google_books"),
        _title_search_timings_json(t),
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(row)


def _record_row(rec: Record) -> list[str]:
    m = rec.result
    return [
        str(rec.path),
        ", ".join(rec.tags),
        m.source,
        m.confidence,
        "yes" if m.verified else "no",
        m.title,
        "; ".join(m.authors),
        m.year,
        m.doi,
        m.isbn,
        m.arxiv,
        "yes" if m.sanity_passed else "no",
        f"{m.sanity_score:.3f}",
        "yes" if m.auto_safe else "no",
        "yes" if m.needs_ocr else "no",
        " | ".join(m.notes),
        "yes" if rec.imported else "no",
        rec.error,
        rec.suggested_command,
        rec.debug.get("vision_used", "no"),
        rec.debug.get("vision_trigger", ""),
        rec.debug.get("vision_status", ""),
        rec.debug.get("vision_error", ""),
        rec.debug.get("final_source", m.source),
    ]


def _debug_row(rec: Record) -> list[str]:
    m = rec.result
    d = rec.debug
    return [
        str(rec.path),
        ", ".join(rec.tags),
        d.get("final_source", m.source),
        m.confidence,
        "yes" if m.verified else "no",
        "yes" if m.sanity_passed else "no",
        f"{m.sanity_score:.3f}",
        "yes" if m.auto_safe else "no",
        "yes" if m.needs_ocr else "no",
        d.get("text_llm_used", "no"),
        d.get("text_llm_model", ""),
        d.get("text_llm_status", ""),
        d.get("text_llm_error", ""),
        d.get("text_llm_cache_hit", ""),
        d.get("text_llm_attempts", ""),
        d.get("text_llm_http_s", "0.000000"),
        d.get("text_llm_retry_sleep_s", "0.000000"),
        d.get("text_llm_pacing_s", "0.000000"),
        d.get("text_llm_tokens_remaining", ""),
        d.get("text_llm_prompt_tokens", ""),
        d.get("text_llm_completion_tokens", ""),
        d.get("text_llm_total_tokens", ""),
        d.get("text_llm_provider_total_s", ""),
        d.get("text_llm_provider_queue_s", ""),
        d.get("text_llm_title", ""),
        d.get("text_llm_authors", ""),
        d.get("text_llm_year", ""),
        d.get("text_llm_doi", ""),
        d.get("text_llm_isbn", ""),
        d.get("text_llm_arxiv", ""),
        d.get("text_llm_http_json", ""),
        d.get("vision_used", "no"),
        d.get("vision_trigger", ""),
        d.get("vision_status", ""),
        d.get("vision_error", ""),
        d.get("vision_cache_hit", ""),
        d.get("vision_attempts", ""),
        d.get("vision_http_s", "0.000000"),
        d.get("vision_retry_sleep_s", "0.000000"),
        d.get("vision_http_json", ""),
        d.get("vision_prompt_tokens", ""),
        d.get("vision_completion_tokens", ""),
        d.get("vision_total_tokens", ""),
        d.get("vision_provider_total_s", ""),
        d.get("vision_provider_queue_s", ""),
        d.get("vision_pacing_s", "0.000000"),
        d.get("vision_tokens_remaining", ""),
        d.get("vision_escalated", "no"),
        d.get("vision_model", ""),
        d.get("vision_pages", ""),
        d.get("vision_dpi", ""),
        d.get("vision_title", ""),
        d.get("vision_authors", ""),
        d.get("vision_year", ""),
        d.get("grobid_used", "no"),
        d.get("grobid_title", ""),
        d.get("grobid_authors", ""),
        d.get("grobid_year", ""),
        d.get("local_best_source", ""),
        d.get("identifier_dois", ""),
        d.get("identifier_isbns", ""),
        d.get("identifier_arxivs", ""),
        d.get("candidate_sources", ""),
        d.get("candidates_json", ""),
        d.get("title_search_queries_json", ""),
        m.title,
        "; ".join(m.authors),
        m.year,
        m.doi,
        m.isbn,
        m.arxiv,
        " | ".join(m.notes),
        "yes" if rec.imported else "no",
        rec.error,
        rec.suggested_command,
    ]


def write_debug_tsv(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="	")
        w.writerow(_DEBUG_TSV_HEADER)
        for rec in records:
            w.writerow(_debug_row(rec))


def init_result_tsvs(auto_path: Path, review_path: Path) -> None:
    for p in (auto_path, review_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(_TSV_HEADER)


def append_result_tsv(auto_path: Path, review_path: Path, rec: Record) -> None:
    path = auto_path if rec.result.auto_safe else review_path
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(_record_row(rec))


def init_debug_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(_DEBUG_TSV_HEADER)


def append_debug_tsv(path: Path, rec: Record) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(_debug_row(rec))


def write_two_tsvs(
    auto_path: Path,
    review_path: Path,
    records: list[Record],
) -> tuple[int, int]:
    """Split *records* into auto_safe / needs-review and write each to its own TSV.

    Returns (auto_count, review_count).

    The auto TSV contains only rows where Metadata.auto_safe is True — i.e.
    verified externally AND passed the sanity check AND confidence is high.
    These are safe to `papis add` without human review.

    The review TSV contains everything else: unverified rows, rows that
    failed the sanity check (likely false positives), low-confidence guesses,
    and rows flagged needs_ocr.  The TSV keeps the full set of candidate
    metadata in the Notes column so manual fix-up is quick.
    """
    auto_records   = [r for r in records if r.result.auto_safe]
    review_records = [r for r in records if not r.result.auto_safe]
    for p, recs in ((auto_path, auto_records), (review_path, review_records)):
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(_TSV_HEADER)
            for rec in recs:
                w.writerow(_record_row(rec))
    return len(auto_records), len(review_records)


def write_tsv(path: Path, records: list[Record]) -> None:
    """Legacy single-TSV writer — kept for callers that haven't migrated.

    Prefer write_two_tsvs in new code.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(_TSV_HEADER)
        for rec in records:
            w.writerow(_record_row(rec))


def collect_pdfs(staging_dir: Path) -> list[Path]:
    return sorted(p for p in staging_dir.rglob("*.pdf") if p.is_file())


def run_ocr_retry(
    path: Path,
    extractor: Extractor,
    verbose: bool = False,
) -> tuple[Metadata | None, str, TimingBreakdown]:
    """Run ocrmypdf on a **temp copy** of *path* and re-resolve.

    Returns (new_meta_or_None, status_string). The status string is one of:
      "ocrmypdf-missing"  — the binary isn't on PATH
      "ocrmypdf-failed:…" — ocrmypdf returned non-zero; suffix has stderr tail
      "ocrmypdf-timeout"  — ocrmypdf didn't finish within the 10-min budget
      "no-output"         — ocrmypdf exited 0 but didn't write the output file
      "resolve-raised:…"  — resolve() threw on the OCR'd file
      "ok"                — OCR produced new metadata (caller decides whether to accept)
    new_meta is the full Metadata from re-resolving the OCR'd PDF, or None
    on any failure.  The returned TimingBreakdown contains only OCR retry
    timings plus any timing collected during the re-resolve.

    Non-destructive: the original PDF is never modified.  The temp
    directory is cleaned up automatically.
    """
    import tempfile
    timing = TimingBreakdown()
    retry_started = perf_counter()
    if not command_exists("ocrmypdf"):
        timing.ocr_retry_s = perf_counter() - retry_started
        return None, "ocrmypdf-missing", timing
    try:
        with tempfile.TemporaryDirectory(prefix="papis_import_ocr_") as td:
            ocr_path = Path(td) / path.name
            # --force-ocr overwrites any existing (garbled) text layer.
            # --optimize 1 keeps the temp file small.
            # --output-type pdf avoids PDF/A conversion (faster, fewer deps).
            try:
                ocr_started = perf_counter()
                cp = subprocess.run(
                    ["ocrmypdf",
                     "--force-ocr", "--optimize", "1",
                     "--output-type", "pdf",
                     str(path), str(ocr_path)],
                    check=False, capture_output=True, text=True,
                    timeout=600,  # 10 min ceiling per file
                )
                timing.ocrmypdf_s = perf_counter() - ocr_started
            except subprocess.TimeoutExpired:
                timing.ocrmypdf_s = perf_counter() - ocr_started
                timing.ocr_retry_s = perf_counter() - retry_started
                return None, "ocrmypdf-timeout", timing
            if cp.returncode != 0:
                stderr_tail = (cp.stderr or cp.stdout or "").strip().splitlines()
                stderr_tail = stderr_tail[-1] if stderr_tail else "(no output)"
                # ocrmypdf stderr is verbose; keep the last line only
                timing.ocr_retry_s = perf_counter() - retry_started
                return None, f"ocrmypdf-failed: {stderr_tail[:200]}", timing
            if not ocr_path.exists():
                timing.ocr_retry_s = perf_counter() - retry_started
                return None, "no-output", timing
            try:
                resolve_started = perf_counter()
                new_meta, _cands, _text, _debug, resolve_timing = resolve(
                    ocr_path, extractor, skip_vision=True
                )
                timing.ocr_reresolve_s = perf_counter() - resolve_started
                timing.identifier_lookups_s = resolve_timing.identifier_lookups_s
                timing.identifier_lookups = resolve_timing.identifier_lookups
            except Exception as exc:
                timing.ocr_reresolve_s = perf_counter() - resolve_started
                timing.ocr_retry_s = perf_counter() - retry_started
                return None, f"resolve-raised: {exc}", timing
            timing.ocr_retry_s = perf_counter() - retry_started
            return new_meta, "ok", timing
    except Exception as exc:
        timing.ocr_retry_s = perf_counter() - retry_started
        return None, f"retry-error: {exc}", timing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_previous_tsv(*tsv_paths: Path) -> tuple[dict[str, Record], int]:
    """Load trusted rows from one or more previous TSVs.

    Returns (skip_dict, legacy_count) where:
      skip_dict   = {file_path_str: Record}  — rows to skip on this run
      legacy_count = number of rows seen that were Verified=yes but lacked
                     an Auto Safe column (i.e. came from a pre-sanity-check
                     TSV and therefore need re-processing on first run).

    Skip criterion: a row is skipped only if it was explicitly marked
    `Auto Safe = yes` in a previous new-format run.  Rows from legacy
    TSVs (no Auto Safe column) and rows from the review TSV (Auto Safe = no)
    are ALWAYS re-processed, so that the new sanity check and pipeline
    changes get applied.  This means upgrading does a one-time full
    re-run — subsequent --retry-unverified calls are cheap again.

    Missing files are silently ignored.
    """
    skip: dict[str, Record] = {}
    legacy_count = 0
    for tsv_path in tsv_paths:
        if not tsv_path or not tsv_path.exists():
            continue
        with tsv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            has_auto_safe = "Auto Safe" in fieldnames
            for row in reader:
                # Track legacy rows (verified but no Auto Safe column) so the
                # caller can print a "re-processing N legacy rows" notice.
                if not has_auto_safe and row.get("Verified", "").strip().lower() == "yes":
                    legacy_count += 1
                    continue
                if row.get("Auto Safe", "").strip().lower() != "yes":
                    continue
                path_str = row.get("File Path", "").strip()
                if not path_str:
                    continue
                try:
                    sanity_score = float(row.get("Sanity Score", "0") or "0")
                except ValueError:
                    sanity_score = 0.0
                meta = Metadata(
                    title=row.get("Title", ""),
                    authors=[a.strip() for a in row.get("Authors", "").split(";") if a.strip()],
                    year=row.get("Year", ""),
                    doi=row.get("DOI", ""),
                    isbn=row.get("ISBN", ""),
                    arxiv=row.get("arXiv", ""),
                    source=row.get("Source", ""),
                    confidence=row.get("Confidence", "low"),
                    verified=row.get("Verified", "").strip().lower() == "yes",
                    sanity_passed=row.get("Sanity Passed", "").strip().lower() == "yes",
                    sanity_score=sanity_score,
                    auto_safe=True,
                    needs_ocr=row.get("Needs OCR", "").strip().lower() == "yes",
                    notes=[n.strip() for n in row.get("Notes", "").split("|") if n.strip()],
                )
                tags = [t.strip() for t in row.get("Tags", "").split(",") if t.strip()]
                rec = Record(
                    path=Path(path_str),
                    tags=tags,
                    result=meta,
                    suggested_command=row.get("Suggested Command", ""),
                    imported=row.get("Imported", "").strip().lower() == "yes",
                    error=row.get("Error", ""),
                )
                # First TSV wins — auto TSV is passed first, so a later
                # review-TSV entry for the same path won't override it.
                skip.setdefault(path_str, rec)
    return skip, legacy_count


def main() -> int:
    args = parse_args()

    # Normalise legacy flags
    if args.crossref_mailto and not args.mailto:
        args.mailto = args.crossref_mailto

    staging_dir = Path(os.path.expanduser(args.staging)).resolve()
    # --tsv now names the AUTO TSV (safe to import without review).
    # Default review TSV is derived by inserting _review before the extension,
    # unless --review-tsv overrides.
    auto_tsv_path = Path(os.path.expanduser(args.tsv)).resolve()
    if getattr(args, "review_tsv", ""):
        review_tsv_path = Path(os.path.expanduser(args.review_tsv)).resolve()
    else:
        review_tsv_path = auto_tsv_path.with_name(
            auto_tsv_path.stem + "_review" + auto_tsv_path.suffix
        )
    debug_tsv_path = Path(os.path.expanduser(args.debug_tsv)).resolve() if getattr(args, "debug_tsv", "") else None
    profile_tsv_path = Path(os.path.expanduser(args.profile_tsv)).resolve() if getattr(args, "profile_tsv", "") else None

    if not staging_dir.exists():
        eprint(f"[error] staging directory does not exist: {staging_dir}")
        return 2
    if not args.dry_run and not args.do_import:
        args.dry_run = True

    files = collect_pdfs(staging_dir)
    if args.offset:
        files = files[args.offset:]
    if args.limit > 0:
        files = files[: args.limit]

    init_result_tsvs(auto_tsv_path, review_tsv_path)
    if debug_tsv_path is not None:
        init_debug_tsv(debug_tsv_path)
    if profile_tsv_path is not None:
        init_profile_tsv(profile_tsv_path)


    # --retry-unverified: load prior auto-safe rows from BOTH TSVs; skip those.
    # Rows from legacy TSVs (no Auto Safe column) and rows in the review TSV
    # are NOT skipped — they get re-processed so the new sanity check and
    # pipeline changes are actually applied.
    prev_verified: dict[str, Record] = {}
    if getattr(args, "retry_unverified", False):
        prev_verified, legacy_count = _load_previous_tsv(auto_tsv_path, review_tsv_path)
        retry_count = sum(1 for f in files if str(f) not in prev_verified)
        skip_count  = sum(1 for f in files if str(f) in prev_verified)
        print(f"Retry mode: {skip_count} already auto-safe, {retry_count} to (re-)process")
        if legacy_count:
            print(f"            {legacy_count} previously-verified row(s) lack sanity data "
                  f"— re-processing to apply the new checks")

    # --ocr requested but ocrmypdf missing → warn up front, not silently.
    if args.ocr and not command_exists("ocrmypdf"):
        eprint("[warning] --ocr was requested but 'ocrmypdf' is not on PATH. "
               "OCR retry will be skipped. Install ocrmypdf (e.g. `sudo pacman -S "
               "ocrmypdf` or `pip install --user ocrmypdf`) to enable it.")

    cache     = Cache(args.cache_dir)
    if getattr(args, "clear_cache_errors", False):
        removed = cache.clear_errors()
        print(f"Cleared {removed} stale error entries from cache.")

    http      = HttpClient(cache=cache, mailto=args.mailto, verbose=args.verbose)
    records: list[Record] = []

    # OCR retry stats
    ocr_attempted = 0
    ocr_recovered = 0

    total = len(files)
    print(f"Scanning {total} PDF(s) in {staging_dir}")

    with local_grobid_session(args) as grobid_session:
        if grobid_session.url:
            if grobid_session.started_here:
                print(f"GROBID    : started local service at {grobid_session.url}")
            elif grobid_session.url == getattr(args, "grobid_url", ""):
                print(f"GROBID    : using {grobid_session.url}")
        extractor = Extractor(args, http)

        for idx, path in enumerate(files, start=1):
            # Skip already-verified entries in retry mode
            if str(path) in prev_verified:
                rec = prev_verified[str(path)]
                records.append(rec)
                append_result_tsv(auto_tsv_path, review_tsv_path, rec)
                if debug_tsv_path is not None:
                    append_debug_tsv(debug_tsv_path, rec)
                if profile_tsv_path is not None:
                    append_profile_tsv(profile_tsv_path, rec, status="skipped: retry-unverified")
                continue

            file_started = perf_counter()
            if args.verbose:
                print(f"[{idx}/{total}] {path.name}")
            elif idx % 25 == 0:
                print(f"  … {idx}/{total}")
            tags = build_tags(staging_dir, path)
            try:
                meta, candidates, _, debug, timing = resolve(path, extractor)

                # OCR retry: only when the pipeline flagged the text as
                # unreadable AND we couldn't verify externally AND --ocr.
                # Preemptive OCR was removed — it wasted compute on the
                # 90%+ of files that didn't need it.
                if (args.ocr
                        and meta.needs_ocr
                        and not meta.verified
                        and command_exists("ocrmypdf")):
                    ocr_attempted += 1
                    if args.verbose:
                        print(f"  [ocr] {path.name}: retrying with ocrmypdf…")
                    new_meta, status, ocr_timing = run_ocr_retry(path, extractor, verbose=args.verbose)
                    timing.ocrmypdf_s += ocr_timing.ocrmypdf_s
                    timing.ocr_reresolve_s += ocr_timing.ocr_reresolve_s
                    timing.ocr_retry_s += ocr_timing.ocr_retry_s
                    timing.identifier_lookups_s += ocr_timing.identifier_lookups_s
                    timing.identifier_lookups.extend(ocr_timing.identifier_lookups)
                    if new_meta is None:
                        if args.verbose:
                            print(f"  [ocr] {path.name}: FAILED ({status})")
                    else:
                        # Accept when the retry produces a meaningfully better
                        # result.  The original check was too strict (required
                        # high-confidence or large sanity jump); in practice
                        # OCR often yields an LLM-extracted title+author that
                        # is correct but unverified — strictly better than a
                        # "filename_title_only" fallback with no metadata.
                        better_flags: list[str] = []
                        if new_meta.verified and not meta.verified:
                            better_flags.append("verified")
                        if _confidence_rank(new_meta.confidence) > _confidence_rank(meta.confidence):
                            better_flags.append(
                                f"confidence {meta.confidence}→{new_meta.confidence}"
                            )
                        if new_meta.sanity_score > meta.sanity_score + 0.01:
                            better_flags.append(
                                f"sanity {meta.sanity_score:.2f}→{new_meta.sanity_score:.2f}"
                            )
                        # New identifier found via OCR — very valuable signal.
                        for attr in ("doi", "isbn", "arxiv"):
                            if getattr(new_meta, attr) and not getattr(meta, attr):
                                better_flags.append(f"new {attr}")
                        # New title/authors found when we had nothing.
                        if new_meta.title and not meta.title:
                            better_flags.append("new title")
                        if new_meta.authors and not meta.authors:
                            better_flags.append("new authors")

                        if better_flags:
                            meta = new_meta
                            reason = ", ".join(better_flags)
                            meta.notes.append(
                                f"OCR retry recovered metadata ({reason}) — "
                                "consider running ocrmypdf to make the original "
                                "PDF searchable"
                            )
                            ocr_recovered += 1
                            if args.verbose:
                                print(f"  [ocr] {path.name}: RECOVERED ({reason})")
                        elif args.verbose:
                            print(f"  [ocr] {path.name}: no improvement — kept original")

                cmd      = build_papis_command(path, tags, meta, args.link)
                imported = False
                err      = ""
                if args.do_import and meta.auto_safe and should_import(meta.confidence, args.min_confidence):
                    if not command_exists("papis"):
                        err = "papis executable not found"
                    else:
                        cp       = subprocess.run(cmd, capture_output=True, text=True, check=False)
                        imported = cp.returncode == 0
                        if not imported:
                            err = clean_text(cp.stderr or cp.stdout)
                elif args.do_import:
                    if not meta.auto_safe:
                        err = "skipped: not auto-safe; manual review required"
                    else:
                        err = f"skipped: confidence {meta.confidence} below threshold {args.min_confidence}"
                rec = Record(
                    path=path, tags=tags, result=meta,
                    suggested_command=" ".join(quote_shell(x) for x in cmd),
                    imported=imported, error=err, debug=debug, timing=timing,
                )
                rec.timing.file_wall_s = perf_counter() - file_started
                records.append(rec)
                append_result_tsv(auto_tsv_path, review_tsv_path, rec)
                if debug_tsv_path is not None:
                    append_debug_tsv(debug_tsv_path, rec)
                if profile_tsv_path is not None:
                    append_profile_tsv(profile_tsv_path, rec)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                timing = TimingBreakdown(file_wall_s=perf_counter() - file_started)
                if args.verbose:
                    traceback.print_exc()
                eprint(f"[error] {path.name}: {exc}")
                rec = Record(
                    path=path, tags=tags,
                    result=Metadata(source="error", confidence="low",
                                    notes=["exception during processing"]),
                    suggested_command="",
                    imported=False,
                    error=clean_text(str(exc)),
                    debug={"final_source": "error", "vision_used": "no", "vision_status": "exception", "vision_error": clean_text(str(exc))},
                    timing=timing,
                )
                records.append(rec)
                append_result_tsv(auto_tsv_path, review_tsv_path, rec)
                if debug_tsv_path is not None:
                    append_debug_tsv(debug_tsv_path, rec)
                if profile_tsv_path is not None:
                    append_profile_tsv(profile_tsv_path, rec, status="error")

    # ---- Output summary: result/debug TSVs were streamed per completed file ----
    auto_count = sum(1 for r in records if r.result.auto_safe)
    review_count = len(records) - auto_count

    # Summary
    high  = sum(1 for r in records if r.result.confidence == "high")
    med   = sum(1 for r in records if r.result.confidence == "medium")
    low   = sum(1 for r in records if r.result.confidence == "low")
    ver   = sum(1 for r in records if r.result.verified)
    sanity_pass = sum(1 for r in records if r.result.sanity_passed)
    ocr_flagged = sum(1 for r in records if r.result.needs_ocr)
    imp   = sum(1 for r in records if r.imported)
    print("-" * 60)
    print(f"Done. {total} PDFs scanned.")
    print(f"Confidence   : high={high}  medium={med}  low={low}")
    print(f"Verified     : {ver}/{total}")
    print(f"Sanity passed: {sanity_pass}/{total}")
    if ocr_attempted or ocr_flagged:
        print(f"OCR          : flagged={ocr_flagged}  retried={ocr_attempted}  recovered={ocr_recovered}")
    print(f"Auto-safe    : {auto_count}/{total}  (→ {auto_tsv_path.name})")
    print(f"Needs review : {review_count}/{total}  (→ {review_tsv_path.name})")
    if args.do_import:
        print(f"Imported     : {imp}/{total}")
    print(f"TSV (auto)   : {auto_tsv_path}")
    print(f"TSV (review) : {review_tsv_path}")
    if debug_tsv_path is not None:
        print(f"TSV (debug)  : {debug_tsv_path}")
    if profile_tsv_path is not None:
        print(f"TSV (profile): {profile_tsv_path}")
    print(f"Cache        : {Path(args.cache_dir).resolve()}")
    return 0
