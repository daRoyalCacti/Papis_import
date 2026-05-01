"""Argument parsing and output-path resolution for the papis_import CLI."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from papis_import.utils import DEFAULT_CACHE_DIR


@dataclass
class OutputPaths:
    auto: Path
    review: Path
    soft: Path
    debug: Path | None
    profile: Path | None


def resolve_output_paths(args: argparse.Namespace) -> OutputPaths:
    auto = Path(os.path.expanduser(args.tsv)).resolve()
    if getattr(args, "review_tsv", ""):
        review = Path(os.path.expanduser(args.review_tsv)).resolve()
    else:
        review = auto.with_name(auto.stem + "_review" + auto.suffix)
    if getattr(args, "soft_tsv", ""):
        soft = Path(os.path.expanduser(args.soft_tsv)).resolve()
    else:
        soft = auto.with_name(auto.stem + "_soft" + auto.suffix)
    debug = (
        Path(os.path.expanduser(args.debug_tsv)).resolve()
        if getattr(args, "debug_tsv", "")
        else None
    )
    profile = (
        Path(os.path.expanduser(args.profile_tsv)).resolve()
        if getattr(args, "profile_tsv", "")
        else None
    )
    return OutputPaths(auto=auto, review=review, soft=soft, debug=debug, profile=profile)


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
    p.add_argument("--soft-tsv", default="",
                   help="Output path for the soft auto-accept TSV (a subset of "
                        "the auto TSV containing rows rescued via local "
                        "corroboration — recommended for spot-checks).  "
                        "Default: auto-derived from --tsv by appending _soft "
                        "before the extension.")
    p.add_argument("--debug-tsv", default="",
                   help="Optional path for a verbose debug JSONL with per-file "
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

    # Vision LLM
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

    # Backward-compat flags
    p.add_argument("--openalex-api-key", default="", help=argparse.SUPPRESS)
    p.add_argument("--crossref-mailto",  default="", help=argparse.SUPPRESS)

    return p.parse_args()
