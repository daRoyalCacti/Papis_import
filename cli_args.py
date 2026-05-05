"""Argument parsing and output-path resolution for the papis_import CLI."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from papis_import.utils import DEFAULT_CACHE_DIR
from papis_import.llm_config import DEFAULT_LOCAL_LLM_MODEL, DEFAULT_OLLAMA_HOST


@dataclass
class OutputPaths:
    auto: Path
    review: Path
    soft: Path
    debug: Path | None
    profile: Path | None
    live_status: Path | None


def _require_ext(path: Path, ext: str, flag: str) -> None:
    if path.suffix.lower() != ext:
        raise ValueError(f"--{flag}: expected a {ext!r} file, got {path.name!r}")


def resolve_output_paths(args: argparse.Namespace) -> OutputPaths:
    auto = Path(os.path.expanduser(args.tsv)).resolve()
    _require_ext(auto, ".tsv", "tsv")
    if getattr(args, "review_tsv", ""):
        review = Path(os.path.expanduser(args.review_tsv)).resolve()
        _require_ext(review, ".tsv", "review-tsv")
    else:
        review = auto.with_name(auto.stem + "_review" + auto.suffix)
    if getattr(args, "soft_tsv", ""):
        soft = Path(os.path.expanduser(args.soft_tsv)).resolve()
        _require_ext(soft, ".tsv", "soft-tsv")
    else:
        soft = auto.with_name(auto.stem + "_soft" + auto.suffix)
    if getattr(args, "debug_jsonl", ""):
        debug = Path(os.path.expanduser(args.debug_jsonl)).resolve()
        _require_ext(debug, ".jsonl", "debug-jsonl")
    else:
        debug = None
    if getattr(args, "profile_tsv", ""):
        profile = Path(os.path.expanduser(args.profile_tsv)).resolve()
        _require_ext(profile, ".tsv", "profile-tsv")
    else:
        profile = None
    if getattr(args, "live_status_json", ""):
        live_status = Path(os.path.expanduser(args.live_status_json)).resolve()
    else:
        live_status = None
    return OutputPaths(
        auto=auto,
        review=review,
        soft=soft,
        debug=debug,
        profile=profile,
        live_status=live_status,
    )


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
    p.add_argument("--accept-mode", choices=["default", "safe"], default="default",
                   dest="accept_mode",
                   help="Controls how strictly identifier-based results (arXiv, DOI, ISBN) "
                        "are accepted.  'default': trust the identifier if it passes the "
                        "sanity check (fast, matches the original pipeline behaviour).  "
                        "'safe': also require at least one independent local extractor "
                        "(GROBID, text_header, structured filename, LLM, vision) to agree "
                        "on both title and author before accepting — prevents citations "
                        "mistaken for the document itself, at the cost of more API calls "
                        "and more entries routed to review. (default: default)")
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
    p.add_argument("--debug-jsonl", default="",
                   help="Optional path for a verbose debug JSONL with per-file "
                        "pipeline diagnostics (vision/GROBID/raw candidates). "
                        "File must end in .jsonl.")
    p.add_argument("--profile-tsv", default="",
                   help="Optional path for a timing TSV. One row is appended "
                        "as each file finishes, with elapsed wall time broken "
                        "down by pipeline phase and per-identifier lookup timings.")
    p.add_argument("--live-status-json", default="",
                   help="Optional path for mutable live status JSON. The file "
                        "is atomically rewritten as the current PDF or HTTP "
                        "rate-limit state changes.")
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
            "Local Ollama, no key needed:\n"
            "  --local-llm --local-llm-model qwen3-vl:8b\n\n"
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
    llm.add_argument("--llm-request-timeout", type=float, default=90.0,
                     help="HTTP timeout in seconds for remote text LLM requests (default: 90)")
    llm.add_argument("--llm-models", default="",
                     help=(
                         "Comma-separated list of remote LLM models to cycle through when "
                         "earlier models hit rate limits or return empty results. "
                         "Overrides --llm-model when set. "
                         "Recommended (strongest first): "
                         "llama-3.3-70b-versatile,openai/gpt-oss-120b,qwen/qwen3-32b,"
                         "meta-llama/llama-4-scout-17b-16e-instruct,"
                         "openai/gpt-oss-20b,llama-3.1-8b-instant"
                     ))
    llm.add_argument("--text-llm-pages-escalated", type=int, default=5,
                     dest="text_llm_pages_escalated",
                     help="Leading PDF pages to extract for the text-LLM escalation retry "
                          "(default: 5). Only used when the normal attempt returns nothing. "
                          "Increases pdftotext coverage on books where the title page is "
                          "past page 2 without paying extra tokens on the working majority.")
    llm.add_argument("--llm-switch-threshold", type=float, default=120.0,
                     help=(
                         "Switch to the next model in --llm-models when a 429 retry "
                         "would sleep longer than this many seconds (default: 120). "
                         "Set to 0 to disable cycling."
                     ))
    llm.add_argument("--local-llm", action="store_true",
                     help="Use local Ollama for both text and vision LLM extraction")
    llm.add_argument("--local-llm-model", default=DEFAULT_LOCAL_LLM_MODEL,
                     help=f"Ollama model to use with --local-llm (default: {DEFAULT_LOCAL_LLM_MODEL})")
    llm.add_argument("--local-llm-num-predict", type=int, default=2048,
                     help="Maximum output tokens for local Ollama LLM calls (default: 2048)")
    llm.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST,
                     help=f"Ollama base URL for --local-llm (default: {DEFAULT_OLLAMA_HOST})")
    llm.add_argument("--ollama-start-timeout", type=int, default=60,
                     help="Seconds to wait for an auto-started Ollama service (default: 60)")
    llm.add_argument("--ollama-request-timeout", type=float, default=300.0,
                     help="HTTP timeout in seconds for local Ollama requests (default: 300)")
    llm.add_argument("--ollama-pull-missing", action="store_true",
                     help="Run 'ollama pull <model>' when the requested local model is missing")
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
            "    --local-llm --local-llm-model qwen3-vl:8b\n"
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
    vis.add_argument("--vision-pages-escalate", type=int, default=8,
                     dest="vision_pages_escalate",
                     help="Pages to render for the vision-LLM escalation pass "
                          "(default: 8). Used when an authoritative identifier "
                          "exists but no local extractor corroborated it after the "
                          "initial vision pass. Only fires once per file in safe mode.")
    vis.add_argument("--vision-dpi",          type=int, default=120,
                     help="Render DPI for the page images (default: 120). "
                          "Higher = sharper but more tokens. 96 is fine for "
                          "most models; 150 for dense text or small fonts.")
    vis.add_argument("--vision-llm-request-timeout", type=float, default=90.0,
                     help="HTTP timeout in seconds for remote vision LLM requests (default: 90)")
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
