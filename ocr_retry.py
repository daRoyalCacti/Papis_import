"""OCR retry: run ocrmypdf on a temp copy of a PDF and re-resolve."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

from papis_import.extractor_parts import ExtractorSet
from papis_import.http_client import HttpClient
from papis_import.models import Metadata, TimingBreakdown
from papis_import.pipeline import resolve
from papis_import.utils import DEFAULT_CACHE_DIR, command_exists


def run_ocr_retry(
    path: Path,
    extractors: ExtractorSet,
    args: argparse.Namespace,
    http: HttpClient,
    verbose: bool = False,
) -> tuple[Metadata | None, str, TimingBreakdown]:
    """Run ocrmypdf on a temp copy of *path* and re-resolve.

    Returns (new_meta_or_None, status_string, timing).  Status values:
      "ocrmypdf-missing"  — binary not on PATH
      "ocrmypdf-failed:…" — non-zero exit; suffix has last stderr line
      "ocrmypdf-timeout"  — exceeded 10-minute budget
      "no-output"         — exited 0 but didn't write the output file
      "resolve-raised:…"  — resolve() threw on the OCR'd file
      "ok"                — OCR succeeded; caller decides whether to accept

    Non-destructive: the original PDF is never modified.
    """
    timing = TimingBreakdown()
    retry_started = perf_counter()

    if not command_exists("ocrmypdf"):
        timing.ocr_retry_s = perf_counter() - retry_started
        return None, "ocrmypdf-missing", timing

    # Persistent cache: avoid re-running ocrmypdf on identical input files.
    cache_dir = Path(getattr(args, "cache_dir", None) or DEFAULT_CACHE_DIR)
    pdf_sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
    ocr_cache_path = cache_dir / "ocr" / f"{pdf_sha1}.pdf"

    try:
        if ocr_cache_path.exists():
            timing.ocrmypdf_s = 0.0
            resolve_started = perf_counter()
            try:
                new_meta, _cands, _text, _debug, resolve_timing = resolve(
                    ocr_cache_path, extractors, args, http, skip_vision=True
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

        with tempfile.TemporaryDirectory(prefix="papis_import_ocr_") as td:
            ocr_path = Path(td) / path.name
            try:
                ocr_started = perf_counter()
                cp = subprocess.run(
                    ["ocrmypdf",
                     "--force-ocr", "--optimize", "1",
                     "--output-type", "pdf",
                     str(path), str(ocr_path)],
                    check=False, capture_output=True, text=True,
                    timeout=600,
                )
                timing.ocrmypdf_s = perf_counter() - ocr_started
            except subprocess.TimeoutExpired:
                timing.ocrmypdf_s = perf_counter() - ocr_started
                timing.ocr_retry_s = perf_counter() - retry_started
                return None, "ocrmypdf-timeout", timing

            if cp.returncode != 0:
                stderr_tail = (cp.stderr or cp.stdout or "").strip().splitlines()
                stderr_tail = stderr_tail[-1] if stderr_tail else "(no output)"
                timing.ocr_retry_s = perf_counter() - retry_started
                return None, f"ocrmypdf-failed: {stderr_tail[:200]}", timing

            if not ocr_path.exists():
                timing.ocr_retry_s = perf_counter() - retry_started
                return None, "no-output", timing

            # Persist the OCR output before the temp dir is deleted.
            ocr_cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ocr_path, ocr_cache_path)

            resolve_started = perf_counter()
            try:
                new_meta, _cands, _text, _debug, resolve_timing = resolve(
                    ocr_cache_path, extractors, args, http, skip_vision=True
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
