"""Command-line interface and main processing loop."""
from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter

from papis_import.cli_args import OutputPaths, parse_args, resolve_output_paths
from papis_import.extractors import Extractor
from papis_import.grobid_service import local_grobid_session
from papis_import.http_client import Cache, HttpClient
from papis_import.models import Metadata, Record, TimingBreakdown
from papis_import.ocr_retry import run_ocr_retry
from papis_import.output.readers import load_previous_tsv
from papis_import.output.writers import DebugWriter, ProfileWriter, ResultWriter
from papis_import.pipeline import resolve
from papis_import.utils import (
    build_tags,
    clean_text,
    command_exists,
    confidence_rank,
    eprint,
    quote_shell,
    should_import,
)


def collect_pdfs(staging_dir: Path) -> list[Path]:
    return sorted(p for p in staging_dir.rglob("*.pdf") if p.is_file())


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


def _apply_ocr_result(
    meta: Metadata,
    path: Path,
    extractor: Extractor,
    timing: TimingBreakdown,
    verbose: bool,
) -> tuple[Metadata, int]:
    """Run OCR retry and merge into timing. Returns (accepted_meta, recovered_count)."""
    if verbose:
        print(f"  [ocr] {path.name}: retrying with ocrmypdf…")
    new_meta, status, ocr_timing = run_ocr_retry(path, extractor, verbose=verbose)
    timing.ocrmypdf_s      += ocr_timing.ocrmypdf_s
    timing.ocr_reresolve_s += ocr_timing.ocr_reresolve_s
    timing.ocr_retry_s     += ocr_timing.ocr_retry_s
    timing.identifier_lookups_s += ocr_timing.identifier_lookups_s
    timing.identifier_lookups.extend(ocr_timing.identifier_lookups)

    if new_meta is None:
        if verbose:
            print(f"  [ocr] {path.name}: FAILED ({status})")
        return meta, 0

    better_flags: list[str] = []
    if new_meta.verified and not meta.verified:
        better_flags.append("verified")
    if confidence_rank(new_meta.confidence) > confidence_rank(meta.confidence):
        better_flags.append(f"confidence {meta.confidence}→{new_meta.confidence}")
    if new_meta.sanity_score > meta.sanity_score + 0.01:
        better_flags.append(f"sanity {meta.sanity_score:.2f}→{new_meta.sanity_score:.2f}")
    for attr in ("doi", "isbn", "arxiv"):
        if getattr(new_meta, attr) and not getattr(meta, attr):
            better_flags.append(f"new {attr}")
    if new_meta.title and not meta.title:
        better_flags.append("new title")
    if new_meta.authors and not meta.authors:
        better_flags.append("new authors")

    if better_flags:
        reason = ", ".join(better_flags)
        new_meta.notes.append(
            f"OCR retry recovered metadata ({reason}) — "
            "consider running ocrmypdf to make the original PDF searchable"
        )
        if verbose:
            print(f"  [ocr] {path.name}: RECOVERED ({reason})")
        return new_meta, 1

    if verbose:
        print(f"  [ocr] {path.name}: no improvement — kept original")
    return meta, 0


def main() -> int:
    args = parse_args()

    if args.crossref_mailto and not args.mailto:
        args.mailto = args.crossref_mailto

    staging_dir = Path(args.staging).expanduser().resolve()
    if not staging_dir.exists():
        eprint(f"[error] staging directory does not exist: {staging_dir}")
        return 2
    if not args.dry_run and not args.do_import:
        args.dry_run = True

    try:
        paths: OutputPaths = resolve_output_paths(args)
    except ValueError as exc:
        eprint(f"[error] {exc}")
        return 2
    result_writer  = ResultWriter(paths.auto, paths.review, paths.soft)
    profile_writer = ProfileWriter(paths.profile) if paths.profile else None
    debug_writer   = DebugWriter(paths.debug)     if paths.debug   else None

    result_writer.init()
    if profile_writer:
        profile_writer.init()
    if debug_writer:
        debug_writer.init()

    files = collect_pdfs(staging_dir)
    if args.offset:
        files = files[args.offset:]
    if args.limit > 0:
        files = files[: args.limit]

    prev_verified: dict[str, Record] = {}
    if getattr(args, "retry_unverified", False):
        prev_verified, legacy_count = load_previous_tsv(paths.auto, paths.review)
        retry_count = sum(1 for f in files if str(f) not in prev_verified)
        skip_count  = sum(1 for f in files if str(f) in prev_verified)
        print(f"Retry mode: {skip_count} already auto-safe, {retry_count} to (re-)process")
        if legacy_count:
            print(f"            {legacy_count} previously-verified row(s) lack sanity data "
                  f"— re-processing to apply the new checks")

    if args.ocr and not command_exists("ocrmypdf"):
        eprint("[warning] --ocr was requested but 'ocrmypdf' is not on PATH. "
               "OCR retry will be skipped. Install ocrmypdf (e.g. `sudo pacman -S "
               "ocrmypdf` or `pip install --user ocrmypdf`) to enable it.")

    cache = Cache(args.cache_dir)
    if getattr(args, "clear_cache_errors", False):
        removed = cache.clear_errors()
        print(f"Cleared {removed} stale error entries from cache.")
    http = HttpClient(cache=cache, mailto=args.mailto, verbose=args.verbose)

    records: list[Record] = []
    ocr_attempted = ocr_recovered = 0
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
            if str(path) in prev_verified:
                rec = prev_verified[str(path)]
                records.append(rec)
                result_writer.append(rec)
                if debug_writer:
                    debug_writer.append(rec)
                if profile_writer:
                    profile_writer.append(rec, status="skipped: retry-unverified")
                continue

            file_started = perf_counter()
            if args.verbose:
                print(f"[{idx}/{total}] {path.name}")
            elif idx % 25 == 0:
                print(f"  … {idx}/{total}")

            tags = build_tags(staging_dir, path)
            try:
                meta, _candidates, _text, debug, timing = resolve(path, extractor)

                if (args.ocr
                        and meta.needs_ocr
                        and not meta.verified
                        and command_exists("ocrmypdf")):
                    ocr_attempted += 1
                    meta, recovered = _apply_ocr_result(
                        meta, path, extractor, timing, args.verbose
                    )
                    ocr_recovered += recovered

                cmd      = build_papis_command(path, tags, meta, args.link)
                imported = False
                err      = ""
                importable = meta.auto_safe or meta.soft_auto
                if args.do_import and importable and should_import(meta.confidence, args.min_confidence):
                    if not command_exists("papis"):
                        err = "papis executable not found"
                    else:
                        cp       = subprocess.run(cmd, capture_output=True, text=True, check=False)
                        imported = cp.returncode == 0
                        if not imported:
                            err = clean_text(cp.stderr or cp.stdout)
                elif args.do_import:
                    if not importable:
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
                result_writer.append(rec)
                if debug_writer:
                    debug_writer.append(rec)
                if profile_writer:
                    profile_writer.append(rec)

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
                    debug={"final_source": "error", "vision_used": "no",
                           "vision_status": "exception",
                           "vision_error": clean_text(str(exc))},
                    timing=timing,
                )
                records.append(rec)
                result_writer.append(rec)
                if debug_writer:
                    debug_writer.append(rec)
                if profile_writer:
                    profile_writer.append(rec, status="error")

    auto_count   = sum(1 for r in records if r.result.auto_safe)
    soft_count   = sum(1 for r in records if r.result.soft_auto and not r.result.auto_safe)
    review_count = len(records) - auto_count - soft_count
    high         = sum(1 for r in records if r.result.confidence == "high")
    med          = sum(1 for r in records if r.result.confidence == "medium")
    low          = sum(1 for r in records if r.result.confidence == "low")
    ver          = sum(1 for r in records if r.result.verified)
    sanity_pass  = sum(1 for r in records if r.result.sanity_passed)
    ocr_flagged  = sum(1 for r in records if r.result.needs_ocr)
    imp          = sum(1 for r in records if r.imported)

    print("-" * 60)
    print(f"Done. {total} PDFs scanned.")
    print(f"Confidence   : high={high}  medium={med}  low={low}")
    print(f"Verified     : {ver}/{total}")
    print(f"Sanity passed: {sanity_pass}/{total}")
    if ocr_attempted or ocr_flagged:
        print(f"OCR          : flagged={ocr_flagged}  retried={ocr_attempted}  recovered={ocr_recovered}")
    print(f"Auto-safe    : {auto_count}/{total}  (→ {paths.auto.name})")
    print(f"Soft auto    : {soft_count}/{total}  (also written to {paths.auto.name}; spot-check via {paths.soft.name})")
    print(f"Needs review : {review_count}/{total}  (→ {paths.review.name})")
    if args.do_import:
        print(f"Imported     : {imp}/{total}")
    print(f"TSV (auto)   : {paths.auto}")
    print(f"TSV (soft)   : {paths.soft}")
    print(f"TSV (review) : {paths.review}")
    if paths.debug:
        print(f"Debug JSONL  : {paths.debug}")
    if paths.profile:
        print(f"TSV (profile): {paths.profile}")
    print(f"Cache        : {Path(args.cache_dir).resolve()}")
    return 0
