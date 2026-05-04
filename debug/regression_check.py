#!/usr/bin/env python3
"""Regression guard for behavior-preserving papis_import refactors.

The script treats ``out_testing`` as a read-only golden run.  It can snapshot a
compact JSON baseline from that directory, then re-run the importer into a
temporary directory and compare stable output fields.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))

from io_utils import DEFAULT_CONFIG_PATH, expand_path, load_json_config  # noqa: E402
from _debug_flatten import read_debug_jsonl  # noqa: E402


DEFAULT_OUT = _PROJECT_ROOT / "out_testing"
DEFAULT_COLD_BASELINE = _PROJECT_ROOT / "debug" / "baselines" / "out_testing_cold_regression.json"
DEFAULT_CACHED_BASELINE = _PROJECT_ROOT / "debug" / "baselines" / "out_testing_cached_regression.json"
DEFAULT_CACHED_CACHE = _PROJECT_ROOT / "debug" / "baselines" / "out_testing_cached_cache"
DEFAULT_STAGING = _PROJECT_ROOT / "pdfs_for_testing"

CONFIG_MAP = {
    "mailto": "mailto",
    "crossref_mailto": "crossref-mailto",
    "title_search_timeout": "title-search-timeout",
    "google_books_api_key": "google-books-api-key",
    "grobid_url": "grobid-url",
    "start_local_grobid": "start-local-grobid",
    "grobid_runtime": "grobid-runtime",
    "grobid_image": "grobid-image",
    "grobid_port": "grobid-port",
    "grobid_start_timeout": "grobid-start-timeout",
    "llm_endpoint": "llm-endpoint",
    "llm_api_key": "llm-api-key",
    "llm_model": "llm-model",
    "llm_chars": "llm-chars",
    "llm_request_timeout": "llm-request-timeout",
    "local_llm": "local-llm",
    "local_llm_model": "local-llm-model",
    "local_llm_num_predict": "local-llm-num-predict",
    "ollama_host": "ollama-host",
    "ollama_start_timeout": "ollama-start-timeout",
    "ollama_request_timeout": "ollama-request-timeout",
    "ollama_pull_missing": "ollama-pull-missing",
    "vision_llm_endpoint": "vision-llm-endpoint",
    "vision_llm_api_key": "vision-llm-api-key",
    "vision_llm_model": "vision-llm-model",
    "vision_llm_request_timeout": "vision-llm-request-timeout",
    "vision_pages": "vision-pages",
    "vision_dpi": "vision-dpi",
    "vision_only_if_hard": "vision-only-if-hard",
    "no_semantic_scholar": "no-semantic-scholar",
    "semantic_scholar_api_key": "semantic-scholar-api-key",
    "ollama_model": "ollama-model",
}

SECRET_FLAGS = {
    "--google-books-api-key",
    "--llm-api-key",
    "--vision-llm-api-key",
    "--semantic-scholar-api-key",
}

TSV_FILES = {
    "auto":    "papis_import.tsv",
    "review":  "papis_import_review.tsv",
    "soft":    "papis_import_soft.tsv",
    "profile": "papis_import_profile.tsv",
}

JSONL_FILES = {
    "debug": "papis_import_debug.jsonl",
}

RESULT_STABLE_COLUMNS = [
    "Final Source",
    "Confidence",
    "Verified",
    "Title",
    "Authors",
    "Year",
    "DOI",
    "ISBN",
    "arXiv",
    "Sanity Passed",
    "Sanity Score",
    "Auto Safe",
    "Needs OCR",
    "Soft Auto",
    "Soft Auto Reasons",
]

DEBUG_STABLE_COLUMNS = [
    "Final Source",
    "Confidence",
    "Verified",
    "Sanity Passed",
    "Sanity Score",
    "Auto Safe",
    "Needs OCR",
    "Text LLM Used",
    "Text LLM Model",
    "Text LLM Status",
    "Vision Used",
    "Vision Trigger",
    "Vision Status",
    "Vision Escalated",
    "Vision Model",
    "GROBID Used",
    "Local Best Source",
    "Identifier DOIs",
    "Identifier ISBNs",
    "Identifier arXivs",
    "Candidate Sources",
    "Title",
    "Authors",
    "Year",
    "DOI",
    "ISBN",
    "arXiv",
    "Soft Auto",
    "Soft Auto Reasons",
]

PROFILE_STABLE_COLUMNS = [
    "Status",
    "Error",
    "Final Source",
    "Confidence",
    "Verified",
    "Identifier Lookup Count",
    "Crossref Search Matches",
    "Crossref Search Errors",
    "OpenAlex Search Matches",
    "OpenAlex Search Errors",
    "Semantic Scholar Search Matches",
    "Semantic Scholar Search Errors",
    "OpenLibrary Search Matches",
    "OpenLibrary Search Errors",
    "Google Books Search Matches",
    "Google Books Search Errors",
]

PROFILE_TIMING_COLUMNS = [
    "File Wall s",
    "Resolve Total s",
    "Text Extract s",
    "Embedded Metadata s",
    "GROBID s",
    "Text LLM s",
    "Text LLM HTTP s",
    "Text LLM Retry Sleep s",
    "Text LLM Pacing s",
    "Vision LLM s",
    "Vision HTTP s",
    "Vision Retry Sleep s",
    "Vision Pacing s",
    "Identifier Lookups s",
    "Title Search s",
    "OCR Retry s",
    "OCRMyPDF s",
    "OCR Reresolve s",
    "Best Local s",
    "Header Candidate s",
]

JSON_CELL_COLUMNS = {
    "profile": ["Identifier Lookups JSON", "Title Searches JSON"],
}


class CheckResult:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def ok(self) -> bool:
        return not self.errors


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def file_key(row: dict[str, str]) -> str:
    value = row.get("File Path", "")
    return Path(value).name if value else ""


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def stable_row(row: dict[str, str], columns: list[str]) -> dict[str, str]:
    return {col: row.get(col, "") for col in columns}


def rows_by_file(rows: list[dict[str, str]], columns: list[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        key = file_key(row)
        if key in out:
            duplicates.append(key)
        out[key] = stable_row(row, columns)
    if duplicates:
        raise RuntimeError(f"Duplicate file basenames in TSV: {', '.join(sorted(duplicates))}")
    return out


def parse_json_cell(value: str) -> bool:
    if not str(value or "").strip():
        return True
    try:
        json.loads(value)
    except Exception:
        return False
    return True


def cache_counts(root: Path) -> dict[str, int]:
    cache_root = root / "cache"
    if not cache_root.exists():
        return {}
    counts: dict[str, int] = {}
    for child in sorted(cache_root.iterdir()):
        if child.is_dir():
            counts[child.name] = len(list(child.rglob("*.json")))
    return counts


def timing_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    file_wall = [as_float(row.get("File Wall s")) for row in rows]
    total = sum(file_wall)
    return {
        "file_count": len(rows),
        "file_wall_total_s": total,
        "file_wall_median_s": statistics.median(file_wall) if file_wall else 0.0,
        "file_wall_max_s": max(file_wall) if file_wall else 0.0,
        "timing_totals_s": {
            col: sum(as_float(row.get(col)) for row in rows)
            for col in PROFILE_TIMING_COLUMNS
        },
    }


def classification_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    fields = ["Final Source", "Confidence", "Verified", "Auto Safe", "Soft Auto", "Vision Used"]
    return {field: dict(Counter(row.get(field, "") for row in rows)) for field in fields}


def load_output_snapshot(out_dir: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "schemas": {},
        "row_counts": {},
        "rows": {},
        "summaries": {},
        "json_cells_valid": {},
        "cache_counts": cache_counts(out_dir),
    }

    for key, filename in TSV_FILES.items():
        header, rows = read_tsv(out_dir / filename)
        snapshot["schemas"][key] = header
        snapshot["row_counts"][key] = len(rows)

        if key in {"auto", "review", "soft"}:
            snapshot["rows"][key] = rows_by_file(rows, RESULT_STABLE_COLUMNS)
            snapshot["summaries"][key] = classification_summary(rows)
        elif key == "profile":
            snapshot["rows"][key] = rows_by_file(rows, PROFILE_STABLE_COLUMNS)
            snapshot["summaries"][key] = timing_summary(rows)

        valid_cells: dict[str, bool] = {}
        for col in JSON_CELL_COLUMNS.get(key, []):
            if col in header:
                valid_cells[col] = all(parse_json_cell(row.get(col, "")) for row in rows)
        if valid_cells:
            snapshot["json_cells_valid"][key] = valid_cells

    for key, filename in JSONL_FILES.items():
        rows = read_debug_jsonl(out_dir / filename, required=True)
        snapshot["schemas"][key] = "jsonl"
        snapshot["row_counts"][key] = len(rows)
        if key == "debug":
            snapshot["rows"][key] = rows_by_file(rows, DEBUG_STABLE_COLUMNS)

    return snapshot


def make_baseline(out_dir: Path, args: argparse.Namespace, *, source: str) -> dict[str, Any]:
    baseline_path = expand_path(args.baseline)
    snapshot = load_output_snapshot(out_dir)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "source": source,
        "source_out_dir": str(out_dir),
        "staging_dir": str(expand_path(args.staging)),
        "command_template": [
            "python",
            "run_papis_import.py",
            "--staging",
            str(expand_path(args.staging)),
            "--dry-run",
            "--limit",
            str(args.limit),
            "--ocr",
            "--retry-unverified",
            "--tsv",
            "<out>/papis_import.tsv",
            "--review-tsv",
            "<out>/papis_import_review.tsv",
            "--soft-tsv",
            "<out>/papis_import_soft.tsv",
            "--debug-jsonl",
            "<out>/papis_import_debug.jsonl",
            "--profile-tsv",
            "<out>/papis_import_profile.tsv",
            "--cache-dir",
            "<out>/cache",
        ],
        "snapshot": snapshot,
    }


def save_baseline(baseline: dict[str, Any], path: str | Path) -> None:
    baseline_path = expand_path(path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote baseline: {baseline_path}")


def write_baseline(args: argparse.Namespace) -> int:
    out_dir = expand_path(args.out)
    baseline = make_baseline(out_dir, args, source="existing-output")
    save_baseline(baseline, args.baseline)
    print_summary(baseline["snapshot"])
    return 0


def replace_dir(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def write_cached_run_baseline(args: argparse.Namespace) -> int:
    if args.compileall:
        rc = run_compileall()
        if rc != 0:
            return rc
    args.cache_seed = str(expand_path(args.out) / "cache")
    rc, out_dir = run_importer(args, mode="cached")
    if rc != 0:
        print(f"Importer failed with exit code {rc}. Output kept at {out_dir}", file=sys.stderr)
        return rc
    baseline = make_baseline(out_dir, args, source="cached-run")
    save_baseline(baseline, args.baseline)
    cached_cache = expand_path(args.cached_cache)
    replace_dir(out_dir / "cache", cached_cache)
    print(f"Wrote cached cache seed: {cached_cache}")
    print_summary(baseline["snapshot"])
    if not args.keep_output:
        shutil.rmtree(out_dir)
    else:
        print(f"Output kept at {out_dir}")
    return 0


def load_baseline(path: str | Path) -> dict[str, Any]:
    baseline_path = expand_path(path)
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def print_summary(snapshot: dict[str, Any]) -> None:
    print("Rows:")
    for key in TSV_FILES:
        print(f"  {key}: {snapshot['row_counts'].get(key, 0)}")
    profile = snapshot.get("summaries", {}).get("profile", {})
    if profile:
        print(
            "Timing: "
            f"total={profile.get('file_wall_total_s', 0):.2f}s, "
            f"median={profile.get('file_wall_median_s', 0):.2f}s, "
            f"max={profile.get('file_wall_max_s', 0):.2f}s"
        )
    counts = snapshot.get("cache_counts", {})
    if counts:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"Cache: {rendered}")


def compare_snapshots(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    mode: str,
    strict_network: bool,
) -> CheckResult:
    result = CheckResult()
    base = baseline["snapshot"]
    strict_rows = mode == "cached" or strict_network

    all_output_keys = list(TSV_FILES) + list(JSONL_FILES)
    for key in all_output_keys:
        if current["schemas"].get(key) != base["schemas"].get(key):
            result.error(f"{key}: schema changed")
        if current["row_counts"].get(key) != base["row_counts"].get(key):
            message = (
                f"{key}: row count changed "
                f"{base['row_counts'].get(key)} -> {current['row_counts'].get(key)}"
            )
            if strict_rows or key in {"profile"} or key in JSONL_FILES:
                result.error(message)
            else:
                result.warn(message)

    for key, cells in current.get("json_cells_valid", {}).items():
        for col, valid in cells.items():
            if not valid:
                result.error(f"{key}: JSON column no longer parses: {col}")

    for key in ["auto", "review", "soft", "debug", "profile"]:
        base_rows = base["rows"].get(key, {})
        current_rows = current["rows"].get(key, {})
        missing = sorted(set(base_rows) - set(current_rows))
        added = sorted(set(current_rows) - set(base_rows))
        if missing:
            message = f"{key}: missing files: {', '.join(missing)}"
            if strict_rows or key in {"profile"} or key in JSONL_FILES:
                result.error(message)
            else:
                result.warn(message)
        if added:
            message = f"{key}: added files: {', '.join(added)}"
            if strict_rows or key in {"profile"} or key in JSONL_FILES:
                result.error(message)
            else:
                result.warn(message)

        changed: list[str] = []
        for pdf_name in sorted(set(base_rows) & set(current_rows)):
            if current_rows[pdf_name] != base_rows[pdf_name]:
                changed.append(pdf_name)
        if changed and strict_rows:
            shown = ", ".join(changed[:8])
            extra = "" if len(changed) <= 8 else f" (+{len(changed) - 8} more)"
            result.error(f"{key}: stable row fields changed for {shown}{extra}")
        elif changed:
            shown = ", ".join(changed[:8])
            extra = "" if len(changed) <= 8 else f" (+{len(changed) - 8} more)"
            result.warn(f"{key}: stable row fields drifted for {shown}{extra}")

    compare_timing(base, current, mode=mode, result=result)
    compare_cache_counts(base, current, mode=mode, result=result)
    return result


def compare_timing(base: dict[str, Any], current: dict[str, Any], *, mode: str, result: CheckResult) -> None:
    base_profile = base.get("summaries", {}).get("profile", {})
    current_profile = current.get("summaries", {}).get("profile", {})
    base_total = float(base_profile.get("file_wall_total_s", 0.0) or 0.0)
    current_total = float(current_profile.get("file_wall_total_s", 0.0) or 0.0)
    base_median = float(base_profile.get("file_wall_median_s", 0.0) or 0.0)
    current_median = float(current_profile.get("file_wall_median_s", 0.0) or 0.0)
    base_max = float(base_profile.get("file_wall_max_s", 0.0) or 0.0)
    current_max = float(current_profile.get("file_wall_max_s", 0.0) or 0.0)

    if mode == "cached":
        total_limit = base_total * 2.0 + 10.0
        median_limit = base_median * 2.5 + 2.0
        if current_total > total_limit:
            result.error(f"cached timing: total wall time grew {base_total:.2f}s -> {current_total:.2f}s")
        if current_median > median_limit:
            result.warn(f"cached timing: median wall time grew {base_median:.2f}s -> {current_median:.2f}s")
        if current_max > base_max * 3.0 + 10.0:
            result.warn(f"cached timing: max file wall time grew {base_max:.2f}s -> {current_max:.2f}s")
    else:
        total_limit = base_total * 2.5 + 10.0
        if current_total > total_limit:
            result.warn(f"network timing: total wall time grew {base_total:.2f}s -> {current_total:.2f}s")


def compare_cache_counts(base: dict[str, Any], current: dict[str, Any], *, mode: str, result: CheckResult) -> None:
    base_counts = base.get("cache_counts", {})
    current_counts = current.get("cache_counts", {})
    if mode == "cached":
        for namespace, count in base_counts.items():
            if current_counts.get(namespace, 0) < count:
                result.error(
                    f"cache namespace shrank in cached run: "
                    f"{namespace} {count} -> {current_counts.get(namespace, 0)}"
                )
    else:
        missing = sorted(set(base_counts) - set(current_counts))
        if missing:
            result.warn(f"network run did not populate cache namespaces: {', '.join(missing)}")


def output_paths(out_dir: Path) -> dict[str, Path]:
    paths = {key: out_dir / filename for key, filename in TSV_FILES.items()}
    paths.update({key: out_dir / filename for key, filename in JSONL_FILES.items()})
    return paths


def config_defaults(args: argparse.Namespace, *, mode: str) -> list[str]:
    if args.no_config:
        return []
    try:
        cfg = load_json_config(args.config, missing_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load config {args.config}: {exc}") from exc

    if not args.allow_start_local_grobid and cfg.get("start_local_grobid"):
        cfg = dict(cfg)
        cfg["start_local_grobid"] = False
        if not cfg.get("grobid_url"):
            port = int(cfg.get("grobid_port") or 8070)
            cfg["grobid_url"] = f"http://127.0.0.1:{port}"
        if mode == "network":
            print("Note: suppressing --start-local-grobid; pass --allow-start-local-grobid to test it.")

    injected: list[str] = []
    for cfg_key, flag_name in CONFIG_MAP.items():
        value = cfg.get(cfg_key)
        if value is None:
            continue
        flag = f"--{flag_name}"
        if isinstance(value, bool):
            if value:
                injected.append(flag)
        else:
            injected.extend([flag, str(value)])
    return injected


def redacted_command(cmd: list[str]) -> str:
    redacted: list[str] = []
    hide_next = False
    for arg in cmd:
        if hide_next:
            redacted.append("[redacted]")
            hide_next = False
            continue
        if any(arg.startswith(flag + "=") for flag in SECRET_FLAGS):
            flag, _, _ = arg.partition("=")
            redacted.append(f"{flag}=[redacted]")
            continue
        redacted.append(arg)
        if arg in SECRET_FLAGS:
            hide_next = True
    return subprocess.list2cmdline(redacted)


def build_import_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    paths = output_paths(out_dir)
    cmd = [
        sys.executable,
        "-m",
        "papis_import",
        *config_defaults(args, mode=args.mode),
        "--staging",
        str(expand_path(args.staging)),
        "--dry-run",
        "--limit",
        str(args.limit),
        "--ocr",
        "--retry-unverified",
        "--tsv",
        str(paths["auto"]),
        "--review-tsv",
        str(paths["review"]),
        "--soft-tsv",
        str(paths["soft"]),
        "--debug-jsonl",
        str(paths["debug"]),
        "--profile-tsv",
        str(paths["profile"]),
        "--cache-dir",
        str(out_dir / "cache"),
    ]
    return cmd


def run_compileall() -> int:
    cmd = [sys.executable, "-m", "compileall", "-q", str(_PROJECT_ROOT)]
    print("Compile:", subprocess.list2cmdline(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(_PROJECT_ROOT))


def run_importer(args: argparse.Namespace, *, mode: str) -> tuple[int, Path]:
    args.mode = mode
    out_dir = Path(tempfile.mkdtemp(prefix=f"papis_import_regression_{mode}_"))
    if mode == "cached":
        source_cache = expand_path(getattr(args, "cache_seed", "") or getattr(args, "cached_cache", ""))
        if not source_cache.exists() and getattr(args, "cached_cache", ""):
            raise FileNotFoundError(
                f"Cached cache seed not found: {source_cache}. "
                "Run `python debug/regression_check.py init-cached-baseline` first."
            )
        if not source_cache:
            source_cache = expand_path(args.out) / "cache"
        if source_cache.exists():
            shutil.copytree(source_cache, out_dir / "cache")
    cmd = build_import_command(args, out_dir)
    print("Output:", out_dir, flush=True)
    print("Running:", redacted_command(cmd), flush=True)
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_PROJECT_ROOT.parent) + (os.pathsep + python_path if python_path else "")
    completed = subprocess.run(cmd, cwd=str(_PROJECT_ROOT.parent), env=env, text=True)
    return completed.returncode, out_dir


def check_run(args: argparse.Namespace, *, mode: str) -> int:
    if args.compileall:
        rc = run_compileall()
        if rc != 0:
            return rc

    baseline = load_baseline(args.baseline)
    try:
        rc, out_dir = run_importer(args, mode=mode)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if rc != 0:
        print(f"Importer failed with exit code {rc}. Output kept at {out_dir}", file=sys.stderr)
        return rc

    current = load_output_snapshot(out_dir)
    print("Current run summary:")
    print_summary(current)
    result = compare_snapshots(
        baseline,
        current,
        mode=mode,
        strict_network=getattr(args, "strict_network", False),
    )

    for warning in result.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in result.errors:
        print(f"ERROR: {error}", file=sys.stderr)

    if result.ok():
        print(f"{mode} regression check passed")
        if not args.keep_output:
            shutil.rmtree(out_dir)
        else:
            print(f"Output kept at {out_dir}")
        return 0

    print(f"{mode} regression check failed. Output kept at {out_dir}", file=sys.stderr)
    return 1


def run_helper_checks(args: argparse.Namespace) -> int:
    # Import package modules with the parent on sys.path, otherwise the repo's
    # papis_import/http package can shadow the stdlib http package.
    project_root_str = str(_PROJECT_ROOT)
    while project_root_str in sys.path:
        sys.path.remove(project_root_str)
    from papis_import.core.text import clean_author_name
    from papis_import.models import Candidate, Metadata
    from papis_import.pipeline_parts.candidates import CandidateSelector
    from papis_import.pipeline_parts.finalization import (
        ResolutionFinalizer,
        _identifier_corroboration_decision,
        _is_distinctive_identifier_title,
    )

    def assert_equal(actual: object, expected: object, label: str) -> None:
        if actual != expected:
            raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")

    assert_equal(clean_author_name("Evarist Giné (auth.)"), "Evarist Giné", "author role cleanup")
    assert_equal(clean_author_name("Jean Picard [eds.]"), "Jean Picard", "editor role cleanup")
    assert_equal(
        _is_distinctive_identifier_title("Introduction to Real Analysis"),
        True,
        "Introduction to Real Analysis is distinctive for identifier gating",
    )
    assert_equal(
        _is_distinctive_identifier_title("Introduction to High-Dimensional Statistics"),
        True,
        "Introduction to High-Dimensional Statistics is distinctive for identifier gating",
    )
    assert_equal(_is_distinctive_identifier_title("Thesis"), False, "exact Thesis is generic")

    selector = CandidateSelector()
    synthesized = selector.synthesize([
        Candidate(
            title="Decoupling From Dependence to Independence",
            authors=["Víctor H. de la Peña", "Evarist Giné (auth.)"],
            year="1999",
            source="filename_author_title",
            priority=45,
        ),
        Candidate(
            title="Decoupling: From Dependence to Independence",
            authors=["Víctor H. de la Peña", "Evarist Giné"],
            year="1999",
            source="llm:fixture",
            priority=30,
        ),
    ])
    assert_equal(synthesized[0].authors, ["Víctor H. de la Peña", "Evarist Giné"], "synthesized clean authors")

    meta = Metadata(
        title="Decoupling",
        authors=["Víctor H. de la Peña", "Evarist Giné"],
        year="1999",
        source="crossref_doi",
        confidence="high",
        verified=True,
        sanity_passed=True,
        sanity_score=1.0,
    )
    filename_cand = Candidate(
        title="Decoupling From Dependence to Independence",
        authors=["Víctor H. de la Peña", "Evarist Giné (auth.)"],
        year="1999",
        source="filename_author_title",
        priority=45,
    )
    llm_cand = Candidate(
        title="Decoupling: From Dependence to Independence",
        authors=["Víctor H. de la Peña", "Evarist Giné"],
        year="1999",
        source="llm:fixture",
        priority=30,
    )
    # filename_author_title carries both title and author — counts as strong.
    strong = _identifier_corroboration_decision(meta, [filename_cand, llm_cand])
    assert_equal(strong.accepted, True, "strong subset accepted")
    assert_equal(strong.force_review, False, "strong subset is auto-safe eligible")
    assert_equal(strong.soft_reason, "", "strong subset does not use soft-auto path")

    one_strong = _identifier_corroboration_decision(meta, [llm_cand])
    assert_equal(one_strong.accepted, True, "one strong subset accepted")
    assert_equal(one_strong.force_review, False, "one strong subset is auto-safe eligible")
    assert_equal(one_strong.soft_reason, "", "one strong subset does not use soft-auto path")

    one_filename_strong = _identifier_corroboration_decision(meta, [filename_cand])
    assert_equal(one_filename_strong.accepted, True, "filename_author_title subset accepted")
    assert_equal(one_filename_strong.force_review, False, "filename_author_title subset is auto-safe eligible")

    # filename_structured carries structured metadata but less author fidelity — weak.
    structured_cand = Candidate(
        title="Decoupling: From Dependence to Independence",
        authors=["Víctor H. de la Peña", "Evarist Giné"],
        year="1999",
        source="filename_structured",
        priority=45,
    )
    weak = _identifier_corroboration_decision(meta, [structured_cand])
    assert_equal(weak.accepted, True, "weak-only subset accepted for review")
    assert_equal(weak.force_review, True, "weak-only subset still capped below auto-safe")
    assert_equal(weak.soft_reason, "", "weak subset not soft-auto")

    low_sanity = Metadata(
        title="Decoupling",
        authors=["Víctor H. de la Peña", "Evarist Giné"],
        year="1999",
        source="crossref_doi",
        confidence="high",
        verified=True,
        sanity_passed=True,
        sanity_score=0.6,
    )
    low_sanity_decision = _identifier_corroboration_decision(low_sanity, [llm_cand])
    assert_equal(low_sanity_decision.accepted, True, "low-sanity subset accepted")
    assert_equal(low_sanity_decision.force_review, False, "low-sanity strong subset is auto-safe eligible")
    assert_equal(low_sanity_decision.soft_reason, "", "low-sanity subset does not use soft-auto path")

    missing_year_cand = Candidate(
        title="Decoupling: From Dependence to Independence",
        authors=["Víctor H. de la Peña", "Evarist Giné"],
        year="",
        source="llm:fixture",
        priority=30,
    )
    missing_year_decision = _identifier_corroboration_decision(meta, [missing_year_cand])
    assert_equal(missing_year_decision.accepted, True, "missing-year subset accepted")
    assert_equal(missing_year_decision.force_review, False, "missing-year strong subset is auto-safe eligible")

    author_mismatch = _identifier_corroboration_decision(meta, [
        Candidate(
            title="Decoupling: From Dependence to Independence",
            authors=["Alice Smith"],
            year="1999",
            source="llm:fixture",
            priority=30,
        )
    ])
    assert_equal(author_mismatch.accepted, False, "subset with author mismatch rejected")

    year_mismatch = _identifier_corroboration_decision(meta, [
        Candidate(
            title="Decoupling: From Dependence to Independence",
            authors=["Víctor H. de la Peña", "Evarist Giné"],
            year="2000",
            source="llm:fixture",
            priority=30,
        )
    ])
    assert_equal(year_mismatch.accepted, False, "subset with year mismatch rejected")

    strict = _identifier_corroboration_decision(Metadata(
        title="Introduction to Real Analysis",
        authors=["Christopher Heil"],
        year="2019",
        source="openlibrary_isbn",
        confidence="high",
        verified=True,
        sanity_passed=True,
        sanity_score=1.0,
    ), [
        Candidate(
            title="Introduction to Real Analysis",
            authors=["Christopher Heil"],
            year="2019",
            source="llm:fixture",
            priority=30,
        )
    ])
    assert_equal(strict.accepted, True, "strict distinctive title accepted")
    assert_equal(strict.force_review, False, "strict distinctive title remains auto-safe eligible")

    generic = Metadata(
        title="Thesis",
        authors=["Alice Smith"],
        year="2020",
        source="crossref_doi",
        confidence="high",
        verified=True,
        sanity_passed=True,
        sanity_score=1.0,
    )
    generic_decision = _identifier_corroboration_decision(generic, [
        Candidate(
            title="Thesis on Probability",
            authors=["Alice Smith"],
            year="2020",
            source="llm:fixture",
            priority=30,
        )
    ])
    assert_equal(generic_decision.accepted, False, "generic subset rejected")

    generic_strict = _identifier_corroboration_decision(generic, [
        Candidate(
            title="Thesis",
            authors=["Alice Smith"],
            year="2020",
            source="llm:fixture",
            priority=30,
        )
    ])
    assert_equal(generic_strict.accepted, True, "generic strict accepted for review")
    assert_equal(generic_strict.force_review, True, "generic strict capped below auto-safe")
    assert_equal(generic_strict.soft_reason, "", "generic strict not soft-auto")

    # Uncorroborated identifier fallback: sanity-passed identifier with no local
    # agreement should land in review (auto_safe=False, soft_auto=False) with the
    # corroboration-failure note, not fall back to local garbage.
    unc_meta = Metadata(
        title="Algorithmic Learning Theory",
        authors=["Naoki Abe", "Roni Khardon", "Thomas Zeugmann"],
        year="2003",
        source="openlibrary_isbn",
        confidence="high",
        verified=True,
        sanity_passed=True,
        sanity_score=1.0,
    )
    finalizer = ResolutionFinalizer(CandidateSelector())
    garbage_cand = Candidate(
        title="Lecture Notes in Artificial Intelligence",
        authors=["Subseries of Lecture Notes in Computer Science"],
        year="",
        source="text_header",
        priority=90,
    )
    unc_result = finalizer.finalize_uncorroborated_identifier(
        unc_meta, "resolved by ISBN via OpenLibrary", [garbage_cand], False, {}
    )
    assert_equal(unc_result.auto_safe, False, "uncorroborated identifier is not auto-safe")
    assert_equal(unc_result.soft_auto, False, "uncorroborated identifier is not soft-auto")
    assert_equal(unc_result.confidence, "medium", "uncorroborated identifier confidence demoted")
    if not any("no local extractor corroborated" in n for n in unc_result.notes):
        raise AssertionError("uncorroborated identifier result missing corroboration-failure note")
    assert_equal(unc_result.source, "openlibrary_isbn", "uncorroborated identifier retains API source")
    assert_equal(unc_result.title, "Algorithmic Learning Theory", "uncorroborated identifier retains API title")

    print("helper checks passed")
    return 0


def add_common_args(p: argparse.ArgumentParser, *, baseline: Path) -> None:
    p.add_argument("--baseline", default=str(baseline), help="Compact JSON baseline path")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Golden output directory, default: out_testing")
    p.add_argument("--staging", default=str(DEFAULT_STAGING), help="PDF fixture directory")
    p.add_argument("--limit", type=int, default=300, help="Importer --limit value")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init-baseline", help="Snapshot out_testing into a compact cold/network baseline JSON")
    add_common_args(init, baseline=DEFAULT_COLD_BASELINE)
    init.set_defaults(func=write_baseline)

    init_cached = sub.add_parser(
        "init-cached-baseline",
        help="Run once with copied out_testing cache and snapshot that cached-run behavior",
    )
    add_common_args(init_cached, baseline=DEFAULT_CACHED_BASELINE)
    init_cached.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Config path for API/model defaults")
    init_cached.add_argument("--cached-cache", default=str(DEFAULT_CACHED_CACHE), help="Frozen cache seed to write")
    init_cached.add_argument("--no-config", action="store_true", help="Do not load config defaults")
    init_cached.add_argument("--allow-start-local-grobid", action="store_true", help="Allow config to start GROBID")
    init_cached.add_argument("--keep-output", action="store_true", help="Keep temp output after writing baseline")
    init_cached.add_argument("--no-compileall", dest="compileall", action="store_false")
    init_cached.set_defaults(func=write_cached_run_baseline, compileall=True)

    cached = sub.add_parser("cached", help="Run with a copied golden cache and strict comparisons")
    add_common_args(cached, baseline=DEFAULT_CACHED_BASELINE)
    cached.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Config path for API/model defaults")
    cached.add_argument("--cached-cache", default=str(DEFAULT_CACHED_CACHE), help="Frozen cache seed to use")
    cached.add_argument("--no-config", action="store_true", help="Do not load config defaults")
    cached.add_argument("--allow-start-local-grobid", action="store_true", help="Allow config to start GROBID")
    cached.add_argument("--keep-output", action="store_true", help="Keep temp output even on success")
    cached.add_argument("--no-compileall", dest="compileall", action="store_false")
    cached.set_defaults(func=lambda args: check_run(args, mode="cached"), compileall=True)

    network = sub.add_parser("network", help="Run with an empty cache and lenient drift warnings")
    add_common_args(network, baseline=DEFAULT_COLD_BASELINE)
    network.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Config path for API/model defaults")
    network.add_argument("--no-config", action="store_true", help="Do not load config defaults")
    network.add_argument("--allow-start-local-grobid", action="store_true", help="Allow config to start GROBID")
    network.add_argument("--keep-output", action="store_true", help="Keep temp output even on success")
    network.add_argument("--strict-network", action="store_true", help="Fail on stable field drift")
    network.add_argument("--no-compileall", dest="compileall", action="store_false")
    network.set_defaults(func=lambda args: check_run(args, mode="network"), compileall=True)

    helpers = sub.add_parser("helpers", help="Run fast pure helper/policy checks")
    helpers.set_defaults(func=run_helper_checks)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
