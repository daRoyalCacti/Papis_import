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

from io_utils import DEFAULT_CONFIG_PATH, expand_path, load_json_config  # noqa: E402


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
    "vision_llm_endpoint": "vision-llm-endpoint",
    "vision_llm_api_key": "vision-llm-api-key",
    "vision_llm_model": "vision-llm-model",
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
    "auto": "papis_import.tsv",
    "review": "papis_import_review.tsv",
    "soft": "papis_import_soft.tsv",
    "debug": "papis_import_debug.tsv",
    "profile": "papis_import_profile.tsv",
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
    "Vision Used",
    "Vision Trigger",
    "Vision Status",
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
    "debug": ["Text LLM HTTP JSON", "Vision HTTP JSON", "Candidates JSON", "Title Search Queries JSON"],
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
        elif key == "debug":
            snapshot["rows"][key] = rows_by_file(rows, DEBUG_STABLE_COLUMNS)
        elif key == "profile":
            snapshot["rows"][key] = rows_by_file(rows, PROFILE_STABLE_COLUMNS)
            snapshot["summaries"][key] = timing_summary(rows)

        valid_cells: dict[str, bool] = {}
        for col in JSON_CELL_COLUMNS.get(key, []):
            if col in header:
                valid_cells[col] = all(parse_json_cell(row.get(col, "")) for row in rows)
        if valid_cells:
            snapshot["json_cells_valid"][key] = valid_cells

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
            "--debug-tsv",
            "<out>/papis_import_debug.tsv",
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

    for key in TSV_FILES:
        if current["schemas"].get(key) != base["schemas"].get(key):
            result.error(f"{key}: TSV header changed")
        if current["row_counts"].get(key) != base["row_counts"].get(key):
            message = (
                f"{key}: row count changed "
                f"{base['row_counts'].get(key)} -> {current['row_counts'].get(key)}"
            )
            if strict_rows or key in {"debug", "profile"}:
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
            if strict_rows or key in {"debug", "profile"}:
                result.error(message)
            else:
                result.warn(message)
        if added:
            message = f"{key}: added files: {', '.join(added)}"
            if strict_rows or key in {"debug", "profile"}:
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
    return {key: out_dir / filename for key, filename in TSV_FILES.items()}


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
        "--debug-tsv",
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

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
