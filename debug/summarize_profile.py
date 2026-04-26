#!/usr/bin/env python3
"""Summarize papis_import profile TSV timing, quality, and resolver behavior."""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from io_utils import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROFILE_TSV,
    PROJECT_ROOT,
    as_float,
    as_int,
    expand_path,
    is_yes,
    load_json_config,
    parse_json_list_cell,
    read_tsv_dicts,
)


DEFAULT_DEST = PROJECT_ROOT / "out" / "profile_summary.md"

PROVIDERS = [
    ("Crossref", "Crossref Search"),
    ("OpenAlex", "OpenAlex Search"),
    ("Semantic Scholar", "Semantic Scholar Search"),
    ("OpenLibrary", "OpenLibrary Search"),
    ("Google Books", "Google Books Search"),
]

KEY_SLOW_COLUMNS = [
    "File Wall s",
    "Resolve Total s",
    "Title Search s",
    "Identifier Lookups s",
    "Vision LLM s",
    "Vision Pacing s",
    "Text LLM s",
    "GROBID s",
    "OCR Retry s",
    "OCRMyPDF s",
]


def seconds(value: float) -> str:
    if value >= 3600:
        return f"{value / 3600:.2f}h"
    if value >= 60:
        return f"{value / 60:.2f}m"
    return f"{value:.3f}s"


def pct(part: float, whole: float) -> str:
    if whole <= 0:
        return "0.0%"
    return f"{100.0 * part / whole:.1f}%"


def timing_columns(rows: list[dict[str, str]]) -> list[str]:
    seen: set[str] = set()
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key.endswith(" s") and key not in seen:
                seen.add(key)
                cols.append(key)
    return cols


def top_rows(
    rows: list[dict[str, str]],
    column: str,
    limit: int,
    positive_only: bool = True,
) -> list[dict[str, str]]:
    filtered = rows
    if positive_only:
        filtered = [row for row in rows if as_float(row.get(column, "")) > 0]
    return sorted(filtered, key=lambda row: as_float(row.get(column, "")), reverse=True)[:limit]


def summarize_identifier_json(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], Counter], Counter]:
    resolver_stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    errors: Counter[str] = Counter()
    for row in rows:
        for item in parse_json_list_cell(row.get("Identifier Lookups JSON", "")):
            resolver = str(item.get("resolver") or "(unknown)")
            kind = str(item.get("kind") or "(unknown)")
            key = (resolver, kind)
            resolver_stats[key]["lookups"] += 1
            if item.get("matched") is True:
                resolver_stats[key]["matches"] += 1
            error = str(item.get("error") or "").strip()
            if error:
                resolver_stats[key]["errors"] += 1
                errors[error] += 1
    return resolver_stats, errors


def summarize_title_search_json(rows: list[dict[str, str]]) -> tuple[Counter[str], Counter[str]]:
    errors: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    for row in rows:
        for item in parse_json_list_cell(row.get("Title Searches JSON", "")):
            for message in item.get("error_messages") or []:
                message = str(message).strip()
                if message:
                    errors[message] += 1
            skip_reason = str(item.get("skip_reason") or "").strip()
            if skip_reason:
                skip_reasons[skip_reason] += 1
    return errors, skip_reasons


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["(none)"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def file_label(path: str) -> str:
    return Path(path).name or path or "(unknown)"


def build_summary(rows: list[dict[str, str]], profile_tsv: Path, limit: int) -> str:
    total = len(rows)
    status_counts = Counter(row.get("Status", "") or "(blank)" for row in rows)
    confidence_counts = Counter((row.get("Confidence", "") or "(blank)").lower() for row in rows)
    final_source_counts = Counter(row.get("Final Source", "") or "(blank)" for row in rows)
    verified = sum(1 for row in rows if is_yes(row.get("Verified", "")))
    top_errors = Counter(row.get("Error", "").strip() for row in rows if row.get("Error", "").strip())
    rows_with_errors = sum(1 for row in rows if row.get("Error", "").strip())

    time_cols = timing_columns(rows)
    time_totals = {col: sum(as_float(row.get(col, "")) for row in rows) for col in time_cols}
    file_wall_total = time_totals.get("File Wall s", 0.0)
    resolve_total = time_totals.get("Resolve Total s", 0.0)
    file_wall_values = [as_float(row.get("File Wall s", "")) for row in rows]
    avg_file_wall = statistics.mean(file_wall_values) if file_wall_values else 0.0
    med_file_wall = statistics.median(file_wall_values) if file_wall_values else 0.0
    max_file_wall = max(file_wall_values) if file_wall_values else 0.0

    provider_rows: list[list[str]] = []
    for label, prefix in PROVIDERS:
        provider_time = sum(as_float(row.get(f"{prefix} s", "")) for row in rows)
        tried = sum(as_int(row.get(f"{prefix} Candidates Tried", "")) for row in rows)
        matches = sum(as_int(row.get(f"{prefix} Matches", "")) for row in rows)
        errors = sum(as_int(row.get(f"{prefix} Errors", "")) for row in rows)
        provider_rows.append([
            label,
            seconds(provider_time),
            str(tried),
            str(matches),
            str(errors),
            pct(matches, tried),
        ])

    identifier_count = sum(as_int(row.get("Identifier Lookup Count", "")) for row in rows)
    resolver_stats, identifier_errors = summarize_identifier_json(rows)
    identifier_matches = sum(counter["matches"] for counter in resolver_stats.values())
    identifier_json_lookups = sum(counter["lookups"] for counter in resolver_stats.values())
    title_search_errors, title_skip_reasons = summarize_title_search_json(rows)

    lines: list[str] = [
        "# Profile Summary",
        "",
        f"Profile TSV: {profile_tsv}",
        "",
        "## Run Overview",
        "",
        f"- Total PDFs: {total}",
        f"- Processed: {status_counts.get('processed', 0)}",
        f"- Skipped: {sum(count for status, count in status_counts.items() if status.startswith('skipped'))}",
        f"- Status error: {status_counts.get('error', 0)}",
        f"- Rows with error messages: {rows_with_errors}",
        f"- Total file wall time: {seconds(file_wall_total)}",
        f"- Total resolve time: {seconds(resolve_total)}",
        f"- Average per PDF: {seconds(avg_file_wall)}",
        f"- Median per PDF: {seconds(med_file_wall)}",
        f"- Slowest PDF: {seconds(max_file_wall)}",
        "",
        "## Quality Counts",
        "",
        "### Confidence",
        "",
    ]
    for key in ("high", "medium", "low", "(blank)"):
        if confidence_counts.get(key, 0):
            lines.append(f"- {key}: {confidence_counts[key]}")
    lines.extend([
        "",
        "### Verified",
        "",
        f"- yes: {verified} / {total}",
        f"- no: {total - verified} / {total}",
        "",
        "### Status",
        "",
    ])
    for status, count in status_counts.most_common():
        lines.append(f"- {status}: {count}")
    lines.extend([
        "",
        "### Final Source",
        "",
    ])
    for source, count in final_source_counts.most_common():
        lines.append(f"- {source}: {count}")

    lines.extend([
        "",
        "## Timing Totals",
        "",
    ])
    timing_rows = [
        [col, seconds(value), pct(value, file_wall_total)]
        for col, value in sorted(time_totals.items(), key=lambda item: item[1], reverse=True)
    ]
    lines.extend(markdown_table(["Column", "Total", "Share of File Wall"], timing_rows))

    lines.extend([
        "",
        "## Search Provider Summary",
        "",
    ])
    lines.extend(markdown_table(
        ["Provider", "Time", "Tried", "Matches", "Errors", "Match Rate"],
        provider_rows,
    ))

    lines.extend([
        "",
        "## Identifier Lookups",
        "",
        f"- Total lookup count column: {identifier_count}",
        f"- Parsed JSON lookups: {identifier_json_lookups}",
        f"- Parsed JSON matches: {identifier_matches}",
        f"- Parsed JSON no match/error: {max(0, identifier_json_lookups - identifier_matches)}",
        "",
    ])
    resolver_rows = []
    for (resolver, kind), counter in sorted(
        resolver_stats.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        resolver_rows.append([
            resolver,
            kind,
            str(counter["lookups"]),
            str(counter["matches"]),
            str(counter["errors"]),
        ])
    lines.extend(markdown_table(["Resolver", "Kind", "Lookups", "Matches", "Errors"], resolver_rows))

    lines.extend([
        "",
        "## Error Summary",
        "",
        "### Top-Level Errors",
        "",
    ])
    if top_errors:
        for message, count in top_errors.most_common(limit):
            lines.append(f"- {count}: {message}")
    else:
        lines.append("(none)")

    lines.extend([
        "",
        "### Identifier Errors",
        "",
    ])
    if identifier_errors:
        for message, count in identifier_errors.most_common(limit):
            lines.append(f"- {count}: {message}")
    else:
        lines.append("(none)")

    lines.extend([
        "",
        "### Title Search Errors",
        "",
    ])
    if title_search_errors:
        for message, count in title_search_errors.most_common(limit):
            lines.append(f"- {count}: {message}")
    else:
        lines.append("(none)")

    lines.extend([
        "",
        "### Title Search Skip Reasons",
        "",
    ])
    if title_skip_reasons:
        for message, count in title_skip_reasons.most_common(limit):
            lines.append(f"- {count}: {message}")
    else:
        lines.append("(none)")

    lines.extend([
        "",
        "## Slowest Files",
        "",
    ])
    slow_rows = []
    for row in top_rows(rows, "File Wall s", limit):
        slow_rows.append([
            seconds(as_float(row.get("File Wall s", ""))),
            row.get("Final Source", "") or "(blank)",
            row.get("Confidence", "") or "(blank)",
            row.get("Verified", "") or "(blank)",
            file_label(row.get("File Path", "")),
        ])
    lines.extend(markdown_table(["Time", "Final Source", "Confidence", "Verified", "File"], slow_rows))

    lines.extend([
        "",
        "## Slowest Elements",
        "",
    ])
    slow_columns = []
    for col in KEY_SLOW_COLUMNS:
        if col in time_cols:
            slow_columns.append(col)
    for col, _ in sorted(time_totals.items(), key=lambda item: item[1], reverse=True):
        if col not in slow_columns:
            slow_columns.append(col)

    for col in slow_columns:
        if col not in time_cols or time_totals.get(col, 0.0) <= 0:
            continue
        lines.extend([f"### {col}", ""])
        element_rows = []
        for row in top_rows(rows, col, limit):
            element_rows.append([
                seconds(as_float(row.get(col, ""))),
                row.get("Final Source", "") or "(blank)",
                row.get("Confidence", "") or "(blank)",
                row.get("Verified", "") or "(blank)",
                file_label(row.get("File Path", "")),
            ])
        lines.extend(markdown_table(["Time", "Final Source", "Confidence", "Verified", "File"], element_rows))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a papis_import profile TSV.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Config JSON path")
    parser.add_argument("--profile-tsv", default="", help="Profile TSV override")
    parser.add_argument("--dest", default="", help="Summary output path override")
    parser.add_argument("--top", type=int, default=10, help="Rows to show in top-N sections")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cfg = load_json_config(args.config, missing_ok=True)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    profile_tsv = expand_path(args.profile_tsv or str(cfg.get("profile_tsv") or DEFAULT_PROFILE_TSV))
    dest = expand_path(args.dest or DEFAULT_DEST)
    try:
        rows = read_tsv_dicts(profile_tsv)
        summary = build_summary(rows, profile_tsv, max(1, args.top))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(summary, encoding="utf-8")
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    file_wall_total = sum(as_float(row.get("File Wall s", "")) for row in rows)
    verified = sum(1 for row in rows if is_yes(row.get("Verified", "")))
    confidence_counts = Counter((row.get("Confidence", "") or "(blank)").lower() for row in rows)
    print(f"Wrote profile summary: {dest}")
    print(f"PDFs: {len(rows)}")
    print(f"Total file wall time: {seconds(file_wall_total)}")
    print(f"Verified: {verified} / {len(rows)}")
    print(
        "Confidence: "
        f"high={confidence_counts.get('high', 0)} "
        f"medium={confidence_counts.get('medium', 0)} "
        f"low={confidence_counts.get('low', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
