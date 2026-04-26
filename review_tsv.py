#!/usr/bin/env python3
"""
review_tsv.py — show TSV rows that need manual attention after a dry run.

Usage:
    python review_tsv.py ~/Documents/papis_import.tsv
    python review_tsv.py ~/Documents/papis_import.tsv --confidence low
    python review_tsv.py ~/Documents/papis_import.tsv --unverified
    python review_tsv.py ~/Documents/papis_import.tsv --export fix_me.tsv

Filtering options (can be combined — rows matching ANY filter are shown):
    --confidence low|medium   Show rows at or below this confidence level
    --unverified              Show rows where verified=no
    --no-title                Show rows with an empty or filename-only title
    --no-author               Show rows with no author
    --source TEXT             Show rows whose Source column contains TEXT
                              (e.g. --source text_header  or  --source filename)
    --all-problems            Equivalent to --unverified --no-title --no-author
                              (the most useful default for a first pass)

Output options:
    --export PATH             Write filtered rows to a new TSV file
    --limit N                 Show at most N rows (default: 50)
    --fields title,author     Comma-separated columns to print (default: a
                              compact summary)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


# Header-name helpers so the script survives added columns.
_DEFAULT_COLS = {
    "path": "File Path",
    "tags": "Tags",
    "source": "Source",
    "confidence": "Confidence",
    "verified": "Verified",
    "title": "Title",
    "authors": "Authors",
    "year": "Year",
    "doi": "DOI",
    "isbn": "ISBN",
    "arxiv": "arXiv",
    "notes": "Notes",
    "imported": "Imported",
    "error": "Error",
    "command": "Suggested Command",
    "vision_used": "Vision Used",
    "vision_status": "Vision Status",
    "final_source": "Final Source",
}

_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def _cell(row: list[str], cols: dict[str, int], key: str) -> str:
    idx = cols.get(key)
    return row[idx] if idx is not None and idx < len(row) else ""


def _rank(row: list[str], cols: dict[str, int]) -> int:
    return _CONF_RANK.get(_cell(row, cols, "confidence"), 0)


def _load(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _is_filename_only_title(title: str, path: str) -> bool:
    stem = Path(path).stem
    return title.strip().lower() in (stem.lower(), stem.lower().replace("_", " "))


def _matches(row: list[str], cols: dict[str, int], args: argparse.Namespace) -> bool:
    title    = _cell(row, cols, "title")
    authors  = _cell(row, cols, "authors")
    source   = _cell(row, cols, "source")
    verified = _cell(row, cols, "verified")
    path     = _cell(row, cols, "path")

    if args.unverified or args.all_problems:
        if verified.strip().lower() == "no":
            return True

    if args.no_title or args.all_problems:
        if not title.strip() or _is_filename_only_title(title, path):
            return True

    if args.no_author or args.all_problems:
        if not authors.strip():
            return True

    if args.confidence is not None:
        if _rank(row, cols) <= _CONF_RANK.get(args.confidence, 0):
            return True

    if args.source:
        if args.source.lower() in source.lower():
            return True

    return False


def _fmt_row(row: list[str], cols: dict[str, int], idx: int) -> str:
    conf = _cell(row, cols, "confidence")
    ver  = _cell(row, cols, "verified")
    conf_tag = {"high": "HIGH", "medium": "MED ", "low": "LOW "}.get(conf, conf.upper()[:4])
    ver_tag  = "✓" if ver.strip().lower() == "yes" else "✗"

    filename = Path(_cell(row, cols, "path")).name
    title    = _cell(row, cols, "title")  or "(no title)"
    authors  = _cell(row, cols, "authors") or "(no authors)"
    year     = _cell(row, cols, "year")
    source   = _cell(row, cols, "source")
    doi      = _cell(row, cols, "doi")
    isbn     = _cell(row, cols, "isbn")
    arxiv    = _cell(row, cols, "arxiv")
    notes    = _cell(row, cols, "notes")

    ident = doi or isbn or arxiv or ""
    lines = [
        f"[{idx:4d}] {conf_tag} {ver_tag}  {filename}",
        f"       Title  : {title[:100]}",
        f"       Authors: {authors[:100]}",
    ]
    if year:
        lines.append(f"       Year   : {year}")
    if ident:
        lines.append(f"       ID     : {ident}")
    lines.append(f"       Source : {source}")
    if notes:
        lines.append(f"       Notes  : {notes[:120]}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Review TSV output from papis_import dry runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[0].strip(),
    )
    p.add_argument("tsv", help="TSV file from papis_import --dry-run")
    p.add_argument("--confidence", choices=["low", "medium"],
                   help="Show rows at or below this confidence")
    p.add_argument("--unverified",   action="store_true",
                   help="Show unverified rows")
    p.add_argument("--no-title",     action="store_true",
                   help="Show rows with no real title")
    p.add_argument("--no-author",    action="store_true",
                   help="Show rows with no author")
    p.add_argument("--all-problems", action="store_true",
                   help="Show all obviously problematic rows (recommended first pass)")
    p.add_argument("--source",       default="",
                   help="Show rows whose Source column contains this string")
    p.add_argument("--export",       default="",
                   help="Write matching rows to a new TSV file")
    p.add_argument("--limit",        type=int, default=50,
                   help="Max rows to print (default 50; use 0 for all)")
    args = p.parse_args()

    # Default: --all-problems if no filter specified
    if not any([args.confidence, args.unverified, args.no_title,
                args.no_author, args.all_problems, args.source]):
        args.all_problems = True

    tsv_path = Path(os.path.expanduser(args.tsv))
    if not tsv_path.exists():
        print(f"[error] file not found: {tsv_path}", file=sys.stderr)
        return 1

    header, rows = _load(tsv_path)
    cols = {k: header.index(v) for k, v in _DEFAULT_COLS.items() if v in header}
    if not rows:
        print("TSV is empty.")
        return 0

    matching = [row for row in rows if _matches(row, cols, args)]
    total    = len(rows)

    print(f"Total rows : {total}")
    print(f"Matching   : {len(matching)}")
    print(f"High conf  : {sum(1 for r in rows if _cell(r, cols, "confidence") == "high")}")
    print(f"Verified   : {sum(1 for r in rows if _cell(r, cols, "verified").strip().lower() == "yes")}")
    print("-" * 60)

    limit    = args.limit if args.limit > 0 else len(matching)
    shown    = matching[:limit]
    for i, row in enumerate(shown, start=1):
        print(_fmt_row(row, cols, i))
        print()

    if len(matching) > limit:
        print(f"… and {len(matching) - limit} more (increase --limit to see them)")

    if args.export:
        out = Path(os.path.expanduser(args.export))
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(header)
            w.writerows(matching)
        print(f"\nExported {len(matching)} rows to: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
