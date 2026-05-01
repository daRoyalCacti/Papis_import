#!/usr/bin/env python3
"""Create a review bundle for papis_import rows that need manual inspection."""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from io_utils import (  # noqa: E402
    CONFIDENCE_RANK,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DEBUG_TSV,
    DEFAULT_REVIEW_TSV,
    PROJECT_ROOT,
    expand_path,
    is_yes,
    load_json_config,
    read_debug_jsonl,
    read_tsv_dicts,
)


DEFAULT_DEST = PROJECT_ROOT / "out" / "review_bundle"

BAD_SOURCES = {"error", "none", "filename_title_only", "text_header", "synthesized"}

MANIFEST_FIELDS = [
    "Review ID",
    "Review PDF",
    "Original PDF",
    "Title",
    "Authors",
    "Year",
    "DOI",
    "ISBN",
    "arXiv",
    "Confidence",
    "Verified",
    "Sanity Passed",
    "Sanity Score",
    "Auto Safe",
    "Needs OCR",
    "Source",
    "Final Source",
    "Vision Used",
    "Vision Trigger",
    "Vision Status",
    "Vision Error",
    "GROBID Used",
    "Candidate Sources",
    "Notes",
    "Suggested Command",
    "Error",
]


@dataclass
class BundleItem:
    original_index: int
    review: dict[str, str]
    debug: dict[str, str]
    review_id: str = ""
    review_pdf: str = ""
    copied: bool = False
    copy_error: str = ""

    def value(self, key: str) -> str:
        value = self.review.get(key, "")
        if value:
            return value
        return self.debug.get(key, "")


def index_debug_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        file_path = row.get("File Path", "")
        if file_path:
            indexed[file_path] = row
    return indexed


def confidence_rank(value: str) -> int:
    return CONFIDENCE_RANK.get(value.strip().lower(), 0)


def review_sort_key(item: BundleItem) -> tuple[int, int, int, int, int, int, int, int]:
    source_path = Path(item.value("File Path")).expanduser()
    source = item.value("Source").strip().lower() or item.value("Final Source").strip().lower()
    return (
        0 if not source_path.exists() else 1,
        0 if item.value("Error").strip() else 1,
        0 if is_yes(item.value("Needs OCR")) else 1,
        0 if not is_yes(item.value("Sanity Passed")) else 1,
        0 if not is_yes(item.value("Verified")) else 1,
        confidence_rank(item.value("Confidence")),
        0 if source in BAD_SOURCES else 1,
        item.original_index,
    )


def sanitize_stem(value: str) -> str:
    value = value.strip() or "pdf"
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.ASCII)
    value = re.sub(r"_+", "_", value).strip("._-")
    return (value or "pdf")[:100]


def unique_pdf_name(dest: Path, used: set[str], source: Path, review_id: str) -> str:
    stem = sanitize_stem(source.stem)
    suffix = source.suffix if source.suffix.lower() == ".pdf" else ".pdf"
    base = f"{review_id}__{stem}"
    name = f"{base}{suffix}"
    n = 2
    while name in used or (dest / name).exists():
        name = f"{base}__{n}{suffix}"
        n += 1
    used.add(name)
    return name


def prepare_dest(dest: Path, force: bool) -> None:
    if not dest.exists():
        dest.mkdir(parents=True)
        return
    if not dest.is_dir():
        raise RuntimeError(f"destination exists and is not a directory: {dest}")
    existing = list(dest.iterdir())
    if existing and not force:
        raise RuntimeError(f"destination is not empty; pass --force to overwrite files: {dest}")
    if force:
        for path in existing:
            if path.is_dir():
                raise RuntimeError(f"refusing to remove subdirectory in bundle destination: {path}")
            path.unlink()


def write_manifest(path: Path, items: list[BundleItem]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow({
                "Review ID": item.review_id,
                "Review PDF": item.review_pdf,
                "Original PDF": item.value("File Path"),
                "Title": item.value("Title"),
                "Authors": item.value("Authors"),
                "Year": item.value("Year"),
                "DOI": item.value("DOI"),
                "ISBN": item.value("ISBN"),
                "arXiv": item.value("arXiv"),
                "Confidence": item.value("Confidence"),
                "Verified": item.value("Verified"),
                "Sanity Passed": item.value("Sanity Passed"),
                "Sanity Score": item.value("Sanity Score"),
                "Auto Safe": item.value("Auto Safe"),
                "Needs OCR": item.value("Needs OCR"),
                "Source": item.value("Source"),
                "Final Source": item.value("Final Source"),
                "Vision Used": item.value("Vision Used"),
                "Vision Trigger": item.value("Vision Trigger"),
                "Vision Status": item.value("Vision Status"),
                "Vision Error": item.value("Vision Error"),
                "GROBID Used": item.value("GROBID Used"),
                "Candidate Sources": item.value("Candidate Sources"),
                "Notes": item.value("Notes"),
                "Suggested Command": item.value("Suggested Command"),
                "Error": item.copy_error or item.value("Error"),
            })


def write_report(path: Path, items: list[BundleItem], review_tsv: Path, debug_tsv: Path | None) -> None:
    copied = sum(1 for item in items if item.copied)
    needs_ocr = sum(1 for item in items if is_yes(item.value("Needs OCR")))
    unverified = sum(1 for item in items if not is_yes(item.value("Verified")))
    sanity_failed = sum(1 for item in items if not is_yes(item.value("Sanity Passed")))

    lines = [
        "# Review Bundle",
        "",
        f"Review TSV: {review_tsv}",
        f"Debug TSV: {debug_tsv if debug_tsv else '(not used)'}",
        f"Total rows: {len(items)}",
        f"Copied PDFs: {copied}",
        f"Needs OCR: {needs_ocr}",
        f"Unverified: {unverified}",
        f"Sanity failed: {sanity_failed}",
        "",
    ]
    for item in items:
        title = item.value("Title") or "(no title)"
        authors = item.value("Authors") or "(no authors)"
        lines.extend([
            f"## {item.review_id} - {Path(item.value('File Path')).name}",
            "",
            f"Review PDF: {item.review_pdf or '(not copied)'}",
            f"Original: {item.value('File Path')}",
            f"Title: {title}",
            f"Authors: {authors}",
            f"Year: {item.value('Year')}",
            f"Confidence: {item.value('Confidence')}",
            f"Verified: {item.value('Verified')}",
            f"Sanity: {item.value('Sanity Passed')}, {item.value('Sanity Score')}",
            f"Needs OCR: {item.value('Needs OCR')}",
            f"Source: {item.value('Source')}",
            f"Final Source: {item.value('Final Source')}",
            f"Vision: {item.value('Vision Used')}, {item.value('Vision Status')}",
        ])
        if item.value("Notes"):
            lines.append(f"Notes: {item.value('Notes')}")
        if item.copy_error:
            lines.append(f"Copy error: {item.copy_error}")
        if item.value("Suggested Command"):
            lines.extend(["", "Command:", "", "```sh", item.value("Suggested Command"), "```"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_or_link_pdf(item: BundleItem, dest: Path, mode: str, used_names: set[str], no_pdfs: bool) -> None:
    item.review_id = f"{int(item.review_id):03d}"
    source = Path(item.value("File Path")).expanduser()
    if no_pdfs:
        return
    if not source.exists():
        item.copy_error = f"source PDF not found: {source}"
        return
    name = unique_pdf_name(dest, used_names, source, item.review_id)
    target = dest / name
    try:
        if mode == "link":
            target.symlink_to(source.resolve())
        else:
            shutil.copy2(source, target)
        item.review_pdf = name
        item.copied = True
    except Exception as exc:
        item.copy_error = str(exc)


def compact_line(item: BundleItem) -> str:
    conf = (item.value("Confidence") or "?").upper()[:3]
    flags = []
    if not is_yes(item.value("Verified")):
        flags.append("unverified")
    if is_yes(item.value("Needs OCR")):
        flags.append("OCR")
    if not is_yes(item.value("Sanity Passed")):
        flags.append("sanity-fail")
    flag_text = ", ".join(flags) if flags else "ok"
    title = item.value("Title") or "(no title)"
    authors = item.value("Authors") or "(no authors)"
    return (
        f"{item.review_id}  {conf:<3}  {flag_text:<28}  {item.review_pdf or '(not copied)'}\n"
        f"     {title} - {authors}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy review PDFs into a numbered bundle and write a review manifest.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Config JSON path")
    parser.add_argument("--review-tsv", default="", help="Review TSV override")
    parser.add_argument("--debug-tsv", default="", help="Debug TSV override")
    parser.add_argument("--dest", default="", help="Bundle output directory override")
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to bundle; default/all = 0")
    parser.add_argument("--sort", choices=["review", "tsv"], default="review", help="Row order")
    parser.add_argument("--link", action="store_true", help="Symlink PDFs instead of copying")
    parser.add_argument("--no-pdfs", action="store_true", help="Only write manifest and report")
    parser.add_argument("--force", action="store_true", help="Overwrite files in the destination")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cfg = load_json_config(args.config, missing_ok=True)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    review_tsv = expand_path(args.review_tsv or str(cfg.get("review_tsv") or DEFAULT_REVIEW_TSV))
    debug_value = args.debug_tsv or str(cfg.get("debug_tsv") or DEFAULT_DEBUG_TSV)
    debug_tsv = expand_path(debug_value) if debug_value else None
    dest = expand_path(args.dest or DEFAULT_DEST)

    try:
        review_rows = read_tsv_dicts(review_tsv, required=True)
        debug_rows = read_debug_jsonl(debug_tsv, required=False) if debug_tsv else []
        prepare_dest(dest, force=args.force)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    debug_by_path = index_debug_rows(debug_rows)
    items = [
        BundleItem(
            original_index=idx,
            review=row,
            debug=debug_by_path.get(row.get("File Path", ""), {}),
        )
        for idx, row in enumerate(review_rows, start=1)
    ]
    if args.sort == "review":
        items.sort(key=review_sort_key)
    if args.limit > 0:
        items = items[:args.limit]

    used_names: set[str] = set()
    mode = "link" if args.link else "copy"
    for idx, item in enumerate(items, start=1):
        item.review_id = f"{idx:03d}"
        copy_or_link_pdf(item, dest, mode, used_names, args.no_pdfs)

    manifest = dest / "manifest.tsv"
    report = dest / "review.md"
    write_manifest(manifest, items)
    write_report(report, items, review_tsv, debug_tsv if debug_rows else None)

    copied = sum(1 for item in items if item.copied)
    print(f"Wrote review bundle: {dest}")
    print(f"Rows: {len(items)}")
    if not args.no_pdfs:
        print(f"PDFs: {copied}")
    print(f"Manifest: {manifest}")
    print(f"Report: {report}")
    print()
    for item in items:
        print(compact_line(item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
