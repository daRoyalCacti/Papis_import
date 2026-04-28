#!/usr/bin/env python3
"""Create an audit bundle for papis_import soft auto-accept rows."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from io_utils import (  # noqa: E402
    CONFIDENCE_RANK,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DEBUG_TSV,
    DEFAULT_SOFT_TSV,
    PROJECT_ROOT,
    as_float,
    expand_path,
    is_yes,
    load_json_config,
    read_tsv_dicts,
)


DEFAULT_DEST = PROJECT_ROOT / "out" / "soft_accept_bundle"

RISKY_FINAL_SOURCES = {"synthesized", "text_header", "pdfinfo", "grobid"}
STRONG_SOURCE_RE = re.compile(r"title corroborated by (\d+) strong sources")

MANIFEST_FIELDS = [
    "Soft ID",
    "Soft PDF",
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
    "Soft Auto",
    "Soft Auto Reasons",
    "Candidate Sources",
    "Local Best Source",
    "GROBID Title",
    "GROBID Authors",
    "GROBID Year",
    "Text LLM Title",
    "Text LLM Authors",
    "Text LLM Year",
    "Vision Title",
    "Vision Authors",
    "Vision Year",
    "Notes",
    "Suggested Command",
    "Copy Error",
    "Audit Verdict",
    "Audit Notes",
]


@dataclass
class SoftItem:
    original_index: int
    soft: dict[str, str]
    debug: dict[str, str]
    soft_id: str = ""
    soft_pdf: str = ""
    copied: bool = False
    copy_error: str = ""

    def value(self, key: str) -> str:
        value = self.soft.get(key, "")
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


def auto_derived_path(path: str | Path, suffix: str) -> Path:
    base = expand_path(path)
    return base.with_name(base.stem + suffix + base.suffix)


def resolve_soft_tsv(args: argparse.Namespace, cfg: dict[str, object]) -> Path:
    if args.soft_tsv:
        return expand_path(args.soft_tsv)
    if cfg.get("soft_tsv"):
        return expand_path(str(cfg["soft_tsv"]))
    if cfg.get("tsv"):
        return auto_derived_path(str(cfg["tsv"]), "_soft")
    return expand_path(DEFAULT_SOFT_TSV)


def soft_source_count(item: SoftItem) -> int:
    match = STRONG_SOURCE_RE.search(item.value("Soft Auto Reasons"))
    if not match:
        return 0
    return int(match.group(1))


def title_word_count(item: SoftItem) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", item.value("Title")))


def author_count(item: SoftItem) -> int:
    authors = item.value("Authors")
    if not authors.strip():
        return 0
    return len([part for part in re.split(r"\s*;\s*|\s+ and \s+", authors) if part.strip()])


def has_identifier(item: SoftItem) -> bool:
    return bool(item.value("DOI") or item.value("ISBN") or item.value("arXiv"))


def confidence_rank(value: str) -> int:
    return CONFIDENCE_RANK.get(value.strip().lower(), 0)


def soft_risk_sort_key(item: SoftItem) -> tuple[int, int, int, float, int, int, int, int, int, int]:
    source_path = Path(item.value("File Path")).expanduser()
    final_source = item.value("Final Source").strip().lower() or item.value("Source").strip().lower()
    source_count = soft_source_count(item)
    return (
        0 if not source_path.exists() else 1,
        0 if is_yes(item.value("Needs OCR")) else 1,
        0 if not is_yes(item.value("Sanity Passed")) else 1,
        as_float(item.value("Sanity Score"), 0.0),
        0 if source_count and source_count <= 2 else 1,
        0 if not has_identifier(item) else 1,
        0 if final_source in RISKY_FINAL_SOURCES else 1,
        title_word_count(item),
        author_count(item),
        item.original_index,
    )


def sanitize_stem(value: str) -> str:
    value = value.strip() or "pdf"
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.ASCII)
    value = re.sub(r"_+", "_", value).strip("._-")
    return (value or "pdf")[:100]


def unique_pdf_name(dest: Path, used: set[str], source: Path, soft_id: str) -> str:
    stem = sanitize_stem(source.stem)
    suffix = source.suffix if source.suffix.lower() == ".pdf" else ".pdf"
    base = f"{soft_id}__{stem}"
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


def copy_or_link_pdf(item: SoftItem, dest: Path, mode: str, used_names: set[str], no_pdfs: bool) -> None:
    source = Path(item.value("File Path")).expanduser()
    if no_pdfs:
        return
    if not source.exists():
        item.copy_error = f"source PDF not found: {source}"
        return
    name = unique_pdf_name(dest, used_names, source, item.soft_id)
    target = dest / name
    try:
        if mode == "link":
            target.symlink_to(source.resolve())
        else:
            shutil.copy2(source, target)
        item.soft_pdf = name
        item.copied = True
    except Exception as exc:
        item.copy_error = str(exc)


def parse_candidates_json(value: str) -> list[dict[str, Any]]:
    text = value.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def candidate_evidence_lines(item: SoftItem) -> list[str]:
    lines: list[str] = []
    named_sources = [
        ("GROBID", "GROBID Title", "GROBID Authors", "GROBID Year"),
        ("Text LLM", "Text LLM Title", "Text LLM Authors", "Text LLM Year"),
        ("Vision", "Vision Title", "Vision Authors", "Vision Year"),
    ]
    for label, title_key, authors_key, year_key in named_sources:
        title = item.value(title_key)
        authors = item.value(authors_key)
        year = item.value(year_key)
        if title or authors or year:
            lines.append(f"- {label}: {title or '(no title)'} - {authors or '(no authors)'} {year}".rstrip())

    candidates = parse_candidates_json(item.debug.get("Candidates JSON", ""))
    for cand in candidates[:8]:
        source = str(cand.get("source") or "(unknown)")
        title = str(cand.get("title") or "(no title)")
        authors = cand.get("authors") or []
        if isinstance(authors, list):
            authors_text = "; ".join(str(author) for author in authors if author)
        else:
            authors_text = str(authors)
        year = str(cand.get("year") or "")
        lines.append(f"- {source}: {title} - {authors_text or '(no authors)'} {year}".rstrip())

    if not lines and item.value("Candidate Sources"):
        lines.append(f"- Candidate sources: {item.value('Candidate Sources')}")
    return lines


def write_manifest(path: Path, items: list[SoftItem]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow({
                "Soft ID": item.soft_id,
                "Soft PDF": item.soft_pdf,
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
                "Soft Auto": item.value("Soft Auto"),
                "Soft Auto Reasons": item.value("Soft Auto Reasons"),
                "Candidate Sources": item.value("Candidate Sources"),
                "Local Best Source": item.value("Local Best Source"),
                "GROBID Title": item.value("GROBID Title"),
                "GROBID Authors": item.value("GROBID Authors"),
                "GROBID Year": item.value("GROBID Year"),
                "Text LLM Title": item.value("Text LLM Title"),
                "Text LLM Authors": item.value("Text LLM Authors"),
                "Text LLM Year": item.value("Text LLM Year"),
                "Vision Title": item.value("Vision Title"),
                "Vision Authors": item.value("Vision Authors"),
                "Vision Year": item.value("Vision Year"),
                "Notes": item.value("Notes"),
                "Suggested Command": item.value("Suggested Command"),
                "Copy Error": item.copy_error,
                "Audit Verdict": "",
                "Audit Notes": "",
            })


def write_report(path: Path, items: list[SoftItem], soft_tsv: Path, debug_tsv: Path | None) -> None:
    copied = sum(1 for item in items if item.copied)
    missing = sum(1 for item in items if item.copy_error)
    needs_ocr = sum(1 for item in items if is_yes(item.value("Needs OCR")))
    auto_safe = sum(1 for item in items if is_yes(item.value("Auto Safe")))
    sanity_failed = sum(1 for item in items if not is_yes(item.value("Sanity Passed")))
    final_sources = Counter(item.value("Final Source") or item.value("Source") or "(blank)" for item in items)
    source_counts = Counter(soft_source_count(item) or "unknown" for item in items)

    lines = [
        "# Soft Accept Bundle",
        "",
        f"Soft TSV: {soft_tsv}",
        f"Debug TSV: {debug_tsv if debug_tsv else '(not used)'}",
        f"Total rows: {len(items)}",
        f"Copied PDFs: {copied}",
        f"Missing PDFs: {missing}",
        f"Needs OCR: {needs_ocr}",
        f"Sanity failed: {sanity_failed}",
        f"Auto Safe rows in soft TSV: {auto_safe}",
        "",
        "## Final Source Breakdown",
        "",
    ]
    for source, count in final_sources.most_common():
        lines.append(f"- {source}: {count}")
    lines.extend(["", "## Corroborator Count Breakdown", ""])
    for count, n in source_counts.most_common():
        lines.append(f"- {count}: {n}")
    lines.append("")

    for item in items:
        title = item.value("Title") or "(no title)"
        authors = item.value("Authors") or "(no authors)"
        lines.extend([
            f"## {item.soft_id} - {Path(item.value('File Path')).name}",
            "",
            f"Bundle PDF: {item.soft_pdf or '(not copied)'}",
            f"Original: {item.value('File Path')}",
            f"Final: {title} - {authors}",
            f"Year: {item.value('Year')}",
            f"Confidence: {item.value('Confidence')}",
            f"Verified: {item.value('Verified')}",
            f"Sanity: {item.value('Sanity Passed')}, {item.value('Sanity Score')}",
            f"Needs OCR: {item.value('Needs OCR')}",
            f"Auto Safe: {item.value('Auto Safe')}",
            f"Source: {item.value('Source')}",
            f"Final Source: {item.value('Final Source')}",
            f"Soft reason: {item.value('Soft Auto Reasons')}",
            "",
            "Evidence:",
        ])
        evidence = candidate_evidence_lines(item)
        lines.extend(evidence or ["- (no debug evidence found)"])
        if item.value("Notes"):
            lines.extend(["", f"Notes: {item.value('Notes')}"])
        if item.copy_error:
            lines.extend(["", f"Copy error: {item.copy_error}"])
        if item.value("Suggested Command"):
            lines.extend(["", "Command:", "", "```sh", item.value("Suggested Command"), "```"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def compact_line(item: SoftItem) -> str:
    conf = (item.value("Confidence") or "?").upper()[:3]
    flags = []
    if item.copy_error:
        flags.append("missing-pdf")
    if is_yes(item.value("Needs OCR")):
        flags.append("OCR")
    if not is_yes(item.value("Sanity Passed")):
        flags.append("sanity-fail")
    if not has_identifier(item):
        flags.append("no-id")
    count = soft_source_count(item)
    if count:
        flags.append(f"{count}src")
    if is_yes(item.value("Auto Safe")):
        flags.append("auto-safe")
    flag_text = ", ".join(flags) if flags else "ok"
    title = item.value("Title") or "(no title)"
    authors = item.value("Authors") or "(no authors)"
    return (
        f"{item.soft_id}  {conf:<3}  {flag_text:<34}  {item.soft_pdf or '(not copied)'}\n"
        f"     {title} - {authors}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy soft auto-accept PDFs into a numbered audit bundle.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Config JSON path")
    parser.add_argument("--soft-tsv", default="", help="Soft auto-accept TSV override")
    parser.add_argument("--debug-tsv", default="", help="Debug TSV override")
    parser.add_argument("--dest", default="", help="Bundle output directory override")
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to bundle; default/all = 0")
    parser.add_argument("--sort", choices=["risk", "tsv"], default="risk", help="Row order")
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

    soft_tsv = resolve_soft_tsv(args, cfg)
    debug_value = args.debug_tsv or str(cfg.get("debug_tsv") or DEFAULT_DEBUG_TSV)
    debug_tsv = expand_path(debug_value) if debug_value else None
    dest = expand_path(args.dest or str(cfg.get("soft_bundle_dest") or DEFAULT_DEST))

    try:
        soft_rows = read_tsv_dicts(soft_tsv, required=True)
        debug_rows = read_tsv_dicts(debug_tsv, required=False) if debug_tsv else []
        prepare_dest(dest, force=args.force)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    soft_rows = [row for row in soft_rows if is_yes(row.get("Soft Auto", "yes"))]
    debug_by_path = index_debug_rows(debug_rows)
    items = [
        SoftItem(
            original_index=idx,
            soft=row,
            debug=debug_by_path.get(row.get("File Path", ""), {}),
        )
        for idx, row in enumerate(soft_rows, start=1)
    ]
    if args.sort == "risk":
        items.sort(key=soft_risk_sort_key)
    if args.limit > 0:
        items = items[:args.limit]

    used_names: set[str] = set()
    mode = "link" if args.link else "copy"
    for idx, item in enumerate(items, start=1):
        item.soft_id = f"{idx:03d}"
        copy_or_link_pdf(item, dest, mode, used_names, args.no_pdfs)

    manifest = dest / "manifest.tsv"
    report = dest / "soft_accept.md"
    write_manifest(manifest, items)
    write_report(report, items, soft_tsv, debug_tsv if debug_rows else None)

    copied = sum(1 for item in items if item.copied)
    print(f"Wrote soft accept bundle: {dest}")
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
