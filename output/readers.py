"""Readers for previously-written result TSVs (used by --retry-unverified)."""
from __future__ import annotations

import csv
from pathlib import Path

from papis_import.models import Metadata, Record


def load_previous_tsv(*tsv_paths: Path) -> tuple[dict[str, Record], int]:
    """Load trusted rows from one or more previous result TSVs.

    Returns (skip_dict, legacy_count):
      skip_dict    — {file_path_str: Record} for rows previously marked auto-safe
      legacy_count — verified rows from old-format TSVs (no Auto Safe column)
                     that will be re-processed to apply new pipeline checks

    Rows without Auto Safe = yes and rows from legacy TSVs are excluded so
    that pipeline improvements get applied on the next run.
    Missing files are silently ignored.
    """
    skip: dict[str, Record] = {}
    legacy_count = 0
    for tsv_path in tsv_paths:
        if not tsv_path or not tsv_path.exists():
            continue
        with tsv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            has_auto_safe = "Auto Safe" in fieldnames
            for row in reader:
                if not has_auto_safe and row.get("Verified", "").strip().lower() == "yes":
                    legacy_count += 1
                    continue
                if row.get("Auto Safe", "").strip().lower() != "yes":
                    continue
                path_str = row.get("File Path", "").strip()
                if not path_str:
                    continue
                try:
                    sanity_score = float(row.get("Sanity Score", "0") or "0")
                except ValueError:
                    sanity_score = 0.0
                meta = Metadata(
                    title=row.get("Title", ""),
                    authors=[a.strip() for a in row.get("Authors", "").split(";") if a.strip()],
                    year=row.get("Year", ""),
                    doi=row.get("DOI", ""),
                    isbn=row.get("ISBN", ""),
                    arxiv=row.get("arXiv", ""),
                    source=row.get("Source", ""),
                    confidence=row.get("Confidence", "low"),
                    verified=row.get("Verified", "").strip().lower() == "yes",
                    sanity_passed=row.get("Sanity Passed", "").strip().lower() == "yes",
                    sanity_score=sanity_score,
                    auto_safe=True,
                    soft_auto=row.get("Soft Auto", "").strip().lower() == "yes",
                    soft_auto_reasons=[
                        r.strip() for r in row.get("Soft Auto Reasons", "").split("|") if r.strip()
                    ],
                    needs_ocr=row.get("Needs OCR", "").strip().lower() == "yes",
                    notes=[n.strip() for n in row.get("Notes", "").split("|") if n.strip()],
                )
                tags = [t.strip() for t in row.get("Tags", "").split(",") if t.strip()]
                rec = Record(
                    path=Path(path_str),
                    tags=tags,
                    result=meta,
                    suggested_command=row.get("Suggested Command", ""),
                    imported=row.get("Imported", "").strip().lower() == "yes",
                    error=row.get("Error", ""),
                )
                # First TSV wins — auto TSV is passed first so a later
                # review-TSV entry for the same path won't override it.
                skip.setdefault(path_str, rec)
    return skip, legacy_count
