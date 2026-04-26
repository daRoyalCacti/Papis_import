#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def _load_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="	"))
    return rows


def _severity(row: dict[str, str]) -> tuple[int, int, int, str]:
    auto = row.get("Auto Safe", "").strip().lower() == "yes"
    verified = row.get("Verified", "").strip().lower() == "yes"
    sanity = row.get("Sanity Passed", "").strip().lower() == "yes"
    conf = row.get("Confidence", "low").strip().lower()
    rank = {"low": 0, "medium": 1, "high": 2}.get(conf, 0)
    source = row.get("Source", "")
    source_penalty = 0 if source not in {"filename_title_only", "text_header", "none", "error"} else 1
    return (1 if auto else 0, 1 if verified else 0, 1 if sanity else 0, str(rank)+str(source_penalty))


def main() -> int:
    p = argparse.ArgumentParser(description="Copy problematic PDFs referenced in a papis_import TSV to a debug folder")
    p.add_argument("--tsv", required=True, help="Input TSV (review TSV is the most useful)")
    p.add_argument("--dest", required=True, help="Destination folder")
    p.add_argument("--limit", type=int, default=10, help="Maximum number of PDFs to copy (0 = all)")
    p.add_argument("--all", action="store_true", help="Copy every row from the TSV without filtering")
    args = p.parse_args()

    tsv = Path(os.path.expanduser(args.tsv)).resolve()
    dest = Path(os.path.expanduser(args.dest)).resolve()
    rows = _load_rows(tsv)
    if not args.all:
        rows = [r for r in rows if r.get("Auto Safe", "").strip().lower() != "yes"]
    rows.sort(key=_severity)
    if args.limit > 0:
        rows = rows[:args.limit]

    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    manifest = []
    for row in rows:
        src = Path(row.get("File Path", "")).expanduser()
        if not src.exists():
            continue
        dst = dest / src.name
        if dst.exists():
            stem, suf = src.stem, src.suffix
            n = 2
            while (dest / f"{stem}_{n}{suf}").exists():
                n += 1
            dst = dest / f"{stem}_{n}{suf}"
        shutil.copy2(src, dst)
        manifest.append({
            "copied_to": str(dst),
            "source": str(src),
            "confidence": row.get("Confidence", ""),
            "verified": row.get("Verified", ""),
            "sanity": row.get("Sanity Passed", ""),
            "title": row.get("Title", ""),
            "authors": row.get("Authors", ""),
            "source_field": row.get("Source", ""),
            "vision_status": row.get("Vision Status", ""),
        })
        copied += 1

    mf = dest / "manifest.tsv"
    with mf.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()) if manifest else ["copied_to", "source"])
        w.writeheader()
        for row in manifest:
            w.writerow(row)

    print(f"Copied {copied} PDF(s) to {dest}")
    print(f"Manifest: {mf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
