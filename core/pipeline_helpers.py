"""Small shared helpers for pipeline and CLI decisions."""
from __future__ import annotations

import re
from pathlib import Path

from papis_import.core.text import clean_text


def confidence_rank(level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(level, 0)


def should_import(level: str, minimum: str) -> bool:
    return confidence_rank(level) >= confidence_rank(minimum)


def build_tags(staging_dir: Path, file_path: Path) -> list[str]:
    rel_parent = file_path.parent.relative_to(staging_dir)
    if str(rel_parent) == ".":
        return ["imported"]
    tags: list[str] = []
    for part in rel_parent.parts:
        if re.fullmatch(r"[0-9\-.]+", part):
            continue
        cleaned = clean_text(part.replace(",", " "))
        if cleaned:
            tags.append(cleaned)
    return tags or ["imported"]

