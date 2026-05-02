from __future__ import annotations

from papis_import.models import Candidate
from papis_import.utils import normalize_title


STRONG_SEARCHABLE_SOURCES = {
    "filename_structured",
    "filename_author_title",
    "filename_series",
    "pdfinfo",
    "pdf_metadata",
    "xmp",
    "grobid",
}


def has_strong_searchable_candidate(candidates: list[Candidate]) -> bool:
    """Return True when local candidates are strong enough to skip vision."""
    for c in candidates:
        if not c.title:
            continue
        words = normalize_title(c.title).split()
        if len(words) < 4:
            continue
        if c.source in STRONG_SEARCHABLE_SOURCES:
            return True
        if c.source.startswith("llm:"):
            return True
    return False
