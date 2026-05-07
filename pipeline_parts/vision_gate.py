from __future__ import annotations

from papis_import.core.title_quality import is_series_page_title
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
    """Return True when local candidates are strong enough to skip vision.

    Series-page banners ("Lecture Notes in Mathematics", "Springer Series in …")
    are excluded even if they come from a normally-trusted source — they are not
    the book's title and must not prevent vision from reading the actual title page.
    """
    for c in candidates:
        if not c.title:
            continue
        words = normalize_title(c.title).split()
        if len(words) < 4:
            continue
        if is_series_page_title(c.title):
            continue
        if c.source in STRONG_SEARCHABLE_SOURCES:
            return True
        if c.source.startswith("llm:"):
            return True
    return False
