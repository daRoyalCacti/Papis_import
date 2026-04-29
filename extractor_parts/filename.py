"""Filename-derived metadata candidate extraction."""
from __future__ import annotations

import re
from pathlib import Path

from papis_import.core.filename_patterns import ANNA_NAME_RE, ANNA_SPLIT_RE, HEX32_RE, LEADING_SERIES_RE
from papis_import.core.identifiers import extract_identifiers, validate_isbn
from papis_import.core.text import (
    YEAR_RE,
    clean_filename_text,
    clean_text,
    first_year,
    split_authors,
    strip_trailing_title_metadata,
)
from papis_import.core.title_quality import is_journal_abbrev_title
from papis_import.models import Candidate


class FilenameExtractor:
    """Build metadata candidates from structured and heuristic filename parsing."""

    def candidate(self, path: Path) -> list[Candidate]:
        stem_raw = clean_text(path.stem)
        stem_raw = re.sub(r"\s+-\s+libgen\.[A-Za-z0-9]+$", "", stem_raw, flags=re.I)
        stem_raw = re.sub(r"\s+-\s+Anna.?s Archive(?:-\d+)?$", "", stem_raw, flags=re.I)
        stem_raw = re.sub(r"(\(\d{4}\))-(\d+)$", r"\1", stem_raw)
        stem = clean_filename_text(stem_raw)
        out: list[Candidate] = []
        dois, isbns, arxivs, stable_ids = extract_identifiers(stem_raw, stem)

        if " -- " in stem_raw:
            pieces = [clean_text(p) for p in ANNA_SPLIT_RE.split(stem_raw) if clean_text(p)]
            pieces = [p for p in pieces if not HEX32_RE.fullmatch(p) and not ANNA_NAME_RE.fullmatch(p)]
            year = ""
            isbn = ""
            nonmeta: list[str] = []
            for p in pieces:
                isbn_cand = validate_isbn(p)
                if isbn_cand and not isbn:
                    isbn = isbn_cand
                    continue
                if YEAR_RE.fullmatch(p) and not year:
                    year = p
                    continue
                if HEX32_RE.fullmatch(p):
                    continue
                nonmeta.append(clean_filename_text(p))
            title = ""
            authors: list[str] = []
            if nonmeta:
                title, stripped_year = strip_trailing_title_metadata(nonmeta[0])
                if stripped_year and not year:
                    year = stripped_year
            if len(nonmeta) >= 2 and not HEX32_RE.fullmatch(nonmeta[1]):
                authors = split_authors(nonmeta[1].replace("_", "; "))
            out.append(Candidate(
                title=title, authors=authors, year=year,
                doi=dois[0] if dois else "",
                isbn=isbn or (isbns[0] if isbns else ""),
                arxiv=arxivs[0] if arxivs else "",
                source="filename_structured", priority=40,
                notes=["parsed structured Anna/libgen filename"],
            ))
            return out

        m = re.match(
            r"^\(([^)]{1,160})\)\s*(.+?)\s+-\s+(.+?)\s*\((\d{4})(?:\s*,[^)]*)?\)?$",
            stem_raw,
        )
        if m:
            author = clean_text(m.group(2)).replace("_", "; ")
            title, stripped_year = strip_trailing_title_metadata(clean_filename_text(m.group(3)))
            out.append(Candidate(
                title=title, authors=split_authors(author),
                year=stripped_year or m.group(4),
                doi=dois[0] if dois else "",
                isbn=isbns[0] if isbns else "",
                arxiv=arxivs[0] if arxivs else "",
                source="filename_series", priority=35,
            ))
            return out

        m2 = re.match(r"^(.*?)\s+-\s+(.+)$", stem_raw)
        if m2:
            author = LEADING_SERIES_RE.sub("", clean_text(m2.group(1))).replace("_", "; ")
            title, stripped_year = strip_trailing_title_metadata(clean_filename_text(m2.group(2)))
            year = stripped_year or first_year(stem_raw)
            if author and title and len(title) > 4:
                if is_journal_abbrev_title(title):
                    out.append(Candidate(
                        title="", authors=split_authors(author), year=year,
                        doi=dois[0] if dois else "",
                        isbn=isbns[0] if isbns else "",
                        arxiv=arxivs[0] if arxivs else "",
                        source="filename_author_only", priority=70,
                        notes=["filename title was journal abbreviation, dropped"],
                    ))
                else:
                    out.append(Candidate(
                        title=title, authors=split_authors(author), year=year,
                        doi=dois[0] if dois else "",
                        isbn=isbns[0] if isbns else "",
                        arxiv=arxivs[0] if arxivs else "",
                        source="filename_author_title", priority=45,
                    ))
                    return out

        title, stripped_year = strip_trailing_title_metadata(stem)
        out.append(Candidate(
            title=title, authors=[],
            year=stripped_year or first_year(stem_raw),
            doi=dois[0] if dois else "",
            isbn=isbns[0] if isbns else "",
            arxiv=arxivs[0] if arxivs else "",
            source="filename_title_only", priority=80,
            notes=([f"jstor stable id: {stable_ids[0]}"] if stable_ids else []),
        ))
        return out

