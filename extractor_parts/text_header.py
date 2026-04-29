from __future__ import annotations

import re

from papis_import.core.identifiers import extract_identifiers
from papis_import.core.text import YEAR_RE, clean_text, first_year, split_authors, strip_footnote_markers
from papis_import.core.title_quality import is_journal_header_title
from papis_import.models import Candidate
from papis_import.utils import NOISE_LINE_PATTERNS


def _is_all_caps_phrase(line: str) -> bool:
    """Return True when *line* looks like an all-caps subtitle."""
    stripped = line.strip()
    if not stripped:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters or not all(c.isupper() for c in letters):
        return False
    if "," in stripped:
        return False
    if re.search(r"\b[A-Z]\.\s", stripped):
        return False
    return len(stripped.split()) >= 2


class TextHeaderExtractor:
    def candidate(self, text: str) -> list[Candidate]:
        lines = [clean_text(ln) for ln in text.splitlines() if clean_text(ln)]
        kept: list[str] = []
        for line in lines[:80]:
            if any(p.search(line) for p in NOISE_LINE_PATTERNS):
                if line.lower().startswith("author(s):"):
                    kept.append(clean_text(line.split(":", 1)[1]))
                continue
            kept.append(line)
        if not kept:
            return []

        title = ""
        authors: list[str] = []
        year = first_year(" ".join(kept[:20]))

        for i, line in enumerate(kept[:20]):
            if len(line) < 6 or len(line) > 220:
                continue
            if is_journal_header_title(line):
                continue
            conf_m = re.match(r"^published as a conference paper at [^:]+:\s*(.+)$", line, re.I)
            if conf_m:
                line = conf_m.group(1).strip()
                if not line:
                    continue
            if re.fullmatch(r"[A-Z\s\-:;,.]{4,}", line):
                title = line.title()
                author_offset = 1
                if i + 1 < len(kept) and _is_all_caps_phrase(kept[i + 1]):
                    title = title.rstrip(":") + ": " + kept[i + 1].title()
                    author_offset = 2
                if i + author_offset < len(kept):
                    raw_auth = kept[i + author_offset]
                    authors = [strip_footnote_markers(a) for a in split_authors(raw_auth)]
                    authors = [a for a in authors if a]
                break
            if line.count(" ") >= 2 and not YEAR_RE.fullmatch(line) and not line.lower().startswith("proceedings"):
                title = line
                if i + 1 < len(kept) and _is_all_caps_phrase(kept[i + 1]):
                    title = title.rstrip(":") + ": " + kept[i + 1].title()
                    if i + 2 < len(kept):
                        raw_auth = kept[i + 2]
                        if len(raw_auth) < 120:
                            authors = [strip_footnote_markers(a) for a in split_authors(raw_auth)]
                            authors = [a for a in authors if a]
                elif i + 1 < len(kept):
                    maybe = kept[i + 1]
                    if len(maybe) < 120:
                        authors = [strip_footnote_markers(a) for a in split_authors(maybe)]
                        authors = [a for a in authors if a]
                break
        if not title:
            title = kept[0][:220]
            authors = [strip_footnote_markers(a) for a in split_authors(kept[1])] if len(kept) > 1 else []
            authors = [a for a in authors if a]

        if is_journal_header_title(title):
            title = ""

        cand = Candidate(title=title, authors=authors, year=year, source="text_header", priority=90)
        dois, isbns, arxivs, stable_ids = extract_identifiers(text)
        cand.doi = dois[0] if dois else ""
        cand.isbn = isbns[0] if isbns else ""
        cand.arxiv = arxivs[0] if arxivs else ""
        if stable_ids:
            cand.notes.append(f"jstor stable id: {stable_ids[0]}")
        return [cand]
