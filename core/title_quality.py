"""Heuristics for filtering title candidates and noisy header lines."""
from __future__ import annotations

import re

from papis_import.core.text import YEAR_RE


def is_garbage_title(title: str) -> bool:
    """Return True if *title* is clearly not a real document title."""
    if not title or not title.strip():
        return True
    t = title.strip()
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 3:
        return True
    if re.fullmatch(r"[A-Z]\.?\s+[A-Z][a-z]+", t):
        return True
    if re.fullmatch(r"(?:[A-Z]\.?\s+){1,3}[A-Z][a-z]+", t):
        return True
    if re.fullmatch(r"[A-Z]{2,20}", t):
        return True
    if re.match(r"^\d{4},?\s*(?:Vol|No|pp)", t):
        return True
    if t in ("®", "©", "™"):
        return True
    blurb_starts = (
        "the book shows", "the area of", "this electronic",
        "this paper", "we empirically", "in this paper",
        "we prove", "we study", "we consider", "we present",
        "we introduce", "we propose", "we show", "we develop",
    )
    lower = t.lower()
    if any(lower.startswith(b) for b in blurb_starts):
        return True
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+,\s*\d+,\s*\d+", t):
        return True
    if re.fullmatch(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", t, re.I):
        return True
    printable = sum(1 for c in t if c.isascii() and c.isprintable())
    if printable < len(t) * 0.5:
        return True
    if is_journal_header_title(t):
        return True
    if is_garbage_pdfinfo_title(t):
        return True
    return False


_GARBAGE_PDFINFO_TITLE_RE = re.compile(
    r"^(?:"
    r"/|https?://|"
    r"\w+://"
    r")",
    re.I,
)
_GARBAGE_PDFINFO_KEYWORDS = (
    "print job", "acdsee", "created by", "untitled", "microsoft word",
    "latex with hyperref", "powerpoint", "libreoffice",
)


def is_garbage_pdfinfo_title(title: str) -> bool:
    """Return True if a pdfinfo /Title field looks like software metadata."""
    if not title:
        return True
    if _GARBAGE_PDFINFO_TITLE_RE.match(title):
        return True
    if re.search(r"\.(tmp|pdf|tex|dvi|docx?)\b", title, re.I):
        return True
    lower = title.lower()
    if any(kw in lower for kw in _GARBAGE_PDFINFO_KEYWORDS):
        return True
    return False


def is_journal_header_title(title: str) -> bool:
    """Return True when a candidate title is actually a journal/conference header."""
    if not title:
        return False
    t = title.strip()
    journal_prefixes = (
        "journal of", "journal: of", "annals of", "annals: of",
        "bulletin of", "bulletin: of", "transactions on", "transactions: on",
        "proceedings of", "proceedings: of", "conference on", "workshop on",
        "reliability engineering", "structural safety",
        "illinois journal", "electronic journal",
        "statistical science", "machine learning,",
        "advances in", "international journal",
    )
    lower = t.lower()
    if any(lower.startswith(p) for p in journal_prefixes):
        if YEAR_RE.search(t) or len(t.split()) <= 5:
            return True
    if re.match(r"^[A-Z][A-Za-z\s]{2,40}\s+\d{1,3}\s+\(\d{4}\)", t):
        return True
    if re.fullmatch(r"[a-z0-9_.-]+\.dvi", t, re.I):
        return True
    return False


def is_journal_abbrev_title(title: str) -> bool:
    """Return True when *title* looks like a journal abbreviation + year."""
    if not title:
        return False
    t = title.strip()
    if re.fullmatch(r"[A-Za-z\s.]{1,30}\s*[-–—]\s*\d{4}", t):
        return True
    if re.fullmatch(r"[A-Z]{2,8}", t):
        return True
    if re.match(r"^J\s+[A-Z][a-z]", t) and len(t) < 40:
        return True
    return False


NOISE_LINE_PATTERNS = [
    re.compile(r"^downloaded from", re.I),
    re.compile(r"^generated on", re.I),
    re.compile(r"^jstor", re.I),
    re.compile(r"^terms and conditions", re.I),
    re.compile(r"^https?://", re.I),
    re.compile(r"^doi[:\s]", re.I),
    re.compile(r"^author\(s\):", re.I),
    re.compile(r"^title[:\s]", re.I),
    re.compile(r"^source[:\s]", re.I),
    re.compile(r"^\d+$"),
    re.compile(r"^journal of machine learning research\b", re.I),
    re.compile(r"^(?:neural information processing|advances in neural)", re.I),
    re.compile(r"^published as a conference paper\b", re.I),
    re.compile(r"^(?:submitted|revised|accepted|published)\s+\d+/\d+", re.I),
    re.compile(r"^under review\b", re.I),
    re.compile(r"^(?:c|©)\s*\d{4}\s+(?:kluwer|springer|elsevier|wiley|ieee)", re.I),
    re.compile(r"^copyright\s+\d{4}\b", re.I),
    re.compile(r"^(?:vol(?:ume)?\.?\s+\d+|no\.?\s+\d+).+\d{4}", re.I),
    re.compile(r"^the annals of (?:statistics|probability|mathematics)\b", re.I),
    re.compile(r"^(?:illinois|annals|bulletin|proceedings)\s+journal\b", re.I),
    re.compile(r"^institute of mathematical statistics\b", re.I),
]

_AFFILIATION_MARKERS = (
    "university", "université", "universidad", "universität", "universitat",
    "cnrs", "inria",
    "institute", "institut",
    "department", "département", "departamento",
    "faculty", "faculté",
    "school of", "college of", "centre for", "center for",
    "laboratory", "laboratoire", "laboratorio",
)

_SUSPICIOUS_SENTENCE_STARTS = (
    "the area of", "the book", "this paper", "this book", "this chapter",
    "this electronic", "this article", "this volume",
    "in this paper", "in this book", "in this chapter",
    "we study", "we prove", "we consider", "we present", "we introduce",
    "we propose", "we show", "we develop", "we empirically", "we examine",
    "abstract ", "abstract.", "introduction ", "introduction.", "keywords",
)


def is_suspicious_title(title: str) -> bool:
    """Return True if the title looks like affiliation, abstract, or metadata."""
    if not title:
        return False
    t = title.strip()
    if len(t) > 180:
        return True
    if "@" in t or "http://" in t or "https://" in t:
        return True
    if re.search(r"\bSPIN\s+(?:\d|Springer)", t):
        return True
    if "internal project num" in t.lower():
        return True
    lower = t.lower()
    if any(m in lower for m in _AFFILIATION_MARKERS):
        return True
    if any(lower.startswith(s) for s in _SUSPICIOUS_SENTENCE_STARTS):
        return True
    if re.search(r"\b\d{4,5},\s*[A-Z][a-zéöäü]+,\s*[A-Z][a-zA-Z]+", t):
        return True
    return False


is_grobid_output_suspicious = is_suspicious_title

