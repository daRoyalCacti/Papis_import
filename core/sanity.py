"""Sanity scoring and document signal heuristics."""
from __future__ import annotations

import re

from papis_import.core.text import MULTISPACE_RE, clean_text

_TITLE_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "and", "or", "for", "to", "with",
    "by", "at", "is", "are", "as", "from", "this", "that", "these", "those",
    "its", "their", "our", "new", "second", "third", "fourth", "fifth",
    "edition", "volume", "vol", "part", "chapter", "section", "ed",
})


def _normalize_for_sanity(s: str) -> str:
    """Like normalize_title but preserves more content for substring scans."""
    s = clean_text(s).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def _normalize_filename_for_sanity(s: str) -> str:
    """Normalize a filename stem for author-surname matching."""
    if not s:
        return ""
    s = re.sub(r"anna.?s archive(?:-\d+)?", " ", s, flags=re.I)
    s = re.sub(r"libgen\.[A-Za-z0-9]+", " ", s, flags=re.I)
    s = re.sub(r"\b[0-9a-f]{32}\b", " ", s, flags=re.I)
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s).lower()
    return MULTISPACE_RE.sub(" ", s).strip()


def _author_surname(author: str) -> str:
    """Extract a lowercase surname token from a full-name string, or ''."""
    tokens = [t for t in re.split(r"[\s,]+", author) if t]
    if not tokens:
        return ""
    for tok in reversed(tokens):
        tok_clean = re.sub(r"[^a-zA-Z]", "", tok).lower()
        if len(tok_clean) >= 3 and not re.fullmatch(r"[a-z]", tok_clean):
            return tok_clean
    return ""


def sanity_score_for_match(
    meta_title: str,
    meta_authors: list[str],
    meta_source: str,
    pdf_text: str,
    filename: str = "",
) -> float:
    """Return a score in [0.0, 1.0] measuring how well *meta* fits *pdf_text*."""
    identifier_sources = {"crossref_doi", "openlibrary_isbn", "arxiv_id"}
    title_search_sources = {
        "crossref_search", "openalex_search", "semanticscholar_search",
        "openlibrary_search", "google_books_search",
    }

    filename_basename = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if filename_basename.lower().endswith(".pdf"):
        filename_basename = filename_basename[:-4]
    filename_norm = _normalize_filename_for_sanity(filename_basename)
    filename_words = set(filename_norm.split()) if filename_norm else set()

    def _author_in_filename() -> bool:
        if not filename_words:
            return False
        for author in meta_authors or []:
            surname = _author_surname(author)
            if surname and surname in filename_words:
                return True
        return False

    if not pdf_text or len(pdf_text.strip()) < 100:
        if meta_source in identifier_sources:
            return 0.6
        if meta_source in title_search_sources and _author_in_filename():
            return 0.6
        return 0.0

    if is_unreadable_text(pdf_text):
        if meta_source in identifier_sources:
            return 0.6
        if meta_source in title_search_sources and _author_in_filename():
            return 0.6

    text_norm = _normalize_for_sanity(pdf_text[:8000])
    text_words = set(text_norm.split())

    title_match = 0.0
    if meta_title:
        title_norm = _normalize_for_sanity(meta_title)
        if not title_norm:
            title_match = 0.0
        elif f" {title_norm} " in f" {text_norm} ":
            title_match = 1.0
        else:
            significant = [w for w in title_norm.split()
                           if w not in _TITLE_STOPWORDS and len(w) >= 3]
            if significant:
                title_match = sum(1 for w in significant if w in text_words) / len(significant)
            else:
                title_match = 0.3

    author_match = 0.0
    for author in meta_authors or []:
        surname = _author_surname(author)
        if not surname:
            continue
        if surname in text_words or surname in filename_words:
            author_match = 1.0
            break

    return 0.6 * title_match + 0.4 * author_match


MIN_SANITY_SCORE = 0.4

_COMMON_ENGLISH_FUNCTION_WORDS = frozenset({
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "on", "with",
    "as", "by", "this", "be", "are", "or", "from", "an", "at", "it", "which",
    "we", "not", "have", "has", "was", "were", "can", "will", "if", "then",
    "also", "these", "such", "our", "all", "any", "one", "two", "more",
    "let", "proof", "theorem", "lemma", "corollary", "proposition",
    "definition", "example", "remark", "where", "thus", "hence",
    "therefore", "exists", "set", "function", "space",
})


def is_unreadable_text(
    text: str,
    min_tokens: int = 30,
    min_ratio: float = 0.04,
) -> bool:
    """Return True if *text* has too few real English words to be usable."""
    if not text or len(text.strip()) < 200:
        return True
    snippet = text[:3000].lower()
    tokens = re.findall(r"\b[a-z]+\b", snippet)
    if len(tokens) < min_tokens:
        return True
    hits = sum(1 for t in tokens if t in _COMMON_ENGLISH_FUNCTION_WORDS)
    return (hits / len(tokens)) < min_ratio


_BOOK_PUBLISHER_HINTS = (
    "springer", "cambridge", "oxford", "elsevier", "wiley",
    "crc press", "world scientific", "north-holland", "north holland",
    "academic press", "mcgraw", "prentice hall", "chapman",
    "princeton", "cup", "oup", "birkh",
    "de gruyter", "marcel dekker", "van nostrand",
)


def is_book_signal(filename: str, text: str) -> bool:
    """Soft heuristic: does this PDF look more like a book than a paper?"""
    fname_lower = filename.lower()
    if any(h in fname_lower for h in _BOOK_PUBLISHER_HINTS):
        return True
    if "libgen" in fname_lower or "anna" in fname_lower:
        return True

    if not text:
        return False
    head = text[:20000].lower()

    if "cataloging-in-publication" in head or "library of congress" in head:
        return True

    has_isbn = bool(re.search(r"\bisbn\b[-\s]?1?[03]?[:\s]*\d", text[:10000], re.I))
    has_page1_doi = bool(re.search(r"10\.\d{4,9}/", text[:3000]))
    if has_isbn and not has_page1_doi:
        return True

    return False

