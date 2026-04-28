"""Regex constants, text utilities, identifier extraction, similarity helpers."""
from __future__ import annotations

import difflib
import re
from pathlib import Path

from papis_import.core.constants import (
    ARXIV_API_URL,
    CROSSREF_BASE,
    DEFAULT_CACHE_DIR,
    GOOGLE_BOOKS_URL,
    OLLAMA_CHAT_URL,
    OPENALEX_BASE,
    OPENLIBRARY_BOOKS_URL,
    OPENLIBRARY_SEARCH_URL,
    SEMANTIC_SCHOLAR_URL,
    USER_AGENT,
)
from papis_import.core.identifiers import (
    ARXIV_RE,
    DOI_RE,
    ELSEVIER_PII_FMT_RE,
    ELSEVIER_SCIDIR_RE,
    ISBN_CANDIDATE_RE,
    JSTOR_STABLE_RE,
    extract_identifiers,
    jstor_filename_doi,
    numeric_filename_dois,
    pii_to_doi,
    validate_isbn,
)
from papis_import.core.process import command_exists, eprint, quote_shell, read_cmd
from papis_import.core.text import (
    CONTROL_RE,
    MULTISPACE_RE,
    YEAR_RE,
    clean_filename_text,
    clean_text,
    decamelize,
    first_year,
    normalize_author_token,
    normalize_title,
    repair_ligature_splits,
    repair_title_ligatures,
    split_authors,
    strip_footnote_markers,
    strip_trailing_title_metadata,
)

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

HEX32_RE          = re.compile(r"^[0-9a-f]{32}$", re.I)
LEADING_SERIES_RE = re.compile(r"^[\[(].{0,160}?[\])]\s*")
ANNA_SPLIT_RE     = re.compile(r"\s+--\s+")
ANNA_NAME_RE      = re.compile(r"^Anna['']?s Archive(?:-\d+)?$", re.I)


def is_garbage_title(title: str) -> bool:
    """Return True if *title* is clearly not a real document title.

    Catches: author names ("R. Tyrrell"), symbols ("®"), journal headers
    ("2004, Vol. 32, No. 1A"), publisher noise, and very short/empty strings.
    """
    if not title or not title.strip():
        return True
    t = title.strip()
    # Almost no alphabetical content
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 3:
        return True
    # Looks like an author name: "R. Tyrrell" or "J. Smith"
    if re.fullmatch(r"[A-Z]\.?\s+[A-Z][a-z]+", t):
        return True
    # Looks like initials + surname: "R. T. Rockafellar"
    if re.fullmatch(r"(?:[A-Z]\.?\s+){1,3}[A-Z][a-z]+", t):
        return True
    # ALL CAPS single word that looks like a surname
    if re.fullmatch(r"[A-Z]{2,20}", t):
        return True
    # Journal/volume header: "2004, Vol. 32, No. 1A, 365-379"
    if re.match(r"^\d{4},?\s*(?:Vol|No|pp)", t):
        return True
    # Publisher symbol
    if t in ("®", "©", "™"):
        return True
    # Starts with sentence-like text rather than a title (blurb/abstract)
    _BLURB_STARTS = (
        "the book shows", "the area of", "this electronic",
        "this paper", "we empirically", "in this paper",
        "we prove", "we study", "we consider", "we present",
        "we introduce", "we propose", "we show", "we develop",
    )
    lower = t.lower()
    if any(lower.startswith(b) for b in _BLURB_STARTS):
        return True
    # Citation format: "Journal Name, Vol, Pages, Year"
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+,\s*\d+,\s*\d+", t):
        return True
    # Date: "August 16, 2024" or "16 August 2024"
    if re.fullmatch(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", t, re.I):
        return True
    # Mostly non-ASCII / control characters (garbled PDF)
    printable = sum(1 for c in t if c.isascii() and c.isprintable())
    if printable < len(t) * 0.5:
        return True
    # Journal header detected
    if is_journal_header_title(t):
        return True
    # Garbage PDF metadata
    if is_garbage_pdfinfo_title(t):
        return True
    return False


_GARBAGE_PDFINFO_TITLE_RE = re.compile(
    r"^(?:"
    r"/|https?://|"           # file path or URL
    r"\w+://"                 # any protocol
    r")",
    re.I,
)
_GARBAGE_PDFINFO_KEYWORDS = (
    "print job", "acdsee", "created by", "untitled", "microsoft word",
    "latex with hyperref", "powerpoint", "libreoffice",
)

def is_garbage_pdfinfo_title(title: str) -> bool:
    """Return True if a pdfinfo /Title field looks like software metadata, not a real title."""
    if not title:
        return True
    if _GARBAGE_PDFINFO_TITLE_RE.match(title):
        return True
    # Looks like a temp file path
    if re.search(r"\.(tmp|pdf|tex|dvi|docx?)\b", title, re.I):
        return True
    lower = title.lower()
    if any(kw in lower for kw in _GARBAGE_PDFINFO_KEYWORDS):
        return True
    return False


def is_journal_header_title(title: str) -> bool:
    """Return True when the candidate title is actually a journal/conference header,
    not the title of the work.  These appear as the first text in many journal PDFs.

    Examples:
      "Journal of Machine Learning Research 23 (2022) 1-109"
      "Reliability Engineering and System Safety 54 (1996) 95-111"
      "The Annals of Probability, Vol. 32 2004"
      "Journal: Of Multivariate" (title-cased from ALL-CAPS)
    """
    if not title:
        return False
    t = title.strip()
    # Matches: starts with a known journal keyword (possibly title-cased)
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
        # With a year, it's definitely a header; without, still likely
        if YEAR_RE.search(t) or len(t.split()) <= 5:
            return True
    # Pattern: "JOURNAL_ABBREVIATION Volume/Year page–page"
    if re.match(r"^[A-Z][A-Za-z\s]{2,40}\s+\d{1,3}\s+\(\d{4}\)", t):
        return True
    # TeX DVI filenames embedded in PDF metadata
    if re.fullmatch(r"[a-z0-9_.-]+\.dvi", t, re.I):
        return True
    return False


def is_journal_abbrev_title(title: str) -> bool:
    """Return True when *title* looks like a journal abbreviation + year, not
    an actual paper title.

    These arise from filenames like 'Sin CY and White H - J Econometrics - 1996.pdf'
    where the filename parser picks up 'J Econometrics - 1996' as the title.

    Examples:
      "J Econometrics - 1996", "JMVA - 1988", "J Approx Th - 1986"
    """
    if not title:
        return False
    t = title.strip()
    # Short text ending with or consisting mainly of a year
    if re.fullmatch(r"[A-Za-z\s.]{1,30}\s*[-–—]\s*\d{4}", t):
        return True
    # Very short journal abbreviation: "JMVA", "JRSSB"
    if re.fullmatch(r"[A-Z]{2,8}", t):
        return True
    # "J Something - Year" pattern
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
    # Journal watermarks / submission headers
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

# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ratio   = difflib.SequenceMatcher(None, na, nb).ratio()
    sa, sb  = set(na.split()), set(nb.split())
    jaccard = len(sa & sb) / max(1, len(sa | sb))
    return max(ratio, jaccard)


def author_overlap(a: list[str], b: list[str]) -> float:
    def surnames(names: list[str]) -> set[str]:
        out: set[str] = set()
        for x in names:
            norm = normalize_author_token(x)
            if not norm:
                continue
            surname = norm.split(",", 1)[0].strip() if "," in norm else (norm.split()[-1] if norm.split() else norm)
            if surname:
                out.add(surname)
        return out
    sa, sb = surnames(list(a)), surnames(list(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, min(len(sa), len(sb)))


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Post-match sanity check (item 1)
#
# After any external resolver returns metadata, verify that the resolved
# title or at least one author surname actually appears in the PDF text.
# This catches false-positive title-search matches like:
#   filename "ass.pdf" → Crossref search → exact match on an entry titled "ass"
# where nothing in the PDF supports that match.
# ---------------------------------------------------------------------------

# Common stopwords that shouldn't count when computing title word overlap.
# If we didn't exclude these, any 3-word title like "The Theory of X" would
# partially match any English text via "the", "of".
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
    """Normalize a filename stem for author-surname matching.

    Filenames often contain structured metadata that pdftotext can't see —
    author names, series info, year — separated by punctuation that we
    need to strip.  Splits on common delimiters and returns lowercased
    alphabetic tokens suitable for set membership checks.
    """
    if not s:
        return ""
    # Strip common Anna's Archive / libgen noise
    s = re.sub(r"anna.?s archive(?:-\d+)?", " ", s, flags=re.I)
    s = re.sub(r"libgen\.[A-Za-z0-9]+", " ", s, flags=re.I)
    # Strip 32-char hex hashes (Anna's Archive file IDs)
    s = re.sub(r"\b[0-9a-f]{32}\b", " ", s, flags=re.I)
    # Replace all non-alphanumeric with spaces, lowercase
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s).lower()
    return MULTISPACE_RE.sub(" ", s).strip()


def _author_surname(author: str) -> str:
    """Extract a lowercase surname token from a full-name string, or ''.

    Uses the last whitespace-separated token that's >=3 alphabetic chars,
    stripping diacritics and punctuation.  Returns '' if no such token.
    """
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
    """Return a score in [0.0, 1.0] measuring how well *meta* fits *pdf_text*.

    Scoring
    -------
    Base:  0.6 * title_match_ratio + 0.4 * author_surname_match
    Author match checks the PDF body text AND the filename (e.g.
    "Halbert White Asymptotic Theory…pdf" lets us corroborate a match
    whose author is "Halbert White" even when pdftotext gave us garbage).

    Unreadable-text policy
    ----------------------
    When pdf_text is empty or unreadable (garbled bytes / scanned without
    OCR / extremely low function-word ratio):

      - Identifier-based lookups (DOI/ISBN/arXiv) are trusted at 0.6.
        The identifier came either from a strict regex over the text
        stream (which wouldn't match on random bytes) or from a reliable
        filename pattern like an Elsevier PII, so the lookup result is
        authoritative even without body-text cross-check.

      - Title-search lookups (crossref_search, openalex_search,
        semanticscholar_search, openlibrary_search, google_books_search)
        normally score 0 in this regime — the API's own similarity
        score can be wrong (the ass.pdf false-positive case).  BUT if
        the resolved author's surname appears in the filename we grant
        0.6: a filename like "van_de_Vel_Convex_Structures.pdf" matching
        a Crossref result "van de Vel / Theory of Convex Structures" is
        strong corroboration even if the PDF body text is unreadable.

    Readable-text policy
    --------------------
    With readable text, the base score applies.  The author component
    checks both the PDF text AND the filename — this catches cases
    where text extraction succeeded but only captured the TOC/contents
    rather than the title page.

    Parameters
    ----------
    meta_title, meta_authors, meta_source : as usual
    pdf_text : str
        Extracted PDF text (usually the first 1-2 pages).
    filename : str, optional
        Original filename (basename, with or without extension).  If
        provided, enables filename-based author corroboration.  Safe
        to pass either "/full/path/to/file.pdf" or just "file.pdf" —
        directory parts and the extension are ignored.
    """
    identifier_sources = {"crossref_doi", "openlibrary_isbn", "arxiv_id"}
    title_search_sources = {
        "crossref_search", "openalex_search", "semanticscholar_search",
        "openlibrary_search", "google_books_search",
    }

    # Normalize filename into a word set for surname lookup.  Strip any
    # directory path and the .pdf extension first.
    filename_basename = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if filename_basename.lower().endswith(".pdf"):
        filename_basename = filename_basename[:-4]
    filename_norm = _normalize_filename_for_sanity(filename_basename)
    filename_words = set(filename_norm.split()) if filename_norm else set()

    def _author_in_filename() -> bool:
        """True if any resolved author's surname appears in the filename."""
        if not filename_words:
            return False
        for author in meta_authors or []:
            surname = _author_surname(author)
            if surname and surname in filename_words:
                return True
        return False

    # Completely empty or near-empty text: identifier lookups get 0.6;
    # title-search lookups get 0.6 if filename corroborates an author,
    # else 0.
    if not pdf_text or len(pdf_text.strip()) < 100:
        if meta_source in identifier_sources:
            return 0.6
        if meta_source in title_search_sources and _author_in_filename():
            return 0.6
        return 0.0

    # Non-empty but garbled: same policy.  Title-search lookups with
    # filename author corroboration escape the 0-score trap.
    if is_unreadable_text(pdf_text):
        if meta_source in identifier_sources:
            return 0.6
        if meta_source in title_search_sources and _author_in_filename():
            return 0.6
        # No corroboration — fall through to the full check, which will
        # score low because text_words is small.  We don't early-return
        # 0 here because sometimes "unreadable" text has enough real
        # fragments to partially match the title.

    text_norm = _normalize_for_sanity(pdf_text[:8000])
    text_words = set(text_norm.split())

    # ---- Title component ----
    title_match = 0.0
    if meta_title:
        title_norm = _normalize_for_sanity(meta_title)
        if not title_norm:
            title_match = 0.0
        elif f" {title_norm} " in f" {text_norm} ":
            # Full normalized title appears as a contiguous phrase — strongest signal.
            title_match = 1.0
        else:
            significant = [w for w in title_norm.split()
                           if w not in _TITLE_STOPWORDS and len(w) >= 3]
            if significant:
                overlap = sum(1 for w in significant if w in text_words) / len(significant)
                title_match = overlap
            else:
                # Title was all stopwords (e.g. normalized to "the of") — can't judge.
                # Give a small credit so DOI lookups with weird titles aren't killed.
                title_match = 0.3

    # ---- Author component ----
    # Check both PDF text AND filename. Filename check catches the case
    # where text extraction pulled the TOC but not the title page, so the
    # author name is absent from the extracted text despite being in the
    # filename (e.g. "Halbert White Asymptotic Theory…pdf").
    author_match = 0.0
    for author in meta_authors or []:
        surname = _author_surname(author)
        if not surname:
            continue
        if surname in text_words or surname in filename_words:
            author_match = 1.0
            break

    return 0.6 * title_match + 0.4 * author_match


# Threshold below which a verified external match is considered a false
# positive and downgraded to unverified.  Above this, keep confidence.
MIN_SANITY_SCORE = 0.4


# ---------------------------------------------------------------------------
# Unreadable-text detection (item 5)
#
# Triggers OCR retry. Uses function-word frequency rather than printable-byte
# ratio so that OCR'd garbage made of real Latin letters but no real words
# is also detected, not just wingdings.
# ---------------------------------------------------------------------------

_COMMON_ENGLISH_FUNCTION_WORDS = frozenset({
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "on", "with",
    "as", "by", "this", "be", "are", "or", "from", "an", "at", "it", "which",
    "we", "not", "have", "has", "was", "were", "can", "will", "if", "then",
    "also", "these", "such", "our", "all", "any", "one", "two", "more",
    # Math-book filler words that appear often in the running prose between
    # equations.  Including these prevents false-positive "unreadable" flags
    # on heavy-math texts like Asymptotic Statistics or Convex Analysis.
    "let", "proof", "theorem", "lemma", "corollary", "proposition",
    "definition", "example", "remark", "where", "thus", "hence",
    "therefore", "exists", "set", "function", "space",
})


def is_unreadable_text(
    text: str,
    min_tokens: int = 30,
    min_ratio: float = 0.04,
) -> bool:
    """Return True if *text* has too few real English words to be usable.

    The original threshold of 0.08 was too aggressive on math-heavy prose,
    which runs about 5–10% function words (versus 20–30% in humanities
    text) because equations and symbolic notation crowd out the English.
    Dropped to 0.04 — wingdings still score 0, OCR-scramble text scores
    < 0.02, but genuinely-math-heavy books are no longer false-positives.

    Also requires the text to contain a minimum number of real-looking
    alphabetic words: scans where pdftotext extracts only digits / page
    numbers still fail the `min_tokens` check and go to OCR as intended.
    """
    if not text or len(text.strip()) < 200:
        return True
    snippet = text[:3000].lower()
    tokens = re.findall(r"\b[a-z]+\b", snippet)
    if len(tokens) < min_tokens:
        return True
    hits = sum(1 for t in tokens if t in _COMMON_ENGLISH_FUNCTION_WORDS)
    return (hits / len(tokens)) < min_ratio


# ---------------------------------------------------------------------------
# Soft is_book detector (item 4)
#
# Used to demote GROBID's priority. GROBID is trained on journal-article
# headers and produces poor results on books (picking up project numbers,
# series metadata, or editor names instead of the actual title).
# ---------------------------------------------------------------------------

_BOOK_PUBLISHER_HINTS = (
    "springer", "cambridge", "oxford", "elsevier", "wiley",
    "crc press", "world scientific", "north-holland", "north holland",
    "academic press", "mcgraw", "prentice hall", "chapman",
    "princeton", "cup", "oup", "birkh",     # birkhäuser
    "de gruyter", "marcel dekker", "van nostrand",
)


def is_book_signal(filename: str, text: str) -> bool:
    """Soft heuristic: does this PDF look more like a book than a paper?

    Returns True when multiple book-specific cues are present. Used only
    as a demotion signal for GROBID — never as a hard rejection.
    """
    fname_lower = filename.lower()

    # Filename hints (strong)
    if any(h in fname_lower for h in _BOOK_PUBLISHER_HINTS):
        return True
    if "libgen" in fname_lower or "anna" in fname_lower:
        return True

    if not text:
        return False
    head = text[:20000].lower()

    # Library of Congress / CIP data blocks appear only in books
    if "cataloging-in-publication" in head or "library of congress" in head:
        return True

    # ISBN without a page-1 DOI is a strong book signal
    has_isbn = bool(re.search(r"\bisbn\b[-\s]?1?[03]?[:\s]*\d", text[:10000], re.I))
    has_page1_doi = bool(re.search(r"10\.\d{4,9}/", text[:3000]))
    if has_isbn and not has_page1_doi:
        return True

    return False


# ---------------------------------------------------------------------------
# Suspicious-title filter
#
# Originally a GROBID-output sanity filter; promoted to a general check after
# the Maillard JMLR 2021 case where the text-LLM extractor produced an
# affiliation string ("Université Paris-Saclay, CNRS, Inria, Laboratoire…")
# that beat the correct title ("Aggregated Hold-Out") in _synthesize().
# Catches the same pathologies regardless of which extractor produced them:
#   - "Université Paris-Saclay, CNRS, Inria, Laboratoire..." (affiliation)
#   - "CONVEX FUNCTIONS ... SPIN Springer's internal project num" (cut off)
#   - "The area of stochastic programming was created..." (abstract first line)
# ---------------------------------------------------------------------------

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
    """Return True if the title looks like an affiliation, abstract fragment,
    or publisher metadata rather than a real title.

    More aggressive than is_garbage_title. Originally a GROBID-only filter;
    now also applied inside _synthesize() so any extractor (text_header,
    llm:*, vision_llm:*, …) producing the same pathologies is rejected.
    """
    if not title:
        return False
    t = title.strip()
    if len(t) > 180:
        return True
    # Email or URL in the title
    if "@" in t or "http://" in t or "https://" in t:
        return True
    # Springer-specific metadata markers ("SPIN 10928341" or
    # "SPIN Springer's internal project number…") — never appear in real titles.
    if re.search(r"\bSPIN\s+(?:\d|Springer)", t):
        return True
    if "internal project num" in t.lower():
        return True
    # Affiliation markers anywhere in the title
    lower = t.lower()
    if any(m in lower for m in _AFFILIATION_MARKERS):
        return True
    # Starts with a sentence pattern (abstract opener)
    if any(lower.startswith(s) for s in _SUSPICIOUS_SENTENCE_STARTS):
        return True
    # Contains a postal-code-like pattern typical of affiliation strings
    # ("91405, Orsay, France")
    if re.search(r"\b\d{4,5},\s*[A-Z][a-zéöäü]+,\s*[A-Z][a-zA-Z]+", t):
        return True
    return False


# Back-compat alias — prior name when this filter was GROBID-specific.
is_grobid_output_suspicious = is_suspicious_title
