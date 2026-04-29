"""Regex constants, text utilities, identifier extraction, similarity helpers."""
from __future__ import annotations

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
from papis_import.core.filename_patterns import (
    ANNA_NAME_RE,
    ANNA_SPLIT_RE,
    HEX32_RE,
    LEADING_SERIES_RE,
)
from papis_import.core.matching import author_overlap, title_similarity
from papis_import.core.pipeline_helpers import build_tags, confidence_rank, should_import
from papis_import.core.process import command_exists, eprint, quote_shell, read_cmd
from papis_import.core.sanity import (
    MIN_SANITY_SCORE,
    is_book_signal,
    is_unreadable_text,
    sanity_score_for_match,
)
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
from papis_import.core.title_quality import (
    NOISE_LINE_PATTERNS,
    is_garbage_pdfinfo_title,
    is_garbage_title,
    is_grobid_output_suspicious,
    is_journal_abbrev_title,
    is_journal_header_title,
    is_suspicious_title,
)
