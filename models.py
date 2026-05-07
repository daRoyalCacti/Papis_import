"""Data classes shared across all modules."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path


@dataclass
class Metadata:
    """Final verified (or best-guess) metadata for one PDF."""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: str = ""
    doi: str = ""
    isbn: str = ""
    arxiv: str = ""
    publisher: str = ""
    source: str = ""
    confidence: str = "low"
    verified: bool = False
    # Sanity-check: does this metadata actually correspond to the PDF?
    # True when the resolved title or at least one author surname appears in
    # the PDF text. False is informative — it flags likely false positives
    # from title-search even when the API score was high.
    sanity_passed: bool = False
    sanity_score: float = 0.0
    # OCR hint: the PDF text extraction looked garbled (non-English noise).
    # Used by the CLI to trigger ocrmypdf and re-resolve once.
    needs_ocr: bool = False
    # auto_safe = verified AND sanity_passed AND confidence == "high".
    # Only auto_safe rows go to papis_import_auto.tsv; everything else
    # goes to papis_import_review.tsv for human review.
    auto_safe: bool = False
    # soft_auto: NOT auto_safe but rescued via local corroboration —
    # multiple independent strong extractors agree on title + authors.
    # Soft-auto rows go to BOTH the auto TSV and a soft-auto TSV for
    # later human review of the rescue rule.
    soft_auto: bool = False
    soft_auto_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def merge_missing(self, other: "Metadata") -> None:
        """Fill in blank fields from *other* without overwriting populated ones."""
        for key in ("title", "year", "doi", "isbn", "arxiv", "publisher"):
            if not getattr(self, key) and getattr(other, key):
                setattr(self, key, getattr(other, key))
        if not self.authors and other.authors:
            self.authors = other.authors[:]
        if other.notes:
            self.notes.extend(other.notes)


@dataclass
class Candidate:
    """A raw metadata guess from one local source (filename, PDF header, …)."""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: str = ""
    doi: str = ""
    isbn: str = ""
    arxiv: str = ""
    source: str = ""
    priority: int = 100   # lower == prefer earlier in verification
    notes: list[str] = field(default_factory=list)




@dataclass
class IdentifierLookupTiming:
    kind: str                 # "doi" | "isbn" | "arxiv"
    value: str
    resolver: str             # "crossref_by_doi", "openlibrary_by_isbn", "arxiv_by_id"
    elapsed_s: float = 0.0
    matched: bool = False
    source: str = ""          # resulting Metadata.source if matched
    sanity_score: float = 0.0
    error: str = ""


@dataclass
class TitleSearchTiming:
    source: str
    elapsed_s: float = 0.0
    candidates_available: int = 0
    candidates_tried: int = 0
    matches_returned: int = 0
    best_score: float = -1.0
    best_source: str = ""
    best_title: str = ""
    best_verified: bool = False
    best_sanity_passed: bool = False
    best_sanity_score: float = 0.0
    errors: int = 0
    error_messages: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    stopped_early: bool = False
    query_traces: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TimingBreakdown:
    file_wall_s: float = 0.0
    resolve_total_s: float = 0.0

    best_local_s: float = 0.0
    header_candidate_s: float=0.0

    text_extract_s: float = 0.0
    embedded_metadata_s: float = 0.0
    grobid_s: float = 0.0
    text_llm_s: float = 0.0
    vision_llm_s: float = 0.0
    vision_pacing_s: float = 0.0   # subset of vision_llm_s spent in proactive quota sleep
    identifier_lookups_s: float = 0.0
    title_search_s: float = 0.0

    ocrmypdf_s: float = 0.0
    ocr_reresolve_s: float = 0.0
    ocr_retry_s: float = 0.0
    ocr_status: str = ""

    identifier_lookups: list[IdentifierLookupTiming] = field(default_factory=list)
    title_searches: list[TitleSearchTiming] = field(default_factory=list)



@dataclass
class Record:
    """One processed PDF with its outcome."""
    path: Path
    tags: list[str]
    result: Metadata
    suggested_command: str
    imported: bool = False
    error: str = ""
    debug: dict[str, Any] = field(default_factory=dict)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
