from __future__ import annotations

from papis_import.models import Candidate, Metadata
from papis_import.pipeline_parts.candidates import CandidateSelector
from papis_import.pipeline_parts.title_search import is_too_generic_to_search
from papis_import.utils import (
    MIN_SANITY_SCORE,
    author_overlap,
    is_garbage_title,
    is_journal_abbrev_title,
    is_suspicious_title,
    normalize_title,
    title_similarity,
)


EXTERNAL_SEARCH_SOURCES = {
    "crossref_search",
    "openalex_search",
    "semanticscholar_search",
    "openlibrary_search",
    "google_books_search",
}

SOFT_AUTO_STRONG_SOURCES = {"grobid", "pdfinfo", "xmp", "text_header"}
SOFT_AUTO_STRONG_PREFIXES = ("llm:", "vision_llm:")
SOFT_AUTO_TITLE_SIM = 0.7
SOFT_AUTO_MIN_TITLE_LEN = 5


class ResolutionFinalizer:
    def __init__(self, selector: CandidateSelector) -> None:
        self.selector = selector

    def finalize_winner(
        self,
        *,
        best_ident: Metadata | None,
        best_ident_score: float,
        best_ident_note: str,
        best_search: Metadata | None,
        best_search_score: float,
        candidates: list[Candidate],
        sanity_text: str,
        needs_ocr_flag: bool,
        debug: dict[str, str],
    ) -> Metadata | None:
        winners: list[tuple[Metadata, float, str]] = []
        if best_ident is not None:
            winners.append((best_ident, best_ident_score, best_ident_note))
        if best_search is not None:
            winners.append((best_search, best_search_score, "title-search match"))
        if not winners:
            return None

        winners.sort(key=lambda x: -x[1])
        winner, _score, note = winners[0]
        winner.merge_missing(self.selector.local_fallback(candidates))
        winner.notes.append(note)
        _rescue_locally_corroborated_match(winner, candidates)
        _downgrade_if_sanity_failed(winner)
        if not winner.sanity_passed:
            winner = _prefer_local_when_external_conflicts(winner, candidates, self.selector)
        winner.needs_ocr = needs_ocr_flag
        winner.auto_safe = (
            winner.verified
            and winner.sanity_passed
            and winner.confidence == "high"
        )
        winner.soft_auto, winner.soft_auto_reasons = _evaluate_soft_auto(winner, candidates)
        debug["final_source"] = winner.source
        return winner

    def finalize_identifier_winner(
        self,
        meta: Metadata,
        candidates: list[Candidate],
        note: str,
        needs_ocr_flag: bool,
        debug: dict[str, str],
    ) -> Metadata:
        meta.merge_missing(self.selector.local_fallback(candidates))
        if note:
            meta.notes.append(note)
        meta.notes.append(f"sanity_score={meta.sanity_score:.3f} (passed)")
        meta.needs_ocr = needs_ocr_flag
        meta.auto_safe = (
            meta.verified
            and meta.sanity_passed
            and meta.confidence == "high"
        )
        meta.soft_auto, meta.soft_auto_reasons = _evaluate_soft_auto(meta, candidates)
        debug["final_source"] = meta.source
        return meta

    def local_fallback(
        self,
        *,
        candidates: list[Candidate],
        stable_ids: list[str],
        needs_ocr_flag: bool,
        debug: dict[str, str],
    ) -> Metadata:
        fallback = self.selector.local_fallback(candidates)
        if stable_ids:
            fallback.notes.append(f"JSTOR stable ID present (not an arXiv ID): {stable_ids[0]}")
        if needs_ocr_flag:
            fallback.notes.append("PDF text appears unreadable — OCR recommended")
        fallback.needs_ocr = needs_ocr_flag
        fallback.sanity_passed = False
        fallback.sanity_score = 0.0
        fallback.auto_safe = False
        fallback.soft_auto, fallback.soft_auto_reasons = _evaluate_soft_auto(fallback, candidates)
        debug["final_source"] = fallback.source
        return fallback


def _downgrade_if_sanity_failed(meta: Metadata) -> Metadata:
    if not meta.sanity_passed:
        meta.verified = False
        if meta.confidence == "high":
            meta.confidence = "medium"
        meta.notes.append(f"sanity_score={meta.sanity_score:.3f} (below threshold {MIN_SANITY_SCORE})")
    else:
        meta.notes.append(f"sanity_score={meta.sanity_score:.3f} (passed)")
    return meta


def _is_strong_local_corroborator(source: str) -> bool:
    if not source:
        return False
    if source in {"synthesized", "filename_title_only", "text_header", "filename_author_only"}:
        return False
    return (
        source in {
            "grobid",
            "pdfinfo",
            "pdf_metadata",
            "xmp",
            "filename_author_title",
            "filename_structured",
            "filename_series",
        }
        or source.startswith("vision_llm:")
        or source.startswith("llm:")
    )


def _local_corroboration_note(meta: Metadata, candidates: list[Candidate]) -> str:
    title_words = normalize_title(meta.title).split()
    strong_title_sources: list[str] = []
    strong_full_sources: list[str] = []

    for cand in candidates:
        if not _is_strong_local_corroborator(cand.source):
            continue
        if not cand.title:
            continue
        ts = title_similarity(meta.title, cand.title)
        if ts < 0.97:
            continue
        strong_title_sources.append(cand.source)

        ao = author_overlap(meta.authors, cand.authors) if meta.authors and cand.authors else 0.0
        ym = bool(meta.year and cand.year and meta.year == cand.year)
        if ao >= 0.5 or ym:
            strong_full_sources.append(cand.source)

    if strong_full_sources:
        seen = list(dict.fromkeys(strong_full_sources))
        return "locally corroborated despite unreadable text via " + ", ".join(seen)

    seen_title = list(dict.fromkeys(strong_title_sources))
    if len(seen_title) >= 2 and len(title_words) >= 4:
        return "title corroborated by multiple local extractors despite unreadable text via " + ", ".join(seen_title)

    return ""


def _is_strong_soft_source(source: str) -> bool:
    return source in SOFT_AUTO_STRONG_SOURCES or source.startswith(SOFT_AUTO_STRONG_PREFIXES)


# Sources considered reliable enough to corroborate an identifier-based result.
# Deliberately wider than SOFT_AUTO_STRONG_SOURCES: structured filename sources
# (filename_author_title, filename_structured, filename_series) carry both title
# AND author and are the primary signal for well-named books and papers, but they
# are excluded from the soft-auto path because they don't add independent evidence
# there. For identifier corroboration they ARE independent (the identifier API is
# a separate lookup), so they count. filename_title_only and filename_author_only
# are excluded because they each carry only one field — not enough for a two-field check.
_IDENTIFIER_CORROBORATION_SOURCES = frozenset({
    "text_header",
    "grobid",
    "pdfinfo",
    "pdf_metadata",
    "xmp",
    "filename_author_title",
    "filename_structured",
    "filename_series",
})


def _is_identifier_corroborator(source: str) -> bool:
    return (
        source in _IDENTIFIER_CORROBORATION_SOURCES
        or source.startswith(("llm:", "vision_llm:"))
    )


def _identifier_corroborating_source(meta: Metadata, candidates: list[Candidate]) -> str:
    """Return the best local source that corroborates meta on title+author, or ''.

    Used by safe-mode accept (--accept-mode safe) to require independent local
    confirmation before trusting an identifier API result.  The check requires:
      - title_similarity(local_title, resolved_title) ≥ SOFT_AUTO_TITLE_SIM (0.7)
      - author_overlap(local_authors, resolved_authors) > 0
    from at least one source in _IDENTIFIER_CORROBORATION_SOURCES or any LLM source.
    """
    title = (meta.title or "").strip()
    if not title:
        return ""
    best_source = ""
    best_sim = 0.0
    for cand in candidates:
        if not _is_identifier_corroborator(cand.source):
            continue
        if not cand.title:
            continue
        sim = title_similarity(cand.title, title)
        if sim < SOFT_AUTO_TITLE_SIM:
            continue
        if not meta.authors or not cand.authors:
            continue
        if author_overlap(meta.authors, cand.authors) <= 0.0:
            continue
        if sim > best_sim:
            best_sim = sim
            best_source = cand.source
    return best_source


def _evaluate_soft_auto(meta: Metadata, candidates: list[Candidate]) -> tuple[bool, list[str]]:
    if meta.auto_safe:
        return False, []
    title = (meta.title or "").strip()
    if len(title) < SOFT_AUTO_MIN_TITLE_LEN:
        return False, []
    if is_garbage_title(title) or is_journal_abbrev_title(title) or is_suspicious_title(title):
        return False, []

    title_match_sources: list[str] = []
    author_match_sources: set[str] = set()
    for cand in candidates:
        if not _is_strong_soft_source(cand.source):
            continue
        if not cand.title:
            continue
        if title_similarity(cand.title, title) < SOFT_AUTO_TITLE_SIM:
            continue
        if cand.source in title_match_sources:
            continue
        title_match_sources.append(cand.source)
        if meta.authors and cand.authors and author_overlap(cand.authors, meta.authors) > 0.0:
            author_match_sources.add(cand.source)

    needed = 3 if meta.needs_ocr else 2
    if len(title_match_sources) < needed:
        return False, []
    if not author_match_sources:
        return False, []

    reason = (
        f"title corroborated by {len(title_match_sources)} strong sources "
        f"({', '.join(title_match_sources)}); authors corroborated by "
        f"{', '.join(sorted(author_match_sources))}"
    )
    if meta.needs_ocr:
        reason = "[needs-ocr threshold] " + reason
    return True, [reason]


def _rescue_locally_corroborated_match(meta: Metadata, candidates: list[Candidate]) -> Metadata:
    if meta.sanity_passed or not meta.verified:
        return meta
    if meta.source not in EXTERNAL_SEARCH_SOURCES:
        return meta

    note = _local_corroboration_note(meta, candidates)
    if not note:
        return meta

    meta.sanity_passed = True
    meta.sanity_score = max(meta.sanity_score, 0.6)
    meta.notes.append(note)
    return meta


def _prefer_local_when_external_conflicts(
    meta: Metadata,
    candidates: list[Candidate],
    selector: CandidateSelector,
) -> Metadata:
    if meta.verified:
        return meta
    if meta.source not in EXTERNAL_SEARCH_SOURCES:
        return meta
    local = selector.local_fallback(candidates)
    if not local.title:
        return meta
    if is_too_generic_to_search(Candidate(title=local.title, authors=local.authors, year=local.year, source=local.source, priority=0)):
        return meta
    if meta.title and title_similarity(meta.title, local.title) >= 0.5:
        return meta
    local.notes.append(f"rejected conflicting external match from {meta.source}")
    for note in meta.notes:
        if note not in local.notes:
            local.notes.append(note)
    return local
