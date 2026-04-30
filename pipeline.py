"""Resolution pipeline: collect-all → verify-in-parallel → best-match wins.

Strategy
--------
The earlier pipeline was first-match-wins: it tried each candidate against
each resolver in sequence and returned as soon as a result cleared the
threshold.  That allowed a weak candidate (e.g. filename "ass") to land a
high-similarity match on an unrelated record before a stronger candidate
(e.g. GROBID or LLM) was even tried.

This version:

1. **Try cheap authoritative identifiers first.**  We gather cheap local
   candidates, run the deeper identifier text scan, and look up DOI/ISBN/arXiv
   before paying for GROBID or LLM calls.

2. **Identifier lookups are authoritative.**  DOIs, ISBNs, and arXiv IDs from any
   source are looked up authoritatively.  Each result is scored by the
   sanity check below, and the highest-scoring identifier match wins.

3. **Expensive candidate sources are deferred.**  GROBID and LLM candidates
   run only after the first identifier pass fails, and any new identifiers
   they reveal get one more authoritative lookup pass.

4. **Parallel title-search.**  Remaining candidates are searched across
   Crossref / Semantic Scholar / OpenAlex / OpenLibrary / Google Books
   concurrently — one thread per *resolver* (not per candidate×resolver
   pair), which keeps per-source rate-limiting sequential and lets the
   five sources run in parallel.

5. **Sanity-check every external match.**  Before a resolver result is
   accepted, we verify that the resolved title or at least one author
   surname appears in the PDF text.  Matches that fail the sanity check
   are downgraded to unverified regardless of the API's similarity score.
   This kills the "ass.pdf → crossref_score=1.000" class of false positive.

6. **Fall through to local synthesis** only when nothing verifies.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import threading
from pathlib import Path
from typing import Callable

from papis_import.extractors import Extractor
from papis_import.http_client import HttpClient
from time import perf_counter
from papis_import.models import (
    Candidate,
    IdentifierLookupTiming,
    Metadata,
    TimingBreakdown,
    TitleSearchTiming,
)
from papis_import.resolvers import (
    arxiv_by_id,
    crossref_by_doi,
    crossref_search,
    google_books_search,
    openalex_search,
    openlibrary_by_isbn,
    openlibrary_search,
    semanticscholar_search,
)
from papis_import.utils import (
    MIN_SANITY_SCORE,
    author_overlap,
    extract_identifiers,
    is_book_signal,
    is_garbage_title,
    is_journal_abbrev_title,
    is_journal_header_title,
    is_suspicious_title,
    is_unreadable_text,
    jstor_filename_doi,
    normalize_title,
    numeric_filename_dois,
    repair_title_ligatures,
    sanity_score_for_match,
    should_import,
    title_similarity,
)


# ---------------------------------------------------------------------------
# Generic titles that must never be sent to title-search alone
# ---------------------------------------------------------------------------
_GENERIC_TITLES = frozenset({
    "thesis", "dissertation", "paper", "notes", "chapter", "chapters",
    "lecture", "lectures", "slides", "document", "main", "draft",
    "introduction", "appendix", "preface", "summary", "abstract",
    "report", "manuscript", "preprint", "article", "book", "review",
    "homework", "exercises", "problems", "solutions", "exam", "quiz",
    "assignment", "handout", "handouts", "tutorial", "worksheet",
    # Added: one- and two-letter stems / short filler words that collide
    # on any search index (the ass.pdf pathology).
    "a", "b", "c", "d", "e", "ass", "pdf",
})


def _is_too_generic_to_search(cand: Candidate) -> bool:
    words = normalize_title(cand.title).split()
    if not words:
        return True
    if len(words) == 1 and (words[0] in _GENERIC_TITLES or len(words[0]) <= 3):
        return not (cand.authors and cand.year)
    if len(words) == 2 and all(w in _GENERIC_TITLES for w in words):
        return not (cand.authors and cand.year)
    return False


def _quality(c: Candidate) -> float:
    score = 0.0
    if c.title:
        if is_garbage_title(c.title):
            score -= 3.0
        elif is_journal_abbrev_title(c.title):
            score -= 2.0
        else:
            score += min(4.0, max(1.0, len(normalize_title(c.title).split()) / 3.0))
    if c.authors:
        score += min(2.0, 0.75 + 0.5 * len(c.authors))
    if c.year:
        score += 0.4
    if c.doi or c.isbn or c.arxiv:
        score += 2.5
    if c.source in {"filename_title_only", "text_header"}:
        score -= 0.8
    if c.source == "pdfinfo" and (c.title.startswith("PII:") or not c.title):
        score -= 1.5
    if c.title and len(normalize_title(c.title).split()) <= 2:
        score -= 0.5
    return score


def choose_best_local(candidates: list[Candidate]) -> Candidate | None:
    useful = [c for c in candidates if c.title or c.authors or c.doi or c.isbn or c.arxiv]
    if not useful:
        return None
    useful.sort(key=lambda c: (-_quality(c), c.priority, -len(c.title)))
    return useful[0]


def _scan_for_any_identifier(filename: str, text: str) -> bool:
    """Return True if the filename OR the first pages of text contain any
    DOI / ISBN / arXiv identifier pattern.  Used only by the
    --vision-only-if-hard gate — a positive hit means the text-based
    pipeline will (almost certainly) find something authoritative, so we
    can safely skip the vision LLM call."""
    try:
        dois, isbns, arxivs, _ = extract_identifiers(filename, text or "")
    except Exception:
        return False
    return bool(dois or isbns or arxivs)


# ---------------------------------------------------------------------------
# Synthetic candidate — combines the best title/authors/year across sources
# ---------------------------------------------------------------------------

def _synthesize(candidates: list[Candidate]) -> list[Candidate]:
    for c in candidates:
        if c.title:
            c.title = repair_title_ligatures(c.title)
    # Maillard JMLR 2021: text-LLM extracted the first author's affiliation
    # ("Université Paris-Saclay, CNRS, Inria, Laboratoire de mathématiques
    # d'Orsay…") as the title and beat the correct "Aggregated Hold-Out" from
    # GROBID/Vision in choose_best_local. is_suspicious_title rejects the
    # affiliation regardless of which extractor emitted it.
    good_titles = [c for c in candidates if c.title and not is_garbage_title(c.title)
                   and not is_journal_abbrev_title(c.title)
                   and not is_suspicious_title(c.title)]
    all_titles  = [c for c in candidates if c.title]
    titles  = good_titles or all_titles
    authors = [c for c in candidates if c.authors]
    years   = [c for c in candidates if c.year]
    idents  = [c for c in candidates if c.doi or c.isbn or c.arxiv]
    if not titles:
        return []
    best_t = choose_best_local(titles)
    best_a = choose_best_local(authors)
    best_y = choose_best_local(years)
    best_i = choose_best_local(idents)
    assert best_t is not None
    syn = Candidate(
        title=best_t.title,
        authors=(best_a.authors[:] if best_a and best_a.authors else best_t.authors[:]),
        year=(best_y.year if best_y else best_t.year),
        doi=(best_i.doi if best_i and best_i.doi else best_t.doi),
        isbn=(best_i.isbn if best_i and best_i.isbn else best_t.isbn),
        arxiv=(best_i.arxiv if best_i and best_i.arxiv else best_t.arxiv),
        source="synthesized",
        priority=min(best_t.priority, 22),
        notes=["combined local title/author/year candidates"],
    )
    return [syn] if (syn.authors or syn.year or syn.doi or syn.isbn or syn.arxiv) else []


# ---------------------------------------------------------------------------
# Local-only fallback metadata (used when nothing verifies externally)
# ---------------------------------------------------------------------------

def _local_fallback(candidates: list[Candidate]) -> Metadata:
    best = choose_best_local(candidates)
    if best is None:
        return Metadata(source="none", confidence="low", notes=["no metadata extracted"])
    notes = best.notes[:]
    best.authors = [
        a.replace("Author(s):", "").replace("author(s):", "").strip()
        for a in best.authors
    ]
    best.authors = [a for a in best.authors if a]
    confidence = "low" if best.source in {"filename_title_only", "text_header"} else "medium"
    if best.source == "pdfinfo" and best.title.startswith("PII:"):
        confidence = "low"
        notes.append("pdfinfo title looks like a publisher internal ID")
    return Metadata(
        title=best.title, authors=best.authors, year=best.year,
        doi=best.doi, isbn=best.isbn, arxiv=best.arxiv,
        source=best.source, confidence=confidence, verified=False,
        notes=notes + ["best local guess; external verification failed"],
    )


# ---------------------------------------------------------------------------
# Sanity-check wrapper — attaches a score to every external result
# ---------------------------------------------------------------------------

def _apply_sanity(meta: Metadata, text: str, filename: str = "") -> Metadata:
    """Compute the sanity score and set sanity_passed/sanity_score on *meta*.

    Does NOT modify verified/confidence — the caller decides how to react
    to the score.  Separate method so identifier lookups and title-searches
    can apply different policies.

    Parameters
    ----------
    filename : str, optional
        Original PDF filename (or full path — only the basename is used).
        Enables author-surname corroboration via the filename even when
        the PDF body text is unreadable or only contains the TOC.  Safe
        to omit; scoring falls back to text-only behaviour.
    """
    score = sanity_score_for_match(
        meta.title, meta.authors, meta.source, text, filename=filename
    )
    meta.sanity_score = round(score, 3)
    meta.sanity_passed = meta.sanity_score >= MIN_SANITY_SCORE
    return meta


def _downgrade_if_sanity_failed(meta: Metadata) -> Metadata:
    """If the sanity check failed, mark the result as unverified and drop
    confidence to medium.  The metadata is still returned — it may be the
    best guess available — but it will land in the review TSV rather than
    the auto-import TSV."""
    if not meta.sanity_passed:
        meta.verified = False
        if meta.confidence == "high":
            meta.confidence = "medium"
        meta.notes.append(f"sanity_score={meta.sanity_score:.3f} (below threshold {MIN_SANITY_SCORE})")
    else:
        meta.notes.append(f"sanity_score={meta.sanity_score:.3f} (passed)")
    return meta


def _has_strong_searchable_candidate(candidates: list[Candidate]) -> bool:
    """Return True if any candidate has a meaningful title from a reliable source.

    Used by the vision_only_if_hard gate to skip vision when the text is
    readable and something searchable already exists. The threshold is ≥4
    words so that short / ambiguous stems don't accidentally qualify.

    Excludes text_header and filename_title_only because those are the two
    weakest heuristics — they fire on every file and are often wrong, so
    their presence alone shouldn't suppress vision on a genuinely hard case.
    """
    _STRONG_SOURCES = {
        "filename_structured", "filename_author_title", "filename_series",
        "pdfinfo", "pdf_metadata", "xmp", "grobid",
    }
    for c in candidates:
        if not c.title:
            continue
        words = normalize_title(c.title).split()
        if len(words) < 4:
            continue
        if c.source in _STRONG_SOURCES:
            return True
        if c.source.startswith("llm:"):
            return True
    return False


def _is_strong_local_corroborator(source: str) -> bool:
    """Return True for local extractors strong enough to rescue an unreadable-
    text title-search hit.

    Excludes synthetic/title-only sources so we do not certify an external hit
    using the same weak signal that produced the query in the first place.
    """
    if not source:
        return False
    if source in {"synthesized", "filename_title_only", "text_header", "filename_author_only"}:
        return False
    return (
        source in {"grobid", "pdfinfo", "pdf_metadata", "xmp", "filename_author_title", "filename_structured", "filename_series"}
        or source.startswith("vision_llm:")
        or source.startswith("llm:")
    )


def _local_corroboration_note(meta: Metadata, candidates: list[Candidate]) -> str:
    """Return a note describing strong local corroboration, or "" if absent.

    Used only to rescue externally verified title-search results when the PDF
    text is unreadable, so the normal text-based sanity check cannot fire.
    """
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

    # Slightly weaker rescue path: two independent strong extractors agree on
    # a non-trivial title verbatim/near-verbatim, even if they did not recover
    # authors.  This catches cases where both vision and GROBID see the same
    # title page but authors are truncated.
    seen_title = list(dict.fromkeys(strong_title_sources))
    if len(seen_title) >= 2 and len(title_words) >= 4:
        return "title corroborated by multiple local extractors despite unreadable text via " + ", ".join(seen_title)

    return ""


# ---------------------------------------------------------------------------
# Soft auto-accept: rescue a review row when multiple independent strong
# local extractors agree on title + authors.  Output is a SUBSET of the auto
# TSV — these rows still get written to papis_import.tsv, but ALSO to
# papis_import_soft.tsv so they can be eyeballed once the run is complete.
# ---------------------------------------------------------------------------
_SOFT_AUTO_STRONG_SOURCES = {"grobid", "pdfinfo", "xmp", "text_header"}
_SOFT_AUTO_STRONG_PREFIXES = ("llm:", "vision_llm:")
_SOFT_AUTO_TITLE_SIM = 0.7
# Short but legitimate titles like "Borel Spaces" (11) or "Convex Analysis"
# (15) must be allowed; the real safeguard is multi-source corroboration plus
# the suspicious/garbage/journal-abbrev filters.
_SOFT_AUTO_MIN_TITLE_LEN = 5


def _is_strong_soft_source(source: str) -> bool:
    return source in _SOFT_AUTO_STRONG_SOURCES or source.startswith(_SOFT_AUTO_STRONG_PREFIXES)


def _evaluate_soft_auto(meta: Metadata, candidates: list[Candidate]) -> tuple[bool, list[str]]:
    """Decide whether a non-auto row should be soft auto-accepted.

    Rules (all must hold):
      1. meta.auto_safe is False  (otherwise the row is already in auto)
      2. meta.needs_ocr is False  OR  >=3 strong corroborators (instead of 2)
      3. Final title is non-empty, length >= 15, and not garbage / journal abbrev / suspicious
      4. >=2 distinct strong-source candidates have a title matching meta.title
         (token+sequence similarity >= 0.7)
      5. >=1 of those agreeing sources also corroborates the authors
         (any shared surname with meta.authors)

    Strong sources: grobid, pdfinfo, xmp, text_header, llm:*, vision_llm:*.
    Excluded: filename_*, synthesized, all external sources.
    """
    if meta.auto_safe:
        return False, []
    title = (meta.title or "").strip()
    if len(title) < _SOFT_AUTO_MIN_TITLE_LEN:
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
        if title_similarity(cand.title, title) < _SOFT_AUTO_TITLE_SIM:
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


def _rescue_locally_corroborated_match(meta: Metadata, candidates: list[Candidate], text: str) -> Metadata:
    """Allow strong local corroboration to rescue an external title-search hit.

    The standard sanity check relies on extracted text. That can fail for two
    different reasons:
      1) the text layer is unreadable/scanned, or
      2) the first short text window only sees a series page / preface rather
         than the real title page.

    In either regime, keep the external verification if an independent strong
    local extractor (especially vision) strongly agrees with it.
    """
    if meta.sanity_passed or not meta.verified:
        return meta
    if meta.source not in {
        "crossref_search", "openalex_search", "semanticscholar_search",
        "openlibrary_search", "google_books_search",
    }:
        return meta

    note = _local_corroboration_note(meta, candidates)
    if not note:
        return meta

    meta.sanity_passed = True
    meta.sanity_score = max(meta.sanity_score, 0.6)
    meta.notes.append(note)
    return meta


def _prefer_local_when_external_conflicts(meta: Metadata, candidates: list[Candidate]) -> Metadata:
    """When a title-search hit fails sanity badly and contradicts strong local
    evidence, show the local guess instead of the wrong external metadata.

    This keeps review rows actionable: ``Marginal_Likelihood.pdf`` should show
    the locally recovered talk title, not an unrelated Crossref encyclopedia
    entry pulled from the filename stem.
    """
    if meta.verified:
        return meta
    if meta.source not in {
        "crossref_search", "openalex_search", "semanticscholar_search",
        "openlibrary_search", "google_books_search",
    }:
        return meta
    local = _local_fallback(candidates)
    if not local.title:
        return meta
    if _is_too_generic_to_search(Candidate(title=local.title, authors=local.authors, year=local.year, source=local.source, priority=0)):
        return meta
    if meta.title and title_similarity(meta.title, local.title) >= 0.5:
        return meta
    local.notes.append(f"rejected conflicting external match from {meta.source}")
    for note in meta.notes:
        if note not in local.notes:
            local.notes.append(note)
    return local


# ---------------------------------------------------------------------------
# Parallel title-search helpers
# ---------------------------------------------------------------------------

# One resolver function takes (HttpClient, Candidate) and returns Metadata|None
_Resolver = Callable[[HttpClient, Candidate], "Metadata | None"]

_TITLE_SEARCH_BUCKETS = {
    "crossref": "crossref",
    "openalex": "openalex",
    "openlibrary": "openlibrary",
    "semanticscholar": "semanticscholar",
    "google_books": "googlebooks",
}


def _search_one_source(
    name: str,
    resolver: _Resolver,
    candidates: list[Candidate],
    extractor: Extractor,
    text: str,
    filename: str = "",
    stop_event: threading.Event | None = None,
    search_started: float | None = None,
    timeout_s: float = 0.0,
) -> tuple[Metadata | None, float, str, TitleSearchTiming]:
    """Run *resolver* against each candidate in sequence (per-source throttle
    safety), keeping the result with the best sanity score.

    Returns (best_meta_or_None, best_score, source_name, timing).
    """
    started = perf_counter()
    timing = TitleSearchTiming(source=name, candidates_available=len(candidates))
    best: Metadata | None = None
    best_score: float = -1.0
    for cand_idx, cand in enumerate(candidates, start=1):
        if stop_event is not None and stop_event.is_set():
            timing.stopped_early = True
            if not timing.skip_reason:
                timing.skip_reason = "strong match found by another resolver"
            break
        if timeout_s > 0 and search_started is not None and perf_counter() - search_started >= timeout_s:
            timing.stopped_early = True
            if not timing.skip_reason:
                timing.skip_reason = "title search timeout"
            break
        timing.candidates_tried += 1
        query_started = perf_counter()
        query_trace: dict[str, object] = {
            "source": name,
            "candidate_index": cand_idx,
            "candidate_source": cand.source,
            "query_title": cand.title,
            "query_authors": cand.authors,
            "query_year": cand.year,
            "elapsed_s": 0.0,
            "status": "",
            "error": "",
            "matched": False,
            "match_title": "",
            "match_source": "",
            "match_score": 0.0,
            "sanity_score": 0.0,
            "cache_hit": "",
            "http": {},
        }
        try:
            meta = resolver(extractor.http, cand)
        except Exception as exc:
            meta = None
            timing.errors += 1
            query_trace["status"] = "exception"
            query_trace["error"] = f"{type(exc).__name__}: {exc}"
            if len(timing.error_messages) < 5:
                timing.error_messages.append(f"{cand.source}: {type(exc).__name__}: {exc}")
        finally:
            http_trace = extractor.http.take_last_request_trace(_TITLE_SEARCH_BUCKETS.get(name, name))
            if http_trace:
                query_trace["http"] = http_trace
                query_trace["cache_hit"] = http_trace.get("cache_hit", "")
                query_trace["status"] = query_trace["status"] or str(http_trace.get("final_status", ""))
            query_trace["elapsed_s"] = perf_counter() - query_started
        if meta is None:
            if not query_trace["status"]:
                query_trace["status"] = "no_match"
            timing.query_traces.append(query_trace)
            continue
        timing.matches_returned += 1
        _apply_sanity(meta, text, filename)
        query_trace["matched"] = True
        query_trace["status"] = query_trace["status"] or "matched"
        query_trace["match_title"] = meta.title
        query_trace["match_source"] = meta.source
        query_trace["sanity_score"] = meta.sanity_score
        # Slight preference for results the API itself scored as high,
        # so that when sanity scores tie we prefer the higher-confidence
        # external match.
        score = meta.sanity_score
        if meta.confidence == "high":
            score += 0.05
        query_trace["match_score"] = score
        if score > best_score:
            best_score = score
            best = meta
            timing.best_score = score
            timing.best_source = meta.source
            timing.best_title = meta.title
            timing.best_verified = meta.verified
            timing.best_sanity_passed = meta.sanity_passed
            timing.best_sanity_score = meta.sanity_score
        if meta.verified and meta.sanity_passed and meta.confidence == "high":
            timing.stopped_early = True
            timing.skip_reason = "high-confidence sanity-passing match"
            if stop_event is not None:
                stop_event.set()
            timing.query_traces.append(query_trace)
            break
        timing.query_traces.append(query_trace)
    timing.elapsed_s = perf_counter() - started
    return best, best_score, name, timing


def _parallel_title_search(
    candidates: list[Candidate],
    extractor: Extractor,
    text: str,
    max_cands: int,
    google_key: str,
    use_ss: bool,
    semantic_scholar_key: str = "",
    filename: str = "",
    timeout_s: float = 0.0,
) -> tuple[Metadata | None, float, list[TitleSearchTiming]]:
    """Query all configured resolvers in parallel; return the (meta, score)
    pair with the highest sanity-adjusted score across all of them."""
    search_cands = [
        c for c in candidates
        if c.title
        and not _is_too_generic_to_search(c)
        and not is_journal_header_title(c.title)
        and not is_garbage_title(c.title)
        and not is_journal_abbrev_title(c.title)
    ]
    search_cands.sort(key=lambda c: c.priority)
    search_cands = search_cands[:max_cands]
    if not search_cands:
        return None, -1.0, [
            TitleSearchTiming(source="crossref", skipped=True, skip_reason="no search candidates"),
            TitleSearchTiming(source="openalex", skipped=True, skip_reason="no search candidates"),
            TitleSearchTiming(source="openlibrary", skipped=True, skip_reason="no search candidates"),
            TitleSearchTiming(source="semanticscholar", skipped=True, skip_reason="no search candidates"),
            TitleSearchTiming(source="google_books", skipped=True, skip_reason="no search candidates"),
        ]

    # Wave 1: fast sources run in parallel.  OpenLibrary search can be very
    # slow (30-90 s/request when their CDN is under load), so it is deferred
    # to wave 2 and only run if wave 1 failed to find a strong match.
    wave1_jobs: list[tuple[str, _Resolver]] = [
        ("crossref",    crossref_search),
        ("openalex",    openalex_search),
    ]
    if google_key:
        wave1_jobs.append(("google_books",
                           lambda http, c: google_books_search(http, c, google_key)))

    best: Metadata | None = None
    best_score: float = -1.0
    timings: list[TitleSearchTiming] = []
    search_started = perf_counter()
    stop_event = threading.Event()

    if not use_ss:
        timings.append(TitleSearchTiming(
            source="semanticscholar",
            candidates_available=len(search_cands),
            skipped=True,
            skip_reason="disabled",
        ))
    if not google_key:
        timings.append(TitleSearchTiming(
            source="google_books",
            candidates_available=len(search_cands),
            skipped=True,
            skip_reason="no api key",
        ))

    def is_strong(meta: Metadata | None) -> bool:
        return bool(meta and meta.verified and meta.sanity_passed and meta.confidence == "high")

    def budget_exhausted() -> bool:
        return timeout_s > 0 and perf_counter() - search_started >= timeout_s

    def run_wave(jobs: list[tuple[str, _Resolver]]) -> None:
        nonlocal best, best_score
        if not jobs:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futures = [
                ex.submit(
                    _search_one_source,
                    name,
                    fn,
                    search_cands,
                    extractor,
                    text,
                    filename,
                    stop_event,
                    search_started,
                    timeout_s,
                )
                for name, fn in jobs
            ]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    meta, score, _name, source_timing = fut.result()
                    timings.append(source_timing)
                except Exception as exc:
                    timings.append(TitleSearchTiming(
                        source="unknown",
                        candidates_available=len(search_cands),
                        errors=1,
                        error_messages=[f"{type(exc).__name__}: {exc}"],
                    ))
                    continue
                if meta and score > best_score:
                    best_score = score
                    best = meta
                if is_strong(meta):
                    stop_event.set()

    run_wave(wave1_jobs)

    # Wave 2: OpenLibrary — only if wave 1 didn't already find a strong match
    # and the time budget hasn't expired.
    if is_strong(best) or budget_exhausted():
        timings.append(TitleSearchTiming(
            source="openlibrary",
            candidates_available=len(search_cands),
            skipped=True,
            skip_reason=(
                "cheaper resolver found high-confidence sanity-passing match"
                if is_strong(best) else "title search timeout"
            ),
        ))
    else:
        run_wave([("openlibrary", openlibrary_search)])

    if use_ss:
        if is_strong(best):
            timings.append(TitleSearchTiming(
                source="semanticscholar",
                candidates_available=len(search_cands),
                skipped=True,
                skip_reason="cheaper resolver found high-confidence sanity-passing match",
            ))
        elif budget_exhausted():
            timings.append(TitleSearchTiming(
                source="semanticscholar",
                candidates_available=len(search_cands),
                skipped=True,
                skip_reason="title search timeout",
            ))
        else:
            run_wave([(
                "semanticscholar",
                lambda http, c: semanticscholar_search(http, c, semantic_scholar_key),
            )])

    return best, best_score, timings


def _collect_identifier_pool(path, text: str, ident_text: str, candidates: list[Candidate]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Collect DOI/ISBN/arXiv identifiers from raw text, filename, and candidates."""
    dois, isbns, arxivs, stable_ids = extract_identifiers(path.name, text, ident_text)
    for c in candidates:
        for val, bucket in ((c.doi, dois), (c.isbn, isbns), (c.arxiv, arxivs)):
            if val and val not in bucket:
                bucket.append(val)
    jstor_doi = jstor_filename_doi(path.stem)
    if jstor_doi and jstor_doi not in dois:
        dois.insert(0, jstor_doi)
    for extra_doi in numeric_filename_dois(path.stem):
        if extra_doi not in dois:
            dois.append(extra_doi)
    return dois, isbns, arxivs, stable_ids


def _update_identifier_debug(debug: dict[str, str], dois: list[str], isbns: list[str], arxivs: list[str]) -> None:
    debug["identifier_dois"] = "; ".join(dois)
    debug["identifier_isbns"] = "; ".join(isbns)
    debug["identifier_arxivs"] = "; ".join(arxivs)


def _candidates_json(candidates: list[Candidate]) -> str:
    return json.dumps(
        [
            {
                "source": c.source,
                "title": c.title,
                "authors": c.authors,
                "year": c.year,
                "doi": c.doi,
                "isbn": c.isbn,
                "arxiv": c.arxiv,
                "priority": c.priority,
                "notes": c.notes,
            }
            for c in candidates
        ],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _update_candidate_debug(debug: dict[str, str], candidates: list[Candidate]) -> None:
    for c in candidates:
        if c.source == "grobid" and not debug["grobid_title"]:
            debug["grobid_title"] = c.title
            debug["grobid_authors"] = "; ".join(c.authors)
            debug["grobid_year"] = c.year
    debug["candidate_sources"] = " | ".join(c.source for c in candidates)
    debug["candidates_json"] = _candidates_json(candidates)
    local_best = choose_best_local(candidates)
    if local_best is not None:
        debug["local_best_source"] = local_best.source


def _candidate_key(c: Candidate) -> tuple:
    return (
        c.source,
        c.title,
        tuple(c.authors),
        c.year,
        c.doi,
        c.isbn,
        c.arxiv,
    )


def _extend_unique_candidates(candidates: list[Candidate], additions: list[Candidate]) -> None:
    seen = {_candidate_key(c) for c in candidates}
    for cand in additions:
        key = _candidate_key(cand)
        if key in seen:
            continue
        candidates.append(cand)
        seen.add(key)


def _run_identifier_lookups(
    extractor: Extractor,
    path,
    sanity_text: str,
    dois: list[str],
    isbns: list[str],
    arxivs: list[str],
    timing: TimingBreakdown,
    tried: set[tuple[str, str]],
) -> tuple[Metadata | None, float, str, bool]:
    """Run authoritative identifier lookups, skipping values already tried."""
    best_ident: Metadata | None = None
    best_ident_score: float = -1.0
    best_ident_note = ""
    matched_any = False
    t_ident = perf_counter()

    for doi in dois:
        key = ("doi", doi)
        if key in tried:
            continue
        tried.add(key)
        lookup_started = perf_counter()
        error = ""
        try:
            meta = crossref_by_doi(extractor.http, doi)
        except Exception as exc:
            meta = None
            error = str(exc)
        elapsed = perf_counter() - lookup_started

        item = IdentifierLookupTiming(
            kind="doi",
            value=doi,
            resolver="crossref_by_doi",
            elapsed_s=elapsed,
            matched=meta is not None,
            source=(meta.source if meta else ""),
            error=error,
        )
        if meta:
            matched_any = True
            _apply_sanity(meta, sanity_text, path.name)
            item.sanity_score = meta.sanity_score
            if meta.sanity_score > best_ident_score:
                best_ident = meta
                best_ident_score = meta.sanity_score
                best_ident_note = "resolved by DOI via Crossref"
        timing.identifier_lookups.append(item)
        if best_ident is not None and best_ident.sanity_passed:
            timing.identifier_lookups_s += perf_counter() - t_ident
            return best_ident, best_ident_score, best_ident_note, matched_any

    if best_ident is None or not best_ident.sanity_passed:
        for isbn in isbns:
            key = ("isbn", isbn)
            if key in tried:
                continue
            tried.add(key)
            lookup_started = perf_counter()
            error = ""
            try:
                meta = openlibrary_by_isbn(extractor.http, isbn)
            except Exception as exc:
                meta = None
                error = str(exc)
            elapsed = perf_counter() - lookup_started

            item = IdentifierLookupTiming(
                kind="isbn",
                value=isbn,
                resolver="openlibrary_by_isbn",
                elapsed_s=elapsed,
                matched=meta is not None,
                source=(meta.source if meta else ""),
                error=error,
            )
            if meta:
                matched_any = True
                _apply_sanity(meta, sanity_text, path.name)
                item.sanity_score = meta.sanity_score
                if meta.sanity_score > best_ident_score:
                    best_ident = meta
                    best_ident_score = meta.sanity_score
                    best_ident_note = "resolved by ISBN via OpenLibrary"
            timing.identifier_lookups.append(item)
            if best_ident is not None and best_ident.sanity_passed:
                timing.identifier_lookups_s += perf_counter() - t_ident
                return best_ident, best_ident_score, best_ident_note, matched_any

    if best_ident is None or not best_ident.sanity_passed:
        for arx in arxivs:
            key = ("arxiv", arx)
            if key in tried:
                continue
            tried.add(key)
            lookup_started = perf_counter()
            error = ""
            try:
                meta = arxiv_by_id(extractor.http, arx)
            except Exception as exc:
                meta = None
                error = str(exc)
            elapsed = perf_counter() - lookup_started

            item = IdentifierLookupTiming(
                kind="arxiv",
                value=arx,
                resolver="arxiv_by_id",
                elapsed_s=elapsed,
                matched=meta is not None,
                source=(meta.source if meta else ""),
                error=error,
            )
            if meta:
                matched_any = True
                _apply_sanity(meta, sanity_text, path.name)
                item.sanity_score = meta.sanity_score
                if meta.sanity_score > best_ident_score:
                    best_ident = meta
                    best_ident_score = meta.sanity_score
                    best_ident_note = "resolved by arXiv ID"
            timing.identifier_lookups.append(item)
            if best_ident is not None and best_ident.sanity_passed:
                timing.identifier_lookups_s += perf_counter() - t_ident
                return best_ident, best_ident_score, best_ident_note, matched_any

    timing.identifier_lookups_s += perf_counter() - t_ident
    return best_ident, best_ident_score, best_ident_note, matched_any


def _finalize_identifier_match(
    meta: Metadata,
    candidates: list[Candidate],
    note: str,
    needs_ocr_flag: bool,
    debug: dict[str, str],
) -> Metadata:
    meta.merge_missing(_local_fallback(candidates))
    if note:
        meta.notes.append(note)
    meta.notes.append(f"sanity_score={meta.sanity_score:.3f} (passed)")
    meta.needs_ocr = needs_ocr_flag
    meta.auto_safe = (
        meta.verified
        and meta.sanity_passed
        and meta.confidence == "high"
    )
    debug["final_source"] = meta.source
    return meta


# ---------------------------------------------------------------------------
# Main resolution function
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ResolutionRun:
    path: Path
    extractor: Extractor
    skip_vision: bool = False

    timing: TimingBreakdown = dataclasses.field(default_factory=TimingBreakdown)
    resolve_started: float = dataclasses.field(default_factory=perf_counter)
    text: str = ""
    sanity_text: str = ""
    filename_cands: list[Candidate] = dataclasses.field(default_factory=list)
    filename_best: Candidate | None = None
    needs_ocr_flag: bool = False
    debug: dict[str, str] = dataclasses.field(default_factory=dict)
    candidates: list[Candidate] = dataclasses.field(default_factory=list)
    ident_text: str = ""
    dois: list[str] = dataclasses.field(default_factory=list)
    isbns: list[str] = dataclasses.field(default_factory=list)
    arxivs: list[str] = dataclasses.field(default_factory=list)
    stable_ids: list[str] = dataclasses.field(default_factory=list)
    tried_identifiers: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    best_ident: Metadata | None = None
    best_ident_score: float = -1.0
    best_ident_note: str = ""
    any_identifier_matched: bool = False

    def extract_initial_text(self) -> None:
        t0 = perf_counter()
        self.text = self.extractor.get_text(self.path)
        self.sanity_text = self.extractor.get_sanity_text(self.path)
        if not self.sanity_text:
            self.sanity_text = self.text
        self.filename_cands = self.extractor.filename_candidate(self.path)
        self.needs_ocr_flag = is_unreadable_text(self.text)
        self.timing.text_extract_s += perf_counter() - t0

    def init_debug(self) -> None:
        self.debug = {
            "vision_used": "no",
            "vision_trigger": "",
            "vision_status": "not_attempted",
            "vision_error": "",
            "final_source": "",
            "grobid_used": "yes" if bool(getattr(self.extractor.args, "grobid_url", "")) else "no",
            "grobid_title": "",
            "grobid_authors": "",
            "grobid_year": "",
            "vision_model": getattr(self.extractor.args, "vision_llm_model", "") or "",
            "vision_pages": str(getattr(self.extractor.args, "vision_pages", "") or ""),
            "vision_dpi": str(getattr(self.extractor.args, "vision_dpi", "") or ""),
            "vision_title": "",
            "vision_authors": "",
            "vision_year": "",
            "text_llm_used": "no",
            "text_llm_status": "not_attempted",
            "text_llm_error": "",
            "text_llm_model": getattr(self.extractor.args, "llm_model", "") or "",
            "text_llm_http_json": "",
            "text_llm_title": "",
            "text_llm_authors": "",
            "text_llm_year": "",
            "local_best_source": "",
            "candidate_sources": "",
            "identifier_dois": "",
            "identifier_isbns": "",
            "identifier_arxivs": "",
            "candidates_json": "",
            "title_search_queries_json": "",
        }

    def collect_cheap_local_candidates(self) -> None:
        t0 = perf_counter()
        self.candidates.extend(self.extractor.embedded_metadata(self.path))
        self.candidates.extend(self.extractor.pdfinfo_metadata(self.path))
        self.timing.embedded_metadata_s = perf_counter() - t0

        self.candidates.extend(self.filename_cands)

        t0 = perf_counter()
        self.filename_best = choose_best_local(self.filename_cands)
        self.timing.best_local_s = perf_counter() - t0
        if self.text:
            t0 = perf_counter()
            self.candidates.extend(self.extractor.text_header_candidate(self.text))
            self.timing.header_candidate_s = perf_counter() - t0

    def refresh_identifier_pool(self) -> None:
        _extend_unique_candidates(self.candidates, _synthesize(self.candidates))
        self.dois, self.isbns, self.arxivs, self.stable_ids = _collect_identifier_pool(
            self.path, self.text, self.ident_text, self.candidates
        )
        _update_identifier_debug(self.debug, self.dois, self.isbns, self.arxivs)

    def collect_initial_identifier_pool(self) -> None:
        t0 = perf_counter()
        self.ident_text = self.extractor.get_identifier_text(self.path)
        self.timing.text_extract_s += perf_counter() - t0

        self.refresh_identifier_pool()

    def try_identifier_resolution(self) -> None:
        ident, score, note, matched_any = _run_identifier_lookups(
            self.extractor,
            self.path,
            self.sanity_text,
            self.dois,
            self.isbns,
            self.arxivs,
            self.timing,
            self.tried_identifiers,
        )
        self.any_identifier_matched = self.any_identifier_matched or matched_any
        if ident is not None and score > self.best_ident_score:
            self.best_ident = ident
            self.best_ident_score = score
            self.best_ident_note = note

    def has_passing_identifier_match(self) -> bool:
        return self.best_ident is not None and self.best_ident.sanity_passed

    def finalize_identifier_match(self) -> Metadata:
        assert self.best_ident is not None
        return _finalize_identifier_match(
            self.best_ident,
            self.candidates,
            self.best_ident_note,
            self.needs_ocr_flag,
            self.debug,
        )


def resolve(
    path,
    extractor: Extractor,
    *,
    skip_vision: bool = False,
) -> tuple[Metadata, list[Candidate], str, dict[str, str], TimingBreakdown]:
    """Full pipeline: extract → verify → return (Metadata, all_candidates, raw_text).

    The returned Metadata has `needs_ocr`, `sanity_passed`, `sanity_score`,
    and `auto_safe` populated to drive CLI-level decisions (two-TSV split,
    OCR retry).

    Parameters
    ----------
    skip_vision : bool, default False
        When True, the vision LLM candidate source is not called.  The CLI
        sets this on OCR retries — vision works on the rendered PDF pages,
        which barely change after OCR (OCR adds a text layer but the
        images are the same), so re-running vision on the OCR'd file
        would just pay for the same inference a second time.
    """

    run = ResolutionRun(path=path, extractor=extractor, skip_vision=skip_vision)
    run.extract_initial_text()
    run.init_debug()
    run.collect_cheap_local_candidates()
    run.collect_initial_identifier_pool()

    timing = run.timing
    resolve_started = run.resolve_started
    text = run.text
    sanity_text = run.sanity_text
    filename_best = run.filename_best
    needs_ocr_flag = run.needs_ocr_flag
    debug = run.debug
    candidates = run.candidates

    max_cands  = int(getattr(extractor.args, "max_search_candidates", 6))
    google_key = getattr(extractor.args, "google_books_api_key", "")
    use_ss     = not getattr(extractor.args, "no_semantic_scholar", False)
    ss_key     = getattr(extractor.args, "semantic_scholar_api_key", "")

    # ---- Phase 1: cheap authoritative identifier lookups (DOI / ISBN / arXiv) ----
    # Each lookup is authoritative, but we still sanity-check the returned
    # metadata against the PDF text.  If multiple identifiers are present,
    # we pick the one with the highest sanity score.
    run.try_identifier_resolution()

    # If an identifier match passed the sanity check, it's the answer.
    if run.has_passing_identifier_match():
        _update_candidate_debug(debug, candidates)
        best_ident = run.finalize_identifier_match()

        timing.resolve_total_s = perf_counter() - resolve_started
        return best_ident, candidates, text, debug, timing

    # ---- Phase 2: expensive candidate sources, only after identifiers fail ----
    t0 = perf_counter()
    grobid_cands = extractor.grobid_candidate(path)
    candidates.extend(grobid_cands)
    timing.grobid_s = perf_counter() - t0

    if text:
        t0 = perf_counter()
        llm_cands = extractor.llm_candidate(path, text, filename_best)
        candidates.extend(llm_cands)
        timing.text_llm_s = perf_counter() - t0
        debug.update({k: str(v) for k, v in getattr(extractor, "last_llm_debug", {}).items() if v is not None})

    # GROBID/text LLM may reveal new identifiers. Try them before vision,
    # since identifier lookup is still cheaper and more authoritative.
    run.refresh_identifier_pool()
    run.try_identifier_resolution()

    if run.has_passing_identifier_match():
        _update_candidate_debug(debug, candidates)
        best_ident = run.finalize_identifier_match()

        timing.resolve_total_s = perf_counter() - resolve_started
        return best_ident, candidates, text, debug, timing

    # Compute book signal once; reused for both the vision tier choice and the
    # GROBID demotion below so we don't call is_book_signal twice.
    _is_book = is_book_signal(path.name, text)

    # Vision LLM is still optional, but now it runs after deep identifier
    # passes so books with ISBNs on copyright pages can avoid the expensive call.
    t0 = perf_counter()
    if not skip_vision:
        should_call_vision = True
        trigger_reason = "configured"
        if getattr(extractor.args, "vision_only_if_hard", False):
            has_deep_ident = bool(run.dois or run.isbns or run.arxivs)
            if has_deep_ident and run.any_identifier_matched:
                should_call_vision = False
                trigger_reason = "skipped: identifier lookup returned metadata"
            elif not needs_ocr_flag and _has_strong_searchable_candidate(candidates):
                should_call_vision = False
                trigger_reason = "skipped: readable text with strong local candidate"
            elif has_deep_ident:
                trigger_reason = "hard-case: identifiers found but lookup failed"
            else:
                trigger_reason = "hard-case: no_identifier"
        debug["vision_trigger"] = trigger_reason
        if should_call_vision:
            vision_cands, vision_dbg = extractor.vision_llm_candidate(
                path, filename_best, is_book=_is_book
            )
            _extend_unique_candidates(candidates, vision_cands)
            debug.update({k: str(v) for k, v in vision_dbg.items() if v is not None})
            debug["vision_used"] = "yes" if debug.get("vision_status") not in {"not_configured", "skipped"} else "no"
        else:
            debug["vision_status"] = "skipped"
    else:
        debug["vision_trigger"] = "skip_vision flag"
        debug["vision_status"] = "skipped"
    timing.vision_llm_s += perf_counter() - t0
    timing.vision_pacing_s = extractor.http.take_pacing_s("vision_llm")
    debug["vision_pacing_s"] = f"{timing.vision_pacing_s:.6f}"
    rem = extractor.http.remaining_tokens("vision_llm")
    if rem is not None:
        debug["vision_tokens_remaining"] = str(rem)

    _extend_unique_candidates(candidates, _synthesize(candidates))

    # is_book signal: demote GROBID candidates on books. GROBID is trained on
    # journal-article headers and picks up editor/affiliation noise on books.
    if _is_book:
        for c in candidates:
            if c.source == "grobid":
                c.priority = max(c.priority, 30)
                c.notes.append("grobid demoted (is_book)")

    # Vision may reveal new identifiers. Try only identifiers that were not
    # already checked before falling back to title search.
    run.refresh_identifier_pool()
    run.try_identifier_resolution()

    if run.has_passing_identifier_match():
        _update_candidate_debug(debug, candidates)
        best_ident = run.finalize_identifier_match()

        timing.resolve_total_s = perf_counter() - resolve_started
        return best_ident, candidates, text, debug, timing

    # Capture raw per-source candidates for debug TSVs after all candidate
    # sources have run.
    _update_candidate_debug(debug, candidates)

    # ---- Phase 3: parallel title-search across all configured resolvers ----
    t0 = perf_counter()
    best_search, best_search_score, title_search_timings = _parallel_title_search(
        candidates, extractor, sanity_text, max_cands, google_key, use_ss,
        semantic_scholar_key=ss_key,
        filename=path.name,
        timeout_s=float(getattr(extractor.args, "title_search_timeout", 12.0) or 0.0),
    )
    timing.title_search_s += perf_counter() - t0
    timing.title_searches.extend(title_search_timings)
    debug["title_search_queries_json"] = json.dumps(
        [
            query
            for source_timing in title_search_timings
            for query in source_timing.query_traces
        ],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )

    # Pick the better of identifier-lookup (failed sanity) vs title-search.
    # Prefer whichever has the higher sanity score if both are present.
    winners: list[tuple[Metadata, float, str]] = []
    if run.best_ident is not None:
        winners.append((run.best_ident, run.best_ident_score, run.best_ident_note))
    if best_search is not None:
        winners.append((best_search, best_search_score, "title-search match"))

    if winners:
        winners.sort(key=lambda x: -x[1])
        winner, _score, note = winners[0]
        winner.merge_missing(_local_fallback(candidates))
        winner.notes.append(note)
        _rescue_locally_corroborated_match(winner, candidates, sanity_text)
        _downgrade_if_sanity_failed(winner)
        if not winner.sanity_passed:
            winner = _prefer_local_when_external_conflicts(winner, candidates)
        winner.needs_ocr = needs_ocr_flag
        winner.auto_safe = (winner.verified
                            and winner.sanity_passed
                            and winner.confidence == "high")
        winner.soft_auto, winner.soft_auto_reasons = _evaluate_soft_auto(winner, candidates)
        debug["final_source"] = winner.source

        timing.resolve_total_s = perf_counter() - resolve_started
        return winner, candidates, text, debug, timing

    # ---- Phase 4: no external verification — return local best ----
    fallback = _local_fallback(candidates)
    if run.stable_ids:
        fallback.notes.append(f"JSTOR stable ID present (not an arXiv ID): {run.stable_ids[0]}")
    if needs_ocr_flag:
        fallback.notes.append("PDF text appears unreadable — OCR recommended")
    fallback.needs_ocr = needs_ocr_flag
    fallback.sanity_passed = False
    fallback.sanity_score = 0.0
    fallback.auto_safe = False
    fallback.soft_auto, fallback.soft_auto_reasons = _evaluate_soft_auto(fallback, candidates)
    debug["final_source"] = fallback.source

    timing.resolve_total_s = perf_counter() - resolve_started
    return fallback, candidates, text, debug, timing
