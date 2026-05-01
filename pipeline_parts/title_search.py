from __future__ import annotations

import concurrent.futures
import threading
from time import perf_counter
from typing import Callable

from papis_import.http_client import HttpClient
from papis_import.models import Candidate, Metadata, TitleSearchTiming
from papis_import.resolvers import (
    crossref_search,
    google_books_search,
    openalex_search,
    openlibrary_search,
    semanticscholar_search,
)
from papis_import.utils import (
    is_garbage_title,
    is_journal_abbrev_title,
    is_journal_header_title,
    normalize_title,
)


ApplySanity = Callable[[Metadata, str, str], Metadata]
Resolver = Callable[[HttpClient, Candidate], Metadata | None]

GENERIC_TITLES = frozenset({
    "thesis", "dissertation", "paper", "notes", "chapter", "chapters",
    "lecture", "lectures", "slides", "document", "main", "draft",
    "introduction", "appendix", "preface", "summary", "abstract",
    "report", "manuscript", "preprint", "article", "book", "review",
    "homework", "exercises", "problems", "solutions", "exam", "quiz",
    "assignment", "handout", "handouts", "tutorial", "worksheet",
    "a", "b", "c", "d", "e", "ass", "pdf",
})

TITLE_SEARCH_BUCKETS = {
    "crossref": "crossref",
    "openalex": "openalex",
    "openlibrary": "openlibrary",
    "semanticscholar": "semanticscholar",
    "google_books": "googlebooks",
}


def is_too_generic_to_search(cand: Candidate) -> bool:
    words = normalize_title(cand.title).split()
    if not words:
        return True
    if len(words) == 1 and (words[0] in GENERIC_TITLES or len(words[0]) <= 3):
        return not (cand.authors and cand.year)
    if len(words) == 2 and all(w in GENERIC_TITLES for w in words):
        return not (cand.authors and cand.year)
    return False


def search_one_source(
    name: str,
    resolver: Resolver,
    candidates: list[Candidate],
    http: HttpClient,
    text: str,
    apply_sanity: ApplySanity,
    filename: str = "",
    stop_event: threading.Event | None = None,
    search_started: float | None = None,
    timeout_s: float = 0.0,
) -> tuple[Metadata | None, float, str, TitleSearchTiming]:
    """Run *resolver* against each candidate in sequence, keeping the best sanity score."""
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
            meta = resolver(http, cand)
        except Exception as exc:
            meta = None
            timing.errors += 1
            query_trace["status"] = "exception"
            query_trace["error"] = f"{type(exc).__name__}: {exc}"
            if len(timing.error_messages) < 5:
                timing.error_messages.append(f"{cand.source}: {type(exc).__name__}: {exc}")
        finally:
            http_trace = http.take_last_request_trace(TITLE_SEARCH_BUCKETS.get(name, name))
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
        apply_sanity(meta, text, filename)
        query_trace["matched"] = True
        query_trace["status"] = query_trace["status"] or "matched"
        query_trace["match_title"] = meta.title
        query_trace["match_source"] = meta.source
        query_trace["sanity_score"] = meta.sanity_score
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


def parallel_title_search(
    *,
    candidates: list[Candidate],
    http: HttpClient,
    text: str,
    max_cands: int,
    google_key: str,
    use_ss: bool,
    apply_sanity: ApplySanity,
    semantic_scholar_key: str = "",
    filename: str = "",
    timeout_s: float = 0.0,
) -> tuple[Metadata | None, float, list[TitleSearchTiming]]:
    """Query all configured resolvers in parallel; return the highest sanity-adjusted match."""
    search_cands = [
        c for c in candidates
        if c.title
        and not is_too_generic_to_search(c)
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

    wave1_jobs: list[tuple[str, Resolver]] = [
        ("crossref", crossref_search),
        ("openalex", openalex_search),
    ]
    if google_key:
        wave1_jobs.append(("google_books", lambda http, c: google_books_search(http, c, google_key)))

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

    def run_wave(jobs: list[tuple[str, Resolver]]) -> None:
        nonlocal best, best_score
        if not jobs:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futures = [
                ex.submit(
                    search_one_source,
                    name,
                    fn,
                    search_cands,
                    http,
                    text,
                    apply_sanity,
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
