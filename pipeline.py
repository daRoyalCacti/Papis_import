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

import dataclasses
import json
from pathlib import Path

from papis_import.extractors import Extractor
from time import perf_counter
from papis_import.models import (
    Candidate,
    Metadata,
    TimingBreakdown,
)
from papis_import.pipeline_parts.candidates import CandidateSelector, extend_unique_candidates
from papis_import.pipeline_parts.finalization import ResolutionFinalizer
from papis_import.pipeline_parts.identifiers import (
    collect_identifier_pool,
    finalize_identifier_match,
    run_identifier_lookups,
    update_identifier_debug,
)
from papis_import.pipeline_parts.title_search import parallel_title_search
from papis_import.utils import (
    MIN_SANITY_SCORE,
    extract_identifiers,
    is_book_signal,
    is_unreadable_text,
    normalize_title,
    sanity_score_for_match,
)


def choose_best_local(candidates: list[Candidate]) -> Candidate | None:
    """Compatibility wrapper for callers that imported this from pipeline."""
    return CandidateSelector().choose_best_local(candidates)


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


# ---------------------------------------------------------------------------
# Main resolution function
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ResolutionRun:
    path: Path
    extractor: Extractor
    skip_vision: bool = False
    selector: CandidateSelector = dataclasses.field(default_factory=CandidateSelector)
    finalizer: ResolutionFinalizer = dataclasses.field(init=False)

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
    is_book: bool = False
    best_search: Metadata | None = None
    best_search_score: float = -1.0

    def __post_init__(self) -> None:
        self.finalizer = ResolutionFinalizer(self.selector)

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
        self.filename_best = self.selector.choose_best_local(self.filename_cands)
        self.timing.best_local_s = perf_counter() - t0
        if self.text:
            t0 = perf_counter()
            self.candidates.extend(self.extractor.text_header_candidate(self.text))
            self.timing.header_candidate_s = perf_counter() - t0

    def refresh_identifier_pool(self) -> None:
        extend_unique_candidates(self.candidates, self.selector.synthesize(self.candidates))
        self.dois, self.isbns, self.arxivs, self.stable_ids = collect_identifier_pool(
            self.path, self.text, self.ident_text, self.candidates
        )
        update_identifier_debug(self.debug, self.dois, self.isbns, self.arxivs)

    def collect_initial_identifier_pool(self) -> None:
        t0 = perf_counter()
        self.ident_text = self.extractor.get_identifier_text(self.path)
        self.timing.text_extract_s += perf_counter() - t0

        self.refresh_identifier_pool()

    def try_identifier_resolution(self) -> None:
        ident, score, note, matched_any = run_identifier_lookups(
            self.extractor.http,
            self.path,
            self.sanity_text,
            self.dois,
            self.isbns,
            self.arxivs,
            self.timing,
            self.tried_identifiers,
            _apply_sanity,
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
        return finalize_identifier_match(
            self.best_ident,
            self.selector.local_fallback(self.candidates),
            self.best_ident_note,
            self.needs_ocr_flag,
            self.debug,
        )

    def collect_deferred_candidates(self) -> None:
        t0 = perf_counter()
        grobid_cands = self.extractor.grobid_candidate(self.path)
        self.candidates.extend(grobid_cands)
        self.timing.grobid_s = perf_counter() - t0

        if self.text:
            t0 = perf_counter()
            llm_cands = self.extractor.llm_candidate(self.path, self.text, self.filename_best)
            self.candidates.extend(llm_cands)
            self.timing.text_llm_s = perf_counter() - t0
            self.debug.update({
                k: str(v)
                for k, v in getattr(self.extractor, "last_llm_debug", {}).items()
                if v is not None
            })

    def compute_book_signal(self) -> None:
        self.is_book = is_book_signal(self.path.name, self.text)

    def maybe_collect_vision_candidates(self) -> None:
        t0 = perf_counter()
        if not self.skip_vision:
            should_call_vision = True
            trigger_reason = "configured"
            if getattr(self.extractor.args, "vision_only_if_hard", False):
                has_deep_ident = bool(self.dois or self.isbns or self.arxivs)
                if has_deep_ident and self.any_identifier_matched:
                    should_call_vision = False
                    trigger_reason = "skipped: identifier lookup returned metadata"
                elif not self.needs_ocr_flag and _has_strong_searchable_candidate(self.candidates):
                    should_call_vision = False
                    trigger_reason = "skipped: readable text with strong local candidate"
                elif has_deep_ident:
                    trigger_reason = "hard-case: identifiers found but lookup failed"
                else:
                    trigger_reason = "hard-case: no_identifier"
            self.debug["vision_trigger"] = trigger_reason
            if should_call_vision:
                vision_cands, vision_dbg = self.extractor.vision_llm_candidate(
                    self.path, self.filename_best, is_book=self.is_book
                )
                extend_unique_candidates(self.candidates, vision_cands)
                self.debug.update({k: str(v) for k, v in vision_dbg.items() if v is not None})
                self.debug["vision_used"] = (
                    "yes" if self.debug.get("vision_status") not in {"not_configured", "skipped"} else "no"
                )
            else:
                self.debug["vision_status"] = "skipped"
        else:
            self.debug["vision_trigger"] = "skip_vision flag"
            self.debug["vision_status"] = "skipped"
        self.timing.vision_llm_s += perf_counter() - t0
        self.timing.vision_pacing_s = self.extractor.http.take_pacing_s("vision_llm")
        self.debug["vision_pacing_s"] = f"{self.timing.vision_pacing_s:.6f}"
        rem = self.extractor.http.remaining_tokens("vision_llm")
        if rem is not None:
            self.debug["vision_tokens_remaining"] = str(rem)

    def synthesize_candidates(self) -> None:
        extend_unique_candidates(self.candidates, self.selector.synthesize(self.candidates))

    def demote_grobid_for_books(self) -> None:
        if self.is_book:
            for c in self.candidates:
                if c.source == "grobid":
                    c.priority = max(c.priority, 30)
                    c.notes.append("grobid demoted (is_book)")

    def run_title_search(self) -> None:
        max_cands = int(getattr(self.extractor.args, "max_search_candidates", 6))
        google_key = getattr(self.extractor.args, "google_books_api_key", "")
        use_ss = not getattr(self.extractor.args, "no_semantic_scholar", False)
        ss_key = getattr(self.extractor.args, "semantic_scholar_api_key", "")

        t0 = perf_counter()
        best_search, best_search_score, title_search_timings = parallel_title_search(
            candidates=self.candidates,
            http=self.extractor.http,
            text=self.sanity_text,
            max_cands=max_cands,
            google_key=google_key,
            use_ss=use_ss,
            apply_sanity=_apply_sanity,
            semantic_scholar_key=ss_key,
            filename=self.path.name,
            timeout_s=float(getattr(self.extractor.args, "title_search_timeout", 12.0) or 0.0),
        )
        self.timing.title_search_s += perf_counter() - t0
        self.timing.title_searches.extend(title_search_timings)
        self.debug["title_search_queries_json"] = json.dumps(
            [
                query
                for source_timing in title_search_timings
                for query in source_timing.query_traces
            ],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.best_search = best_search
        self.best_search_score = best_search_score


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
    needs_ocr_flag = run.needs_ocr_flag
    debug = run.debug
    candidates = run.candidates

    # ---- Phase 1: cheap authoritative identifier lookups (DOI / ISBN / arXiv) ----
    # Each lookup is authoritative, but we still sanity-check the returned
    # metadata against the PDF text.  If multiple identifiers are present,
    # we pick the one with the highest sanity score.
    run.try_identifier_resolution()

    # If an identifier match passed the sanity check, it's the answer.
    if run.has_passing_identifier_match():
        run.selector.update_debug(debug, candidates)
        best_ident = run.finalize_identifier_match()

        timing.resolve_total_s = perf_counter() - resolve_started
        return best_ident, candidates, text, debug, timing

    # ---- Phase 2: expensive candidate sources, only after identifiers fail ----
    run.collect_deferred_candidates()

    # GROBID/text LLM may reveal new identifiers. Try them before vision,
    # since identifier lookup is still cheaper and more authoritative.
    run.refresh_identifier_pool()
    run.try_identifier_resolution()

    if run.has_passing_identifier_match():
        run.selector.update_debug(debug, candidates)
        best_ident = run.finalize_identifier_match()

        timing.resolve_total_s = perf_counter() - resolve_started
        return best_ident, candidates, text, debug, timing

    run.compute_book_signal()

    # Vision LLM is still optional, but now it runs after deep identifier
    # passes so books with ISBNs on copyright pages can avoid the expensive call.
    run.maybe_collect_vision_candidates()
    run.synthesize_candidates()

    # is_book signal: demote GROBID candidates on books. GROBID is trained on
    # journal-article headers and picks up editor/affiliation noise on books.
    run.demote_grobid_for_books()

    # Vision may reveal new identifiers. Try only identifiers that were not
    # already checked before falling back to title search.
    run.refresh_identifier_pool()
    run.try_identifier_resolution()

    if run.has_passing_identifier_match():
        run.selector.update_debug(debug, candidates)
        best_ident = run.finalize_identifier_match()

        timing.resolve_total_s = perf_counter() - resolve_started
        return best_ident, candidates, text, debug, timing

    # Capture raw per-source candidates for debug TSVs after all candidate
    # sources have run.
    run.selector.update_debug(debug, candidates)

    # ---- Phase 3: parallel title-search across all configured resolvers ----
    run.run_title_search()

    winner = run.finalizer.finalize_winner(
        best_ident=run.best_ident,
        best_ident_score=run.best_ident_score,
        best_ident_note=run.best_ident_note,
        best_search=run.best_search,
        best_search_score=run.best_search_score,
        candidates=candidates,
        sanity_text=sanity_text,
        needs_ocr_flag=needs_ocr_flag,
        debug=debug,
    )
    if winner is not None:
        timing.resolve_total_s = perf_counter() - resolve_started
        return winner, candidates, text, debug, timing

    # ---- Phase 4: no external verification — return local best ----
    fallback = run.finalizer.local_fallback(
        candidates=candidates,
        stable_ids=run.stable_ids,
        needs_ocr_flag=needs_ocr_flag,
        debug=debug,
    )

    timing.resolve_total_s = perf_counter() - resolve_started
    return fallback, candidates, text, debug, timing
