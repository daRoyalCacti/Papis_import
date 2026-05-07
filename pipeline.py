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
import dataclasses
from pathlib import Path
from time import perf_counter

from papis_import.extractor_parts import ExtractorSet
from papis_import.http_client import HttpClient
from papis_import.models import (
    Candidate,
    Metadata,
    TimingBreakdown,
)
from papis_import.pipeline_parts.candidates import CandidateSelector
from papis_import.pipeline_parts.debug import PipelineDebug
from papis_import.pipeline_parts.finalization import ResolutionFinalizer
from papis_import.pipeline_parts.phases import CandidatePhase, IdentifierPhase, VisionPhase


@dataclasses.dataclass
class ResolutionRun:
    """Shared state container for one PDF resolution run.

    Phase classes (CandidatePhase, IdentifierPhase, VisionPhase) receive a
    reference to this object and read/write its fields directly.  The only
    method is ``finish()``, which seals timing and returns the pipeline result.
    """

    path: Path
    extractors: ExtractorSet
    args: argparse.Namespace
    http: HttpClient
    skip_vision: bool = False

    # Coordinators (init=False, set in __post_init__)
    selector: CandidateSelector = dataclasses.field(default_factory=CandidateSelector)
    finalizer: ResolutionFinalizer = dataclasses.field(init=False)
    debug: PipelineDebug = dataclasses.field(init=False)

    # Timing
    timing: TimingBreakdown = dataclasses.field(default_factory=TimingBreakdown)
    resolve_started: float = dataclasses.field(default_factory=perf_counter)

    # CandidatePhase state
    text: str = ""
    sanity_text: str = ""
    filename_cands: list[Candidate] = dataclasses.field(default_factory=list)
    filename_best: Candidate | None = None
    needs_ocr_flag: bool = False
    candidates: list[Candidate] = dataclasses.field(default_factory=list)
    is_book: bool = False
    best_search: Metadata | None = None
    best_search_score: float = -1.0

    # IdentifierPhase state
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
    best_ident_corroborated: bool = False
    uncorroborated_ident: Metadata | None = None
    uncorroborated_ident_note: str = ""
    uncorroborated_ident_score: float = -1.0

    def __post_init__(self) -> None:
        self.finalizer = ResolutionFinalizer(self.selector)
        self.debug = PipelineDebug(self.args)

    def finish(self, meta: Metadata) -> tuple[Metadata, list[Candidate], str, dict[str, str], TimingBreakdown]:
        self.timing.resolve_total_s = perf_counter() - self.resolve_started
        return meta, self.candidates, self.text, dict(self.debug), self.timing


def resolve(
    path,
    extractors: ExtractorSet,
    args: argparse.Namespace,
    http: HttpClient,
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

    run = ResolutionRun(path=path, extractors=extractors, args=args, http=http, skip_vision=skip_vision)
    candidates = CandidatePhase(run)
    identifiers = IdentifierPhase(run)
    vision = VisionPhase(run)

    candidates.extract_initial_text()
    candidates.collect_cheap_local_candidates()
    identifiers.collect_initial_identifier_pool()

    # ---- Phase 1: cheap authoritative identifier lookups (DOI / ISBN / arXiv) ----
    if (winner := identifiers.run_phase()) is not None:
        return run.finish(winner)

    # ---- Phase 2: expensive candidate sources, only after identifiers fail ----
    candidates.collect_deferred_candidates()

    # GROBID/text LLM may reveal new identifiers; try before vision.
    if (winner := identifiers.run_phase()) is not None:
        return run.finish(winner)

    # Safe mode: re-check whether GROBID/LLM candidates now corroborate the
    # stashed uncorroborated identifier (corroboration is re-evaluated against
    # the richer candidate pool after each extractor phase).
    if (winner := identifiers.recheck_uncorroborated()) is not None:
        return run.finish(winner)

    candidates.compute_book_signal()

    # Vision LLM runs after deep identifier passes so books with ISBNs on
    # copyright pages can skip the expensive call.
    vision.collect_candidates()
    candidates.synthesize_candidates()

    # For image-only PDFs where OCR produced no text, vision evidence is the
    # only document-level signal we have.  Use the vision candidates' titles and
    # authors as a pseudo-text haystack so the sanity check can score title-search
    # matches against that evidence rather than against nothing.
    if not run.sanity_text and run.needs_ocr_flag:
        vision_texts = []
        for c in run.candidates:
            if c.source.startswith("vision"):
                if c.title:
                    vision_texts.append(c.title)
                if c.authors:
                    vision_texts.append(" ".join(c.authors))
        if vision_texts:
            run.sanity_text = " ".join(vision_texts)

    # Demote GROBID on books: it's trained on article headers and picks up
    # editor/affiliation noise on book cover pages.
    candidates.demote_grobid_for_books()

    # Vision may reveal new identifiers not checked before.
    if (winner := identifiers.run_phase()) is not None:
        return run.finish(winner)

    # Safe mode: re-check whether vision candidates corroborate the stashed
    # uncorroborated identifier.  This is the primary path for case 003 and
    # similar where vision reads the title page but ran after the first
    # corroboration check.
    if (winner := identifiers.recheck_uncorroborated()) is not None:
        return run.finish(winner)

    # Safe mode escalation: if we still have an uncorroborated identifier, retry
    # vision with more pages to catch books whose title page is past page 4.
    if run.uncorroborated_ident is not None:
        vision.collect_escalated_candidates()
        candidates.synthesize_candidates()
        if (winner := identifiers.recheck_uncorroborated()) is not None:
            return run.finish(winner)

    # Capture raw per-source candidates for debug TSVs after all sources ran.
    run.selector.update_debug(run.debug, run.candidates)

    # ---- Phase 3: parallel title-search across all configured resolvers ----
    candidates.run_title_search()

    winner = run.finalizer.finalize_winner(
        best_ident=run.best_ident,
        best_ident_score=run.best_ident_score,
        best_ident_note=run.best_ident_note,
        best_search=run.best_search,
        best_search_score=run.best_search_score,
        candidates=run.candidates,
        sanity_text=run.sanity_text,
        needs_ocr_flag=run.needs_ocr_flag,
        debug=run.debug,
    )
    if winner is not None:
        return run.finish(winner)

    # ---- Phase 4: no external verification ----
    # Prefer a sanity-passed identifier that safe mode couldn't corroborate over
    # the purely local fallback — the identifier is usually more reliable than
    # whatever the local extractors synthesised from a poisoned PDF (e.g. a cover
    # page showing only the LNCS series header instead of the actual book title).
    if run.uncorroborated_ident is not None:
        fallback = run.finalizer.finalize_uncorroborated_identifier(
            run.uncorroborated_ident,
            run.uncorroborated_ident_note,
            run.candidates,
            run.needs_ocr_flag,
            run.debug,
        )
    else:
        fallback = run.finalizer.local_fallback(
            candidates=run.candidates,
            stable_ids=run.stable_ids,
            needs_ocr_flag=run.needs_ocr_flag,
            debug=run.debug,
        )

    return run.finish(fallback)
