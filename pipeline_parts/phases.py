"""Phase coordinator classes for the resolution pipeline.

Each class receives the shared ``ResolutionRun`` data container and
groups a cohesive set of methods around one concern:
- CandidatePhase: text extraction and candidate collection
- IdentifierPhase: DOI/ISBN/arXiv lookups and identifier-based resolution
- VisionPhase: vision-LLM gating and invocation
"""
from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from papis_import.pipeline_parts.candidates import extend_unique_candidates
from papis_import.pipeline_parts.finalization import (
    AUTHORITATIVE_IDENTIFIER_SOURCES,
    _identifier_corroboration_decision,
)
from papis_import.pipeline_parts.identifiers import (
    collect_identifier_pool,
    run_identifier_lookups,
    update_identifier_debug,
)
from papis_import.pipeline_parts.sanity import apply_sanity
from papis_import.pipeline_parts.title_search import build_queries_json, parallel_title_search
from papis_import.pipeline_parts.vision_gate import has_strong_searchable_candidate
from papis_import.utils import is_book_signal, is_unreadable_text

if TYPE_CHECKING:
    from papis_import.models import Candidate, Metadata
    from papis_import.pipeline import ResolutionRun


class CandidatePhase:
    """Text extraction and local candidate collection."""

    def __init__(self, run: ResolutionRun) -> None:
        self.run = run

    def extract_initial_text(self) -> None:
        run = self.run
        t0 = perf_counter()
        run.text = run.extractors.text.get_text(run.path)
        run.sanity_text = run.extractors.text.get_sanity_text(run.path)
        if not run.sanity_text:
            run.sanity_text = run.text
        run.filename_cands = run.extractors.filename.candidate(run.path)
        run.needs_ocr_flag = is_unreadable_text(run.text)
        run.timing.text_extract_s += perf_counter() - t0

    def collect_cheap_local_candidates(self) -> None:
        run = self.run
        t0 = perf_counter()
        run.candidates.extend(run.extractors.pdf.embedded_metadata(run.path))
        run.candidates.extend(run.extractors.pdf.pdfinfo_metadata(run.path))
        run.timing.embedded_metadata_s = perf_counter() - t0

        run.candidates.extend(run.filename_cands)

        t0 = perf_counter()
        run.filename_best = run.selector.choose_best_local(run.filename_cands)
        run.timing.best_local_s = perf_counter() - t0
        if run.text:
            t0 = perf_counter()
            run.candidates.extend(run.extractors.text_header.candidate(run.text))
            run.timing.header_candidate_s = perf_counter() - t0

    def collect_deferred_candidates(self) -> None:
        run = self.run
        t0 = perf_counter()
        run.candidates.extend(run.extractors.grobid.candidate(run.path))
        run.timing.grobid_s = perf_counter() - t0

        # Skip text_LLM when an uncorroborated identifier match already exists:
        # the LLM reads the same pdf_text byte stream that produced it, so it
        # cannot provide independent evidence.  GROBID (above) uses page layout
        # and is the independent signal worth running first.
        if run.text and run.uncorroborated_ident is None:
            t0 = perf_counter()
            llm_cands = run.extractors.text_llm.candidate(run.path, run.text, run.filename_best)
            run.candidates.extend(llm_cands)
            run.timing.text_llm_s = perf_counter() - t0
            run.debug.update({
                k: str(v)
                for k, v in run.extractors.text_llm.last_debug.items()
                if v is not None
            })

    def compute_book_signal(self) -> None:
        run = self.run
        run.is_book = is_book_signal(run.path.name, run.text)

    def synthesize_candidates(self) -> None:
        run = self.run
        extend_unique_candidates(run.candidates, run.selector.synthesize(run.candidates))

    def demote_grobid_for_books(self) -> None:
        run = self.run
        if run.is_book:
            for c in run.candidates:
                if c.source == "grobid":
                    c.priority = max(c.priority, 30)
                    c.notes.append("grobid demoted (is_book)")

    def run_title_search(self) -> None:
        run = self.run
        max_cands = int(getattr(run.args, "max_search_candidates", 6))
        google_key = getattr(run.args, "google_books_api_key", "")
        use_ss = not getattr(run.args, "no_semantic_scholar", False)
        ss_key = getattr(run.args, "semantic_scholar_api_key", "")

        t0 = perf_counter()
        best_search, best_search_score, title_search_timings = parallel_title_search(
            candidates=run.candidates,
            http=run.http,
            text=run.sanity_text,
            max_cands=max_cands,
            google_key=google_key,
            use_ss=use_ss,
            apply_sanity=apply_sanity,
            semantic_scholar_key=ss_key,
            filename=run.path.name,
            timeout_s=float(getattr(run.args, "title_search_timeout", 12.0) or 0.0),
        )
        run.timing.title_search_s += perf_counter() - t0
        run.timing.title_searches.extend(title_search_timings)
        run.debug["title_search_queries_json"] = build_queries_json(title_search_timings)
        run.best_search = best_search
        run.best_search_score = best_search_score


class IdentifierPhase:
    """DOI / ISBN / arXiv lookup and identifier-based early resolution."""

    def __init__(self, run: ResolutionRun) -> None:
        self.run = run

    def collect_initial_identifier_pool(self) -> None:
        run = self.run
        t0 = perf_counter()
        run.ident_text = run.extractors.text.get_identifier_text(run.path)
        run.timing.text_extract_s += perf_counter() - t0
        self._refresh()

    def _refresh(self) -> None:
        run = self.run
        extend_unique_candidates(run.candidates, run.selector.synthesize(run.candidates))
        run.dois, run.isbns, run.arxivs, run.stable_ids = collect_identifier_pool(
            run.path, run.text, run.ident_text, run.candidates
        )
        update_identifier_debug(run.debug, run.dois, run.isbns, run.arxivs)

    def _try_lookups(self) -> None:
        run = self.run
        ident, score, note, matched_any = run_identifier_lookups(
            run.http,
            run.path,
            run.sanity_text,
            run.dois,
            run.isbns,
            run.arxivs,
            run.timing,
            run.tried_identifiers,
            apply_sanity,
        )
        run.any_identifier_matched = run.any_identifier_matched or matched_any
        if ident is not None and score > run.best_ident_score:
            run.best_ident = ident
            run.best_ident_score = score
            run.best_ident_note = note

    def _has_passing_match(self) -> bool:
        run = self.run
        return run.best_ident is not None and run.best_ident.sanity_passed

    def run_phase(self) -> Metadata | None:
        """Refresh pool → lookup → finalize if a match passed sanity (default mode)
        or passed sanity AND local corroboration (safe mode).

        In 'safe' accept mode (--accept-mode safe) the identifier result is only
        accepted when at least one independent local extractor agrees on both title
        and author.  When corroboration fails the result is discarded and the
        pipeline falls through to let GROBID / LLM / vision run, then tries the
        next untried identifier against the richer candidate pool.
        """
        run = self.run
        self._refresh()
        self._try_lookups()
        if not self._has_passing_match():
            return None
        assert run.best_ident is not None
        accept_mode = getattr(run.args, "accept_mode", "default")
        note = run.best_ident_note
        force_review = False
        soft_reason = ""
        if accept_mode == "safe":
            decision = _identifier_corroboration_decision(run.best_ident, run.candidates)
            if not decision.accepted:
                # Preserve the best sanity-passed rejection so Phase 4 can use it
                # as a review fallback if no other winner emerges.
                if (
                    run.best_ident.sanity_passed
                    and run.best_ident.source in AUTHORITATIVE_IDENTIFIER_SOURCES
                    and run.best_ident_score > run.uncorroborated_ident_score
                ):
                    run.uncorroborated_ident = run.best_ident
                    run.uncorroborated_ident_note = run.best_ident_note
                    run.uncorroborated_ident_score = run.best_ident_score
                # Discard so the next round tries the next identifier against a
                # richer candidate pool.
                run.best_ident = None
                run.best_ident_score = -1.0
                run.best_ident_note = ""
                return None
            if decision.force_review and not decision.soft_reason and not _has_deferred_candidate(run.candidates):
                # A single cheap filename corroborator is enough to improve a
                # review row, but not enough to stop before LLM/GROBID have had
                # a chance to provide a second independent signal.
                return None
            force_review = decision.force_review
            soft_reason = decision.soft_reason
            suffix = decision.note or f"corroborated by {decision.source}"
            note = f"{note} ({suffix})"
        run.selector.update_debug(run.debug, run.candidates)
        run.best_ident_corroborated = True
        return run.finalizer.finalize_identifier_winner(
            run.best_ident, run.candidates, note, run.needs_ocr_flag, run.debug,
            force_review=force_review, soft_reason=soft_reason,
        )


def _has_deferred_candidate(candidates: list[Candidate]) -> bool:
    return any(
        c.source == "grobid" or c.source.startswith(("llm:", "vision_llm:"))
        for c in candidates
    )


class VisionPhase:
    """Vision-LLM gating and candidate collection."""

    def __init__(self, run: ResolutionRun) -> None:
        self.run = run

    def collect_candidates(self) -> None:
        run = self.run
        t0 = perf_counter()
        if not run.skip_vision:
            should_call_vision = True
            trigger_reason = "configured"
            if getattr(run.args, "vision_only_if_hard", False):
                has_deep_ident = bool(run.dois or run.isbns or run.arxivs)
                accept_mode = getattr(run.args, "accept_mode", "default")
                # In safe mode, an unverified identifier match doesn't mean we
                # should skip vision — vision might provide the corroboration needed.
                # In default mode, restore the original gate: any metadata returned
                # by an identifier API call is enough to skip vision.
                skip_for_ident = (
                    run.best_ident_corroborated
                    if accept_mode == "safe"
                    else run.any_identifier_matched
                )
                if has_deep_ident and skip_for_ident:
                    should_call_vision = False
                    trigger_reason = "skipped: identifier lookup returned metadata"
                elif not run.needs_ocr_flag and has_strong_searchable_candidate(run.candidates):
                    should_call_vision = False
                    trigger_reason = "skipped: readable text with strong local candidate"
                elif has_deep_ident:
                    trigger_reason = "hard-case: identifiers found but lookup failed"
                else:
                    trigger_reason = "hard-case: no_identifier"
            run.debug["vision_trigger"] = trigger_reason
            if should_call_vision:
                vision_cands, vision_dbg = run.extractors.vision_llm.candidate(
                    run.path, run.filename_best, is_book=run.is_book
                )
                extend_unique_candidates(run.candidates, vision_cands)
                run.debug.update({k: str(v) for k, v in vision_dbg.items() if v is not None})
                run.debug["vision_used"] = (
                    "yes" if run.debug.get("vision_status") not in {"not_configured", "skipped"} else "no"
                )
            else:
                run.debug["vision_status"] = "skipped"
        else:
            run.debug["vision_trigger"] = "skip_vision flag"
            run.debug["vision_status"] = "skipped"
        run.timing.vision_llm_s += perf_counter() - t0
        run.timing.vision_pacing_s = run.http.take_pacing_s("vision_llm")
        run.debug["vision_pacing_s"] = f"{run.timing.vision_pacing_s:.6f}"
        rem = run.http.remaining_tokens("vision_llm")
        if rem is not None:
            run.debug["vision_tokens_remaining"] = str(rem)
