from __future__ import annotations

import json

from papis_import.models import Candidate, Metadata
from papis_import.utils import (
    is_garbage_title,
    is_journal_abbrev_title,
    is_suspicious_title,
    normalize_title,
    repair_title_ligatures,
)


class CandidateSelector:
    def choose_best_local(self, candidates: list[Candidate]) -> Candidate | None:
        useful = [c for c in candidates if c.title or c.authors or c.doi or c.isbn or c.arxiv]
        if not useful:
            return None
        useful.sort(key=lambda c: (-self._quality(c), c.priority, -len(c.title)))
        return useful[0]

    def synthesize(self, candidates: list[Candidate]) -> list[Candidate]:
        for c in candidates:
            if c.title:
                c.title = repair_title_ligatures(c.title)
        good_titles = [
            c for c in candidates
            if c.title
            and not is_garbage_title(c.title)
            and not is_journal_abbrev_title(c.title)
            and not is_suspicious_title(c.title)
        ]
        all_titles = [c for c in candidates if c.title]
        titles = good_titles or all_titles
        authors = [c for c in candidates if c.authors]
        years = [c for c in candidates if c.year]
        idents = [c for c in candidates if c.doi or c.isbn or c.arxiv]
        if not titles:
            return []
        best_t = self.choose_best_local(titles)
        best_a = self.choose_best_local(authors)
        best_y = self.choose_best_local(years)
        best_i = self.choose_best_local(idents)
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

    def local_fallback(self, candidates: list[Candidate]) -> Metadata:
        best = self.choose_best_local(candidates)
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
            title=best.title,
            authors=best.authors,
            year=best.year,
            doi=best.doi,
            isbn=best.isbn,
            arxiv=best.arxiv,
            source=best.source,
            confidence=confidence,
            verified=False,
            notes=notes + ["best local guess; external verification failed"],
        )

    def update_debug(self, debug: dict[str, str], candidates: list[Candidate]) -> None:
        for c in candidates:
            if c.source == "grobid" and not debug["grobid_title"]:
                debug["grobid_title"] = c.title
                debug["grobid_authors"] = "; ".join(c.authors)
                debug["grobid_year"] = c.year
        debug["candidate_sources"] = " | ".join(c.source for c in candidates)
        debug["candidates_json"] = _candidates_json(candidates)
        local_best = self.choose_best_local(candidates)
        if local_best is not None:
            debug["local_best_source"] = local_best.source

    def _quality(self, c: Candidate) -> float:
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


def extend_unique_candidates(candidates: list[Candidate], additions: list[Candidate]) -> None:
    seen = {_candidate_key(c) for c in candidates}
    for cand in additions:
        key = _candidate_key(cand)
        if key in seen:
            continue
        candidates.append(cand)
        seen.add(key)


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
