from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

from papis_import.http_client import HttpClient
from papis_import.models import Candidate, IdentifierLookupTiming, Metadata, TimingBreakdown
from papis_import.resolvers import arxiv_by_id, crossref_by_doi, openlibrary_by_isbn
from papis_import.utils import extract_identifiers, jstor_filename_doi, numeric_filename_dois


ApplySanity = Callable[[Metadata, str, str], Metadata]


def collect_identifier_pool(
    path: Path,
    text: str,
    ident_text: str,
    candidates: list[Candidate],
) -> tuple[list[str], list[str], list[str], list[str]]:
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


def update_identifier_debug(
    debug: dict[str, str],
    dois: list[str],
    isbns: list[str],
    arxivs: list[str],
) -> None:
    debug["identifier_dois"] = "; ".join(dois)
    debug["identifier_isbns"] = "; ".join(isbns)
    debug["identifier_arxivs"] = "; ".join(arxivs)


def run_identifier_lookups(
    http: HttpClient,
    path: Path,
    sanity_text: str,
    dois: list[str],
    isbns: list[str],
    arxivs: list[str],
    timing: TimingBreakdown,
    tried: set[tuple[str, str]],
    apply_sanity: ApplySanity,
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
            meta = crossref_by_doi(http, doi)
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
            apply_sanity(meta, sanity_text, path.name)
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
                meta = openlibrary_by_isbn(http, isbn)
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
                apply_sanity(meta, sanity_text, path.name)
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
                meta = arxiv_by_id(http, arx)
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
                apply_sanity(meta, sanity_text, path.name)
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
