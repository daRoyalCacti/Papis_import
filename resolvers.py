"""External metadata resolvers.

Each function takes an HttpClient and a Candidate (or an identifier string)
and returns a verified Metadata object, or None if lookup failed or scored
below the acceptance threshold.

Notes on the services used
---------------------------
Crossref       — free, no key, add --mailto for polite pool (higher rate limit).
arXiv          — free, no key.
OpenLibrary    — free, no key.  Good for books.
Semantic Scholar — free, no key.  Excellent for maths/stats papers.
OpenAlex       — free, no key.  Pass --mailto as a courtesy for polite pool.
                 (The --openalex-api-key flag is kept for backward compat but
                 OpenAlex no longer sells API keys; everything is free.)
Google Books   — free tier, optional key via --google-books-api-key.
"""
from __future__ import annotations

import json
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Callable

from papis_import.http_client import HttpClient
from papis_import.models import Candidate, Metadata
from papis_import.utils import (
    ARXIV_API_URL,
    CROSSREF_BASE,
    GOOGLE_BOOKS_URL,
    OPENALEX_BASE,
    OPENLIBRARY_BOOKS_URL,
    OPENLIBRARY_SEARCH_URL,
    SEMANTIC_SCHOLAR_URL,
    author_overlap,
    clean_text,
    extract_identifiers,
    first_year,
    split_authors,
    title_similarity,
    validate_isbn,
)

# Thresholds for accepting a title-search result as a match
_HIGH_THRESHOLD   = 0.97
_MEDIUM_THRESHOLD = 0.90
_BOOK_THRESHOLD   = 0.92   # OpenLibrary / Google Books tend to be noisier

# ---------------------------------------------------------------------------
# Score helpers shared across search functions
# ---------------------------------------------------------------------------

def _match_score(cand: Candidate, result_title: str, result_authors: list[str], result_year: str) -> float:
    sim    = title_similarity(cand.title, result_title)
    abonus = 0.05 if cand.authors and author_overlap(cand.authors, result_authors) >= 0.5 else 0.0
    ybonus = 0.05 if cand.year and result_year and cand.year == result_year else 0.0
    return sim + abonus + ybonus


def _confidence_from_score(score: float) -> str:
    if score >= _HIGH_THRESHOLD:
        return "high"
    if score >= _MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _search_with(
    cand: Candidate,
    *,
    fetcher: Callable[[], Any],
    item_iter: Callable[[dict[str, Any]], list[Any]],
    parse_item: Callable[[Any], tuple[str, list[str], str, dict[str, Any]]],
    source: str,
    threshold: float,
) -> Metadata | None:
    """Shared skeleton for the five title-search functions.

    *parse_item* returns (title, authors, year, extra_kwargs) where
    extra_kwargs are additional Metadata fields (doi, isbn, arxiv, publisher).
    """
    if not cand.title:
        return None
    data = fetcher()
    if not isinstance(data, dict):
        return None
    items = item_iter(data)
    best_score = 0.0
    best_item = None
    for item in items:
        title, result_authors, result_year, _ = parse_item(item)
        if not title:
            continue
        score = _match_score(cand, title, result_authors, result_year)
        if score > best_score:
            best_score, best_item = score, item
    if best_item is None or best_score < threshold:
        return None
    title, authors, year, extra = parse_item(best_item)
    return Metadata(
        title=title, authors=authors, year=year,
        source=source,
        confidence=_confidence_from_score(best_score),
        verified=True,
        notes=[f"{source}_score={best_score:.3f}"],
        **extra,
    )


# ---------------------------------------------------------------------------
# Crossref — DOI lookup (authoritative)
# ---------------------------------------------------------------------------

def _crossref_authors(item: dict[str, Any]) -> list[str]:
    authors = []
    for a in (item.get("author") or []):
        full = clean_text(" ".join(x for x in (a.get("given", ""), a.get("family", "")) if x))
        if full:
            authors.append(full)
    return authors


def _crossref_year(item: dict[str, Any]) -> str:
    for key in ("published-print", "published-online", "issued", "created"):
        entry = item.get(key) or {}
        parts = (entry.get("date-parts") or [[]])[0]
        if parts:
            return str(parts[0])
    return ""


def crossref_by_doi(http: HttpClient, doi: str) -> Metadata | None:
    doi = doi.strip()
    if not doi:
        return None
    url = f"{CROSSREF_BASE}/works/{urllib.parse.quote(doi, safe='')}"
    if http.mailto:
        url += f"?mailto={urllib.parse.quote(http.mailto)}"
    data = http.get_json(url, "crossref_doi", doi, bucket="crossref", min_interval=0.2)
    if not isinstance(data, dict) or "message" not in data:
        return None
    item = data["message"]
    titles = item.get("title") or []
    title  = clean_text(titles[0]) if titles else ""
    if not title:
        return None
    authors = _crossref_authors(item)
    year    = _crossref_year(item)
    return Metadata(
        title=title, authors=authors, year=year,
        doi=clean_text(item.get("DOI", "")),
        publisher=clean_text(item.get("publisher", "")),
        source="crossref_doi", confidence="high", verified=True,
    )


def crossref_search(http: HttpClient, cand: Candidate) -> Metadata | None:
    params: dict[str, str] = {
        "query.title": cand.title,
        "rows": "5",
        "select": "DOI,title,author,issued,publisher,score,type",
    }
    if cand.authors:
        params["query.author"] = cand.authors[0]
    if http.mailto:
        params["mailto"] = http.mailto
    url = f"{CROSSREF_BASE}/works?{urllib.parse.urlencode(params)}"

    def parse_item(item: dict[str, Any]) -> tuple[str, list[str], str, dict[str, Any]]:
        titles = item.get("title") or []
        title  = clean_text(titles[0]) if titles else ""
        return title, _crossref_authors(item), _crossref_year(item), {
            "doi": clean_text(item.get("DOI", "")),
            "publisher": clean_text(item.get("publisher", "")),
        }

    return _search_with(
        cand,
        fetcher=lambda: http.get_json(url, "crossref_search", json.dumps(params, sort_keys=True),
                                      bucket="crossref", min_interval=0.2),
        item_iter=lambda data: (data.get("message") or {}).get("items") or [],
        parse_item=parse_item,
        source="crossref_search",
        threshold=_MEDIUM_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# OpenLibrary — ISBN lookup (authoritative for books)
# ---------------------------------------------------------------------------

def openlibrary_by_isbn(http: HttpClient, isbn: str) -> Metadata | None:
    isbn = validate_isbn(isbn)
    if not isbn:
        return None
    url  = f"{OPENLIBRARY_BOOKS_URL}?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
    data = http.get_json(url, "openlibrary_isbn", isbn, bucket="openlibrary", min_interval=0.2)
    if not isinstance(data, dict):
        return None
    item = data.get(f"ISBN:{isbn}")
    if not isinstance(item, dict):
        return None
    authors = [clean_text(a.get("name", "")) for a in (item.get("authors") or []) if a.get("name")]
    year    = first_year(str(item.get("publish_date") or ""))
    return Metadata(
        title=clean_text(item.get("title", "")),
        authors=authors, year=year, isbn=isbn,
        publisher=clean_text(((item.get("publishers") or [{}])[0]).get("name", "")),
        source="openlibrary_isbn", confidence="high", verified=True,
    )


def openlibrary_search(http: HttpClient, cand: Candidate) -> Metadata | None:
    params: dict[str, str] = {"title": cand.title, "limit": "5"}
    if cand.authors:
        params["author"] = cand.authors[0]
    url = f"{OPENLIBRARY_SEARCH_URL}?{urllib.parse.urlencode(params)}"

    def parse_item(doc: dict[str, Any]) -> tuple[str, list[str], str, dict[str, Any]]:
        title   = clean_text(doc.get("title", ""))
        authors = [clean_text(a) for a in (doc.get("author_name") or [])]
        year    = str(doc.get("first_publish_year", ""))
        isbn = ""
        for raw in (doc.get("isbn") or []):
            isbn = validate_isbn(str(raw))
            if isbn:
                break
        return title, authors, year, {"isbn": isbn}

    return _search_with(
        cand,
        fetcher=lambda: http.get_json(url, "openlibrary_search", json.dumps(params, sort_keys=True),
                                      bucket="openlibrary", min_interval=0.2),
        item_iter=lambda data: data.get("docs") or [],
        parse_item=parse_item,
        source="openlibrary_search",
        threshold=_BOOK_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Semantic Scholar — excellent for maths/stats papers
# ---------------------------------------------------------------------------

def semanticscholar_search(
    http: HttpClient,
    cand: Candidate,
    api_key: str = "",
) -> Metadata | None:
    query  = f"{cand.title} {cand.authors[0]}" if cand.authors else cand.title
    params = {"query": query, "limit": "5", "fields": "title,authors,year,externalIds"}
    url    = f"{SEMANTIC_SCHOLAR_URL}/paper/search?{urllib.parse.urlencode(params)}"
    headers = {"x-api-key": api_key.strip()} if api_key.strip() else None

    def parse_item(item: dict[str, Any]) -> tuple[str, list[str], str, dict[str, Any]]:
        title   = clean_text(item.get("title", ""))
        authors = [clean_text((a or {}).get("name", "")) for a in (item.get("authors") or [])]
        year    = str(item.get("year", "")) if item.get("year") else ""
        ext     = item.get("externalIds") or {}
        return title, authors, year, {
            "doi":   clean_text(ext.get("DOI", "") or ""),
            "arxiv": clean_text(ext.get("ArXiv", "") or ext.get("ARXIV", "") or ""),
        }

    return _search_with(
        cand,
        fetcher=lambda: http.get_json(
            url, "semanticscholar", json.dumps(params, sort_keys=True),
            headers=headers,
            bucket="semanticscholar",
            min_interval=1.05 if api_key.strip() else 0.5,
            max_retries=3 if api_key.strip() else 1,
        ),
        item_iter=lambda data: data.get("data") or [],
        parse_item=parse_item,
        source="semanticscholar_search",
        threshold=_MEDIUM_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# OpenAlex — free, no key required
# Use --mailto to enter the polite pool for higher rate limits.
# ---------------------------------------------------------------------------

def openalex_search(http: HttpClient, cand: Candidate) -> Metadata | None:
    params: dict[str, str] = {
        "search":   f"{cand.title} {cand.authors[0]}" if cand.authors else cand.title,
        "per-page": "5",
    }
    if cand.year:
        params["filter"] = f"publication_year:{cand.year}"
    if http.mailto:
        params["mailto"] = http.mailto
    url = f"{OPENALEX_BASE}/works?{urllib.parse.urlencode(params)}"

    def parse_item(item: dict[str, Any]) -> tuple[str, list[str], str, dict[str, Any]]:
        title = clean_text(item.get("display_name", "") or item.get("title", "") or "")
        authors = [
            clean_text((((a or {}).get("author") or {}).get("display_name", "")))
            for a in (item.get("authorships") or [])
        ]
        year = str(item.get("publication_year", "")) if item.get("publication_year") else ""
        raw_doi = (item.get("doi") or "") or (str((item.get("ids") or {}).get("doi", "")) or "")
        doi = clean_text(raw_doi.replace("https://doi.org/", "").replace("http://doi.org/", ""))
        venue = clean_text(
            (((item.get("primary_location") or {}).get("source") or {}).get("display_name", ""))
        )
        return title, [a for a in authors if a], year, {"doi": doi, "publisher": venue}

    return _search_with(
        cand,
        fetcher=lambda: http.get_json(url, "openalex_search", json.dumps(params, sort_keys=True),
                                      bucket="openalex", min_interval=0.15),
        item_iter=lambda data: data.get("results") or [],
        parse_item=parse_item,
        source="openalex_search",
        threshold=_MEDIUM_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Google Books — useful for books, requires optional API key
# ---------------------------------------------------------------------------

def google_books_search(http: HttpClient, cand: Candidate, api_key: str) -> Metadata | None:
    if not cand.title or not api_key:
        return None
    q_parts = [f"intitle:{cand.title}"]
    if cand.authors:
        q_parts.append(f"inauthor:{cand.authors[0]}")
    params = {"q": " ".join(q_parts), "maxResults": "5", "key": api_key}
    url = f"{GOOGLE_BOOKS_URL}?{urllib.parse.urlencode(params)}"

    def parse_item(item: dict[str, Any]) -> tuple[str, list[str], str, dict[str, Any]]:
        info   = item.get("volumeInfo") or {}
        title  = clean_text(info.get("title", ""))
        authors = [clean_text(a) for a in (info.get("authors") or [])]
        year   = first_year(str(info.get("publishedDate", "")))
        isbn = ""
        for ident in (info.get("industryIdentifiers") or []):
            isbn = validate_isbn(str((ident or {}).get("identifier", ""))) or ""
            if isbn:
                break
        return title, [a for a in authors if a], year, {
            "isbn":      isbn,
            "publisher": clean_text(info.get("publisher", "")),
        }

    return _search_with(
        cand,
        fetcher=lambda: http.get_json(url, "google_books", json.dumps(params, sort_keys=True),
                                      bucket="googlebooks", min_interval=0.2),
        item_iter=lambda data: data.get("items") or [],
        parse_item=parse_item,
        source="google_books_search",
        threshold=_BOOK_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# arXiv — ID lookup (authoritative)
# ---------------------------------------------------------------------------

def arxiv_by_id(http: HttpClient, arxiv_id: str) -> Metadata | None:
    arxiv_id = clean_text(arxiv_id)
    if not arxiv_id:
        return None
    url = f"{ARXIV_API_URL}?id_list={urllib.parse.quote(arxiv_id)}"
    xml_text = http.get_xml(url, "arxiv_id", arxiv_id, bucket="arxiv", min_interval=3.1)
    if not xml_text.strip():
        return None
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None
    ns    = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        return None
    title   = clean_text(entry.findtext("atom:title", default="", namespaces=ns) or "")
    authors = [
        clean_text(el.text or "")
        for el in entry.findall("atom:author/atom:name", ns)
        if clean_text(el.text or "")
    ]
    published = clean_text(entry.findtext("atom:published", default="", namespaces=ns) or "")
    blob      = clean_text(" ".join(entry.itertext()))
    dois, isbns, _, _ = extract_identifiers(blob)
    return Metadata(
        title=title, authors=authors,
        year=first_year(published),
        arxiv=arxiv_id,
        doi=dois[0] if dois else "",
        isbn=isbns[0] if isbns else "",
        source="arxiv_id", confidence="high", verified=True,
    )
