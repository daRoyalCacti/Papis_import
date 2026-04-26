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
from typing import Any

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


# ---------------------------------------------------------------------------
# Crossref — DOI lookup (authoritative)
# ---------------------------------------------------------------------------

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
    if not cand.title:
        return None
    params: dict[str, str] = {
        "query.title": cand.title,
        "rows": "5",
        "select": "DOI,title,author,issued,publisher,score,type",
    }
    if cand.authors:
        params["query.author"] = cand.authors[0]
    if http.mailto:
        params["mailto"] = http.mailto
    url  = f"{CROSSREF_BASE}/works?{urllib.parse.urlencode(params)}"
    data = http.get_json(url, "crossref_search", json.dumps(params, sort_keys=True),
                         bucket="crossref", min_interval=0.2)
    if not isinstance(data, dict):
        return None
    items = (data.get("message") or {}).get("items") or []
    best_score = 0.0
    best_item: dict[str, Any] | None = None
    for item in items:
        titles = item.get("title") or []
        title  = clean_text(titles[0]) if titles else ""
        if not title:
            continue
        result_authors = _crossref_authors(item)
        result_year    = _crossref_year(item)
        score = _match_score(cand, title, result_authors, result_year)
        if score > best_score:
            best_score, best_item = score, item
    if best_item is None or best_score < _MEDIUM_THRESHOLD:
        return None
    titles  = best_item.get("title") or []
    title   = clean_text(titles[0]) if titles else ""
    authors = _crossref_authors(best_item)
    year    = _crossref_year(best_item)
    return Metadata(
        title=title, authors=authors, year=year,
        doi=clean_text(best_item.get("DOI", "")),
        publisher=clean_text(best_item.get("publisher", "")),
        source="crossref_search",
        confidence=_confidence_from_score(best_score),
        verified=True,
        notes=[f"crossref_score={best_score:.3f}"],
    )


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
    year    = str((item.get("publish_date") or ""))
    year    = first_year(year)
    return Metadata(
        title=clean_text(item.get("title", "")),
        authors=authors, year=year, isbn=isbn,
        publisher=clean_text(((item.get("publishers") or [{}])[0]).get("name", "")),
        source="openlibrary_isbn", confidence="high", verified=True,
    )


def openlibrary_search(http: HttpClient, cand: Candidate) -> Metadata | None:
    if not cand.title:
        return None
    params: dict[str, str] = {"title": cand.title, "limit": "5"}
    if cand.authors:
        params["author"] = cand.authors[0]
    url  = f"{OPENLIBRARY_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    data = http.get_json(url, "openlibrary_search", json.dumps(params, sort_keys=True),
                         bucket="openlibrary", min_interval=0.2)
    if not isinstance(data, dict):
        return None
    docs = data.get("docs") or []
    best_score = 0.0
    best_doc: dict[str, Any] | None = None
    for doc in docs:
        title = clean_text(doc.get("title", ""))
        if not title:
            continue
        result_authors = [clean_text(a) for a in (doc.get("author_name") or [])]
        result_year    = str(doc.get("first_publish_year", ""))
        score = _match_score(cand, title, result_authors, result_year)
        if score > best_score:
            best_score, best_doc = score, doc
    if best_doc is None or best_score < _BOOK_THRESHOLD:
        return None
    isbn = ""
    for raw in (best_doc.get("isbn") or []):
        isbn = validate_isbn(str(raw))
        if isbn:
            break
    return Metadata(
        title=clean_text(best_doc.get("title", "")),
        authors=[clean_text(a) for a in (best_doc.get("author_name") or []) if clean_text(a)],
        year=str(best_doc.get("first_publish_year", "")),
        isbn=isbn,
        source="openlibrary_search",
        confidence=_confidence_from_score(best_score),
        verified=True,
        notes=[f"openlibrary_score={best_score:.3f}"],
    )


# ---------------------------------------------------------------------------
# Semantic Scholar — excellent for maths/stats papers
# ---------------------------------------------------------------------------

def semanticscholar_search(http: HttpClient, cand: Candidate) -> Metadata | None:
    if not cand.title:
        return None
    query  = f"{cand.title} {cand.authors[0]}" if cand.authors else cand.title
    params = {"query": query, "limit": "5", "fields": "title,authors,year,externalIds"}
    url    = f"{SEMANTIC_SCHOLAR_URL}/paper/search?{urllib.parse.urlencode(params)}"
    data   = http.get_json(url, "semanticscholar", json.dumps(params, sort_keys=True),
                           bucket="semanticscholar", min_interval=0.5)
    if not isinstance(data, dict):
        return None
    items = data.get("data") or []
    best_score = 0.0
    best_item: dict[str, Any] | None = None
    for item in items:
        title = clean_text(item.get("title", ""))
        if not title:
            continue
        result_authors = [clean_text((a or {}).get("name", "")) for a in (item.get("authors") or [])]
        result_year    = str(item.get("year", "")) if item.get("year") else ""
        score = _match_score(cand, title, result_authors, result_year)
        if score > best_score:
            best_score, best_item = score, item
    if best_item is None or best_score < _MEDIUM_THRESHOLD:
        return None
    authors = [clean_text((a or {}).get("name", "")) for a in (best_item.get("authors") or []) if (a or {}).get("name")]
    ext  = best_item.get("externalIds") or {}
    doi  = clean_text(ext.get("DOI", "") or "")
    arxiv = clean_text(ext.get("ArXiv", "") or ext.get("ARXIV", "") or "")
    return Metadata(
        title=clean_text(best_item.get("title", "")),
        authors=authors,
        year=str(best_item.get("year", "")) if best_item.get("year") else "",
        doi=doi, arxiv=arxiv,
        source="semanticscholar_search",
        confidence=_confidence_from_score(best_score),
        verified=True,
        notes=[f"semanticscholar_score={best_score:.3f}"],
    )


# ---------------------------------------------------------------------------
# OpenAlex — free, no key required
# Use --mailto to enter the polite pool for higher rate limits.
# ---------------------------------------------------------------------------

def openalex_search(http: HttpClient, cand: Candidate) -> Metadata | None:
    if not cand.title:
        return None
    params: dict[str, str] = {
        "search":   f"{cand.title} {cand.authors[0]}" if cand.authors else cand.title,
        "per-page": "5",
    }
    if cand.year:
        params["filter"] = f"publication_year:{cand.year}"
    if http.mailto:
        params["mailto"] = http.mailto
    url  = f"{OPENALEX_BASE}/works?{urllib.parse.urlencode(params)}"
    data = http.get_json(url, "openalex_search", json.dumps(params, sort_keys=True),
                         bucket="openalex", min_interval=0.15)
    if not isinstance(data, dict):
        return None
    items = data.get("results") or []
    best_score = 0.0
    best_item: dict[str, Any] | None = None
    for item in items:
        title = clean_text(item.get("display_name", "") or item.get("title", "") or "")
        if not title:
            continue
        result_authors = [
            clean_text((((a or {}).get("author") or {}).get("display_name", "")))
            for a in (item.get("authorships") or [])
        ]
        result_year = str(item.get("publication_year", "")) if item.get("publication_year") else ""
        score = _match_score(cand, title, result_authors, result_year)
        if score > best_score:
            best_score, best_item = score, item
    if best_item is None or best_score < _MEDIUM_THRESHOLD:
        return None
    authors = [
        clean_text((((a or {}).get("author") or {}).get("display_name", "")))
        for a in (best_item.get("authorships") or [])
        if (((a or {}).get("author") or {}).get("display_name", ""))
    ]
    raw_doi = (best_item.get("doi") or "") or (str((best_item.get("ids") or {}).get("doi", "")) or "")
    doi = clean_text(raw_doi.replace("https://doi.org/", "").replace("http://doi.org/", ""))
    venue = clean_text((((best_item.get("primary_location") or {}).get("source") or {}).get("display_name", "")))
    return Metadata(
        title=clean_text(best_item.get("display_name", "") or best_item.get("title", "")),
        authors=authors,
        year=str(best_item.get("publication_year", "")) if best_item.get("publication_year") else "",
        doi=doi, publisher=venue,
        source="openalex_search",
        confidence=_confidence_from_score(best_score),
        verified=True,
        notes=[f"openalex_score={best_score:.3f}"],
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
    url  = f"{GOOGLE_BOOKS_URL}?{urllib.parse.urlencode(params)}"
    data = http.get_json(url, "google_books", json.dumps(params, sort_keys=True),
                         bucket="googlebooks", min_interval=0.2)
    if not isinstance(data, dict):
        return None
    items = data.get("items") or []
    best_score = 0.0
    best_info: dict[str, Any] | None = None
    for item in items:
        info  = item.get("volumeInfo") or {}
        title = clean_text(info.get("title", ""))
        if not title:
            continue
        result_authors = [clean_text(a) for a in (info.get("authors") or [])]
        result_year    = first_year(str(info.get("publishedDate", "")))
        score = _match_score(cand, title, result_authors, result_year)
        if score > best_score:
            best_score, best_info = score, info
    if best_info is None or best_score < _BOOK_THRESHOLD:
        return None
    isbn = ""
    for ident in (best_info.get("industryIdentifiers") or []):
        isbn = validate_isbn(str((ident or {}).get("identifier", ""))) or ""
        if isbn:
            break
    return Metadata(
        title=clean_text(best_info.get("title", "")),
        authors=[clean_text(a) for a in (best_info.get("authors") or []) if clean_text(a)],
        year=first_year(str(best_info.get("publishedDate", ""))),
        isbn=isbn,
        publisher=clean_text(best_info.get("publisher", "")),
        source="google_books_search",
        confidence=_confidence_from_score(best_score),
        verified=True,
        notes=[f"google_books_score={best_score:.3f}"],
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
