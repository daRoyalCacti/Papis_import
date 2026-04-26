"""Local metadata extraction: embedded PDF metadata, filename parsing,
text-header heuristics, GROBID (optional), and LLM fallback (optional).

LLM support
-----------
Any OpenAI-compatible chat-completions endpoint is supported, including:
  - Groq (free tier, no local download):
      endpoint  https://api.groq.com/openai/v1
      model     llama-3.1-8b-instant  or  llama-3.3-70b-versatile
      sign up   https://console.groq.com
  - OpenRouter (free models available):
      endpoint  https://openrouter.ai/api/v1
      model     meta-llama/llama-3.1-8b-instruct:free
      sign up   https://openrouter.ai
  - Ollama (local, no API key):
      endpoint  http://localhost:11434/v1
      model     llama3.2  (or whatever you have pulled)

The LLM is only invoked as a last resort when no identifier was found and
title-search verification also failed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

from papis_import.http_client import HttpClient
from papis_import.models import Candidate
from papis_import.utils import (
    ANNA_NAME_RE,
    ANNA_SPLIT_RE,
    HEX32_RE,
    LEADING_SERIES_RE,
    NOISE_LINE_PATTERNS,
    OLLAMA_CHAT_URL,
    YEAR_RE,
    clean_filename_text,
    clean_text,
    extract_identifiers,
    first_year,
    is_garbage_pdfinfo_title,
    is_suspicious_title,
    is_journal_abbrev_title,
    is_journal_header_title,
    read_cmd,
    repair_ligature_splits,
    split_authors,
    strip_footnote_markers,
    strip_trailing_title_metadata,
    validate_isbn,
)


def _coerce_str(val: object) -> str:
    """Coerce an LLM output field to a plain string.

    LLMs occasionally return lists for fields that should be strings
    (e.g. {"title": ["Part 1", "Part 2"]}). Join list items with a space;
    convert anything else with str(), treating None as "".
    """
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x)
    return str(val)


def _is_all_caps_phrase(line: str) -> bool:
    """Return True when *line* looks like an ALL-CAPS subtitle rather than an
    author name.  Heuristics:
    - All letters are uppercase (digits/spaces/basic punctuation are neutral)
    - At least two words
    - No comma (author names have commas; subtitles typically don't)
    - No period-after-initial pattern like "E. T." (author initials)
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Must contain at least one letter and all letters must be uppercase
    letters = [c for c in stripped if c.isalpha()]
    if not letters or not all(c.isupper() for c in letters):
        return False
    if "," in stripped:
        return False
    if re.search(r"\b[A-Z]\.\s", stripped):   # initials like "E. T."
        return False
    return len(stripped.split()) >= 2


class Extractor:
    """Wraps all local (non-network) extraction strategies plus optional LLM."""

    def __init__(self, args: argparse.Namespace, http: HttpClient) -> None:
        self.args = args
        self.http = http
        self.last_llm_debug: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # PDF text
    # ------------------------------------------------------------------

    def get_text(self, path: Path) -> str:
        """Extract text from the first N pages and repair common artifacts."""
        pages = max(1, int(self.args.text_pages))
        raw = read_cmd(["pdftotext", "-f", "1", "-l", str(pages), str(path), "-"])
        return repair_ligature_splits(raw)

    def get_identifier_text(self, path: Path) -> str:
        """Extract text from more pages specifically for ISBN / DOI scanning.

        The copyright page (where ISBNs are printed) is typically page 3-4 of
        a book PDF, so we read further here than for title/author heuristics.
        """
        pages = max(8, int(self.args.text_pages))
        return read_cmd(["pdftotext", "-f", "1", "-l", str(pages), str(path), "-"])

    def get_sanity_text(self, path: Path) -> str:
        """Extract a slightly larger front-matter window for sanity checks.

        Many books hide the real title page on page 3-5 after a series page,
        half-title, or publisher imprint. Using only the first 1-2 pages makes
        otherwise clean matches look unsupported, which sends good PDFs like
        ``convex_5.pdf`` and ``3-540-49730-7.pdf`` to review.
        """
        pages = max(5, int(self.args.text_pages))
        raw = read_cmd(["pdftotext", "-f", "1", "-l", str(pages), str(path), "-"])
        return repair_ligature_splits(raw)

    # ------------------------------------------------------------------
    # Embedded PDF metadata (standard + XMP/Dublin Core)
    # ------------------------------------------------------------------

    def embedded_metadata(self, path: Path) -> list[Candidate]:
        if PdfReader is None:
            return []
        out: list[Candidate] = []
        try:
            reader = PdfReader(str(path))
        except Exception:
            return []

        # Standard /Title /Author fields
        try:
            meta = reader.metadata or {}
        except Exception:
            meta = {}
        title  = clean_text(str(meta.get("/Title", "")))
        author = clean_text(str(meta.get("/Author", "")))
        if title or author:
            cand = Candidate(
                title=title,
                authors=split_authors(author),
                year=first_year(
                    str(meta.get("/Subject", "")),
                    str(meta.get("/Keywords", "")),
                    str(meta.get("/CreationDate", "")),
                ),
                source="pdf_metadata",
                priority=10,
            )
            dois, isbns, arxivs, _ = extract_identifiers(title, author)
            cand.doi   = dois[0]   if dois   else ""
            cand.isbn  = isbns[0]  if isbns  else ""
            cand.arxiv = arxivs[0] if arxivs else ""
            out.append(cand)

        # XMP / Dublin Core
        try:
            xmp = reader.xmp_metadata
        except Exception:
            xmp = None
        if xmp is not None:
            xmp_title   = ""
            xmp_authors: list[str] = []
            for attr in ("dc_title", "dc_subject", "dc_description"):
                value = getattr(xmp, attr, None)
                if value and not xmp_title:
                    xmp_title = clean_text(
                        " ".join(str(v) for v in value.values())
                        if isinstance(value, dict)
                        else " ".join(str(v) for v in value)
                        if isinstance(value, list)
                        else str(value)
                    )
            for attr in ("dc_creator", "dc_contributor"):
                value = getattr(xmp, attr, None)
                if value:
                    vals = value if isinstance(value, list) else [value]
                    xmp_authors.extend(clean_text(str(v)) for v in vals if clean_text(str(v)))
            if xmp_title or xmp_authors:
                cand = Candidate(
                    title=xmp_title,
                    authors=xmp_authors,
                    year=first_year(
                        str(getattr(xmp, "xmp_create_date", "")),
                        str(getattr(xmp, "xmp_modify_date", "")),
                    ),
                    source="xmp",
                    priority=5,
                )
                dois, isbns, arxivs, _ = extract_identifiers(xmp_title, " ".join(xmp_authors))
                cand.doi   = dois[0]   if dois   else ""
                cand.isbn  = isbns[0]  if isbns  else ""
                cand.arxiv = arxivs[0] if arxivs else ""
                out.append(cand)
        return out

    # ------------------------------------------------------------------
    # pdfinfo
    # ------------------------------------------------------------------

    def pdfinfo_metadata(self, path: Path) -> list[Candidate]:
        txt = read_cmd(["pdfinfo", str(path)])
        if not txt:
            return []
        fields: dict[str, str] = {}
        for line in txt.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            fields[k.strip().lower()] = clean_text(v)
        title  = fields.get("title", "")
        author = fields.get("author", "")
        # Discard titles that are clearly software-generated garbage
        if is_garbage_pdfinfo_title(title):
            title = ""
        if not title and not author:
            return []
        cand = Candidate(
            title=title,
            authors=[strip_footnote_markers(a) for a in split_authors(author) if strip_footnote_markers(a)],
            year=first_year(fields.get("creationdate", ""), fields.get("moddate", "")),
            source="pdfinfo",
            priority=15,
        )
        dois, isbns, arxivs, _ = extract_identifiers(title, author)
        cand.doi   = dois[0]   if dois   else ""
        cand.isbn  = isbns[0]  if isbns  else ""
        cand.arxiv = arxivs[0] if arxivs else ""
        return [cand]

    # ------------------------------------------------------------------
    # Filename parsing
    # ------------------------------------------------------------------

    def filename_candidate(self, path: Path) -> list[Candidate]:
        stem_raw = clean_text(path.stem)
        stem_raw = re.sub(r"\s+-\s+libgen\.[A-Za-z0-9]+$", "", stem_raw, flags=re.I)
        stem_raw = re.sub(r"\s+-\s+Anna.?s Archive(?:-\d+)?$", "", stem_raw, flags=re.I)
        stem_raw = re.sub(r"(\(\d{4}\))-(\d+)$", r"\1", stem_raw)
        stem = clean_filename_text(stem_raw)
        out: list[Candidate] = []
        dois, isbns, arxivs, stable_ids = extract_identifiers(stem_raw, stem)

        # ---- Anna's Archive / libgen structured names (title -- author -- ...) ----
        if " -- " in stem_raw:
            pieces = [clean_text(p) for p in ANNA_SPLIT_RE.split(stem_raw) if clean_text(p)]
            pieces = [p for p in pieces if not HEX32_RE.fullmatch(p) and not ANNA_NAME_RE.fullmatch(p)]
            year   = ""
            isbn   = ""
            nonmeta: list[str] = []
            for p in pieces:
                isbn_cand = validate_isbn(p)
                if isbn_cand and not isbn:
                    isbn = isbn_cand
                    continue
                if YEAR_RE.fullmatch(p) and not year:
                    year = p
                    continue
                if HEX32_RE.fullmatch(p):
                    continue
                nonmeta.append(clean_filename_text(p))
            title = ""
            authors: list[str] = []
            if nonmeta:
                title, stripped_year = strip_trailing_title_metadata(nonmeta[0])
                if stripped_year and not year:
                    year = stripped_year
            if len(nonmeta) >= 2 and not HEX32_RE.fullmatch(nonmeta[1]):
                authors = split_authors(nonmeta[1].replace("_", "; "))
            out.append(Candidate(
                title=title, authors=authors, year=year,
                doi=dois[0] if dois else "",
                isbn=isbn or (isbns[0] if isbns else ""),
                arxiv=arxivs[0] if arxivs else "",
                source="filename_structured", priority=40,
                notes=["parsed structured Anna/libgen filename"],
            ))
            return out

        # ---- (Series Name) Author - Title (Year) ----
        m = re.match(
            r"^\(([^)]{1,160})\)\s*(.+?)\s+-\s+(.+?)\s*\((\d{4})(?:\s*,[^)]*)?\)?$",
            stem_raw,
        )
        if m:
            author = clean_text(m.group(2)).replace("_", "; ")
            title, stripped_year = strip_trailing_title_metadata(clean_filename_text(m.group(3)))
            out.append(Candidate(
                title=title, authors=split_authors(author),
                year=stripped_year or m.group(4),
                doi=dois[0] if dois else "",
                isbn=isbns[0] if isbns else "",
                arxiv=arxivs[0] if arxivs else "",
                source="filename_series", priority=35,
            ))
            return out

        # ---- Generic: Author - Title ----
        m2 = re.match(r"^(.*?)\s+-\s+(.+)$", stem_raw)
        if m2:
            author = LEADING_SERIES_RE.sub("", clean_text(m2.group(1))).replace("_", "; ")
            title, stripped_year = strip_trailing_title_metadata(clean_filename_text(m2.group(2)))
            year = stripped_year or first_year(stem_raw)
            if author and title and len(title) > 4:
                # Check if the "title" is actually a journal abbreviation
                # (e.g. "Sin CY and White H - J Econometrics - 1996.pdf")
                if is_journal_abbrev_title(title):
                    out.append(Candidate(
                        title="", authors=split_authors(author), year=year,
                        doi=dois[0] if dois else "",
                        isbn=isbns[0] if isbns else "",
                        arxiv=arxivs[0] if arxivs else "",
                        source="filename_author_only", priority=70,
                        notes=["filename title was journal abbreviation, dropped"],
                    ))
                    # Don't return — fall through to title-only fallback
                else:
                    out.append(Candidate(
                        title=title, authors=split_authors(author), year=year,
                        doi=dois[0] if dois else "",
                        isbn=isbns[0] if isbns else "",
                        arxiv=arxivs[0] if arxivs else "",
                        source="filename_author_title", priority=45,
                    ))
                    return out

        # ---- Title-only fallback ----
        title, stripped_year = strip_trailing_title_metadata(stem)
        out.append(Candidate(
            title=title, authors=[],
            year=stripped_year or first_year(stem_raw),
            doi=dois[0] if dois else "",
            isbn=isbns[0] if isbns else "",
            arxiv=arxivs[0] if arxivs else "",
            source="filename_title_only", priority=80,
            notes=([f"jstor stable id: {stable_ids[0]}"] if stable_ids else []),
        ))
        return out

    # ------------------------------------------------------------------
    # Text-header heuristics
    # ------------------------------------------------------------------

    def text_header_candidate(self, text: str) -> list[Candidate]:
        lines = [clean_text(ln) for ln in text.splitlines() if clean_text(ln)]
        kept: list[str] = []
        for line in lines[:80]:
            if any(p.search(line) for p in NOISE_LINE_PATTERNS):
                if line.lower().startswith("author(s):"):
                    kept.append(clean_text(line.split(":", 1)[1]))
                continue
            kept.append(line)
        if not kept:
            return []

        title   = ""
        authors: list[str] = []
        year    = first_year(" ".join(kept[:20]))

        for i, line in enumerate(kept[:20]):
            if len(line) < 6 or len(line) > 220:
                continue
            # Skip lines that are journal headers, not titles
            if is_journal_header_title(line):
                continue
            # Strip "Published as a conference paper at VENUE YEAR: " prefix
            # to recover the actual title that follows it
            conf_m = re.match(r"^published as a conference paper at [^:]+:\s*(.+)$", line, re.I)
            if conf_m:
                line = conf_m.group(1).strip()
                if not line:
                    continue
            if re.fullmatch(r"[A-Z\s\-:;,.]{4,}", line):
                title = line.title()
                author_offset = 1
                if i + 1 < len(kept) and _is_all_caps_phrase(kept[i + 1]):
                    title = title.rstrip(":") + ": " + kept[i + 1].title()
                    author_offset = 2
                if i + author_offset < len(kept):
                    raw_auth = kept[i + author_offset]
                    authors = [strip_footnote_markers(a) for a in split_authors(raw_auth)]
                    authors = [a for a in authors if a]
                break
            if line.count(" ") >= 2 and not YEAR_RE.fullmatch(line) and not line.lower().startswith("proceedings"):
                title = line
                if i + 1 < len(kept) and _is_all_caps_phrase(kept[i + 1]):
                    title = title.rstrip(":") + ": " + kept[i + 1].title()
                    if i + 2 < len(kept):
                        raw_auth = kept[i + 2]
                        if len(raw_auth) < 120:
                            authors = [strip_footnote_markers(a) for a in split_authors(raw_auth)]
                            authors = [a for a in authors if a]
                elif i + 1 < len(kept):
                    maybe = kept[i + 1]
                    if len(maybe) < 120:
                        authors = [strip_footnote_markers(a) for a in split_authors(maybe)]
                        authors = [a for a in authors if a]
                break
        if not title:
            title   = kept[0][:220]
            authors = [strip_footnote_markers(a) for a in split_authors(kept[1])] if len(kept) > 1 else []
            authors = [a for a in authors if a]

        # Final guard: if the title still looks like a journal header, discard it
        if is_journal_header_title(title):
            title = ""

        cand = Candidate(title=title, authors=authors, year=year, source="text_header", priority=90)
        dois, isbns, arxivs, stable_ids = extract_identifiers(text)
        cand.doi   = dois[0]   if dois   else ""
        cand.isbn  = isbns[0]  if isbns  else ""
        cand.arxiv = arxivs[0] if arxivs else ""
        if stable_ids:
            cand.notes.append(f"jstor stable id: {stable_ids[0]}")
        return [cand]

    # ------------------------------------------------------------------
    # GROBID (optional — only if --grobid-url is provided)
    # ------------------------------------------------------------------

    def grobid_candidate(self, path: Path) -> list[Candidate]:
        if not self.args.grobid_url:
            return []
        base      = self.args.grobid_url.rstrip("/")
        cache_key = f"{path.resolve()}::{path.stat().st_mtime_ns}"
        tei = self.http.post_multipart(
            f"{base}/api/processHeaderDocument",
            fields={"consolidateHeader": "1"},
            file_field="input",
            file_path=path,
            namespace="grobid",
            cache_key=cache_key,
        )
        if not tei.strip():
            return []
        try:
            root = ET.fromstring(tei)
        except Exception:
            return []
        ns    = {"tei": "http://www.tei-c.org/ns/1.0"}
        title = clean_text(" ".join((root.findtext(".//tei:titleStmt/tei:title", default="", namespaces=ns) or "").split()))
        authors: list[str] = []
        for auth in root.findall(".//tei:sourceDesc//tei:author", ns):
            forename = clean_text(auth.findtext(".//tei:forename", default="", namespaces=ns) or "")
            surname  = clean_text(auth.findtext(".//tei:surname",  default="", namespaces=ns) or "")
            full     = clean_text(" ".join(x for x in (forename, surname) if x))
            if full:
                authors.append(full)
        blob = clean_text(" ".join(root.itertext()))
        dois, isbns, arxivs, _ = extract_identifiers(blob)
        # Reject GROBID outputs that look like affiliations, abstract openers,
        # publisher metadata (SPIN ...), or are simply too long to be a title.
        # GROBID is trained on journal-article headers and is prone to picking
        # up Paris-Saclay affiliation strings, Springer project numbers, or
        # the first sentence of an abstract when used on books.
        if title and is_suspicious_title(title):
            # Keep the authors/year/identifiers but drop the bad title so
            # other candidate sources (filename, pdfinfo, LLM) can supply
            # the title instead.
            title = ""
        if not (title or authors or dois or isbns or arxivs):
            return []
        return [Candidate(
            title=title, authors=authors, year=first_year(blob),
            doi=dois[0] if dois else "",
            isbn=isbns[0] if isbns else "",
            arxiv=arxivs[0] if arxivs else "",
            source="grobid", priority=12,
        )]

    # ------------------------------------------------------------------
    # LLM fallback — supports any OpenAI-compatible endpoint
    #
    # Configuration (via CLI args):
    #   --llm-endpoint  Base URL, e.g. https://api.groq.com/openai/v1
    #   --llm-api-key   Bearer token (leave empty for local Ollama)
    #   --llm-model     Model name,  e.g. llama-3.1-8b-instant
    #   --llm-chars     Characters of PDF text to include (default 4000)
    #
    # Free options that require no local download:
    #   Groq   — https://console.groq.com  (generous free tier, very fast)
    #   OpenRouter — https://openrouter.ai (has several free models)
    # ------------------------------------------------------------------

    def llm_candidate(
        self,
        path: Path,
        text: str,
        filename_cand: Candidate | None,
    ) -> list[Candidate]:
        self.last_llm_debug = {
            "text_llm_used": "no",
            "text_llm_status": "not_attempted",
            "text_llm_error": "",
        }
        endpoint = getattr(self.args, "llm_endpoint", "").strip()
        model    = getattr(self.args, "llm_model",    "").strip()
        api_key  = getattr(self.args, "llm_api_key",  "").strip()
        chars    = int(getattr(self.args, "llm_chars", 4000))

        # Legacy: --ollama-model maps to Ollama's OpenAI-compat endpoint
        ollama_model = getattr(self.args, "ollama_model", "").strip()
        if ollama_model and not model:
            endpoint = "http://localhost:11434/v1"
            model    = ollama_model
            chars    = int(getattr(self.args, "ollama_chars", chars))

        if not endpoint or not model:
            return []
        self.last_llm_debug.update({
            "text_llm_used": "yes",
            "text_llm_status": "requesting",
            "text_llm_model": model,
            "text_llm_chars": str(chars),
        })

        snippet = text[:chars]
        prompt_data = {
            "task": (
                "Extract bibliographic metadata from the PDF text and filename below. "
                "Return only what you can directly observe — do not guess or hallucinate."
            ),
            "rules": [
                "Return empty strings for unknown fields.",
                "Strip 'Author(s):' prefixes.",
                "Ignore download watermarks, page numbers, hashes, and JSTOR stable IDs.",
                "Do not include publisher/series names in the title unless they are part of the real title.",
                "authors must be a JSON array of individual name strings.",
            ],
            "filename": path.name,
            "filename_guess": dataclasses.asdict(filename_cand) if filename_cand else {},
            "text": snippet,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a bibliographic metadata extractor. "
                    "Respond ONLY with a valid JSON object matching the schema: "
                    '{"title": string, "authors": [string], "year": string, '
                    '"doi": string, "isbn": string, "arxiv": string, "notes": string}.'
                ),
            },
            {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=False)},
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }
        cache_key = f"{path.resolve()}::{path.stat().st_mtime_ns}::{endpoint}::{model}::{chars}"

        extra_headers: dict[str, str] = {}
        if api_key:
            extra_headers["Authorization"] = f"Bearer {api_key}"

        url = endpoint.rstrip("/") + "/chat/completions"
        resp = self.http.post_json(
            url, payload, "llm", cache_key,
            extra_headers=extra_headers,
            bucket="llm", min_interval=0.5,
        )
        http_trace = self.http.take_last_request_trace("llm")
        if http_trace:
            self.last_llm_debug["text_llm_http_json"] = json.dumps(
                http_trace,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            self.last_llm_debug["text_llm_cache_hit"] = "yes" if http_trace.get("cache_hit") else "no"
            self.last_llm_debug["text_llm_attempts"] = str(http_trace.get("attempt_count", ""))
            self.last_llm_debug["text_llm_http_s"] = f"{float(http_trace.get('elapsed_s') or 0.0):.6f}"
            self.last_llm_debug["text_llm_retry_sleep_s"] = f"{float(http_trace.get('retry_sleep_s') or 0.0):.6f}"
            self.last_llm_debug["text_llm_pacing_s"] = f"{float(http_trace.get('pacing_sleep_s') or 0.0):.6f}"
            self.last_llm_debug["text_llm_status"] = str(http_trace.get("final_status", "")) or "unknown"
            self.last_llm_debug["text_llm_error"] = str(http_trace.get("final_error", ""))
        rem = self.http.remaining_tokens("llm")
        if rem is not None:
            self.last_llm_debug["text_llm_tokens_remaining"] = str(rem)

        # Handle both OpenAI-style and Ollama-style response shapes
        try:
            usage = resp.get("usage") if isinstance(resp, dict) else None
            if isinstance(usage, dict):
                for src, dst in (
                    ("prompt_tokens", "text_llm_prompt_tokens"),
                    ("completion_tokens", "text_llm_completion_tokens"),
                    ("total_tokens", "text_llm_total_tokens"),
                    ("total_time", "text_llm_provider_total_s"),
                    ("queue_time", "text_llm_provider_queue_s"),
                ):
                    if src in usage and usage[src] is not None:
                        self.last_llm_debug[dst] = str(usage[src])
            if isinstance(resp, dict) and "choices" in resp:
                content = resp["choices"][0]["message"]["content"]
            elif isinstance(resp, dict) and "message" in resp:          # Ollama native
                content = resp["message"]["content"]
            else:
                self.last_llm_debug["text_llm_status"] = "bad_response_shape"
                return []
            data = json.loads(content) if isinstance(content, str) else content
        except Exception as exc:
            self.last_llm_debug["text_llm_status"] = "parse_error"
            self.last_llm_debug["text_llm_error"] = clean_text(str(exc))
            return []

        title   = clean_text(_coerce_str(data.get("title")))
        authors = [clean_text(a) for a in (data.get("authors") or []) if clean_text(str(a))]
        year    = clean_text(_coerce_str(data.get("year")))
        doi     = clean_text(_coerce_str(data.get("doi")))
        isbn    = validate_isbn(_coerce_str(data.get("isbn")))
        arxiv   = clean_text(_coerce_str(data.get("arxiv")))
        notes_s = clean_text(_coerce_str(data.get("notes")))
        notes   = [notes_s] if notes_s else []

        if not title and not authors:
            self.last_llm_debug["text_llm_status"] = "empty_result"
            return []
        self.last_llm_debug["text_llm_status"] = "ok"
        self.last_llm_debug["text_llm_title"] = title
        self.last_llm_debug["text_llm_authors"] = "; ".join(authors)
        self.last_llm_debug["text_llm_year"] = year
        self.last_llm_debug["text_llm_doi"] = doi
        self.last_llm_debug["text_llm_isbn"] = isbn
        self.last_llm_debug["text_llm_arxiv"] = arxiv
        return [Candidate(
            title=title, authors=authors, year=year,
            doi=doi, isbn=isbn, arxiv=arxiv,
            source=f"llm:{model}", priority=30, notes=notes,
        )]

    # ------------------------------------------------------------------
    # Vision LLM — renders the first few pages of the PDF as images and
    # asks a multimodal model to extract bibliographic metadata.
    #
    # Motivation: pdftotext + GROBID + text-LLM all fail on:
    #   - Scanned PDFs with no text layer (BOOK.pdf-class garbage).
    #   - PDFs with useless filenames where the text extraction works but
    #     doesn't yield a searchable title (ass.pdf → van der Vaart).
    #   - Image-heavy title pages where text extraction mangles author
    #     names or title layout.
    # The vision model reads the pages as images, so OCR quality and
    # text-layer corruption are irrelevant.  Cover pages, title pages, and
    # copyright pages (typically pages 1–3 of a book) contain everything
    # we need — title, authors, publisher, ISBN, sometimes DOI.
    #
    # Configuration (via CLI args):
    #   --vision-llm-endpoint  Base URL (Groq / OpenAI / Ollama / OpenRouter)
    #   --vision-llm-api-key   Bearer token (empty for local Ollama)
    #   --vision-llm-model     Model name, e.g. meta-llama/llama-4-scout-17b-16e-instruct
    #   --vision-pages         How many leading pages to render (default: 3)
    #   --vision-dpi           DPI for the rendered images (default: 120)
    #
    # The rendered images are sent as base64 data URLs in OpenAI format.
    # Groq's Llama 4 Scout/Maverick accept up to 5 images per message;
    # defaulting to 3 pages keeps cost low while covering cover +
    # title + copyright pages on typical books.
    # ------------------------------------------------------------------

    def _render_pages_to_b64(
        self,
        path: Path,
        pages: int,
        dpi: int,
        debug: dict[str, str],
    ) -> list[str]:
        """Render the first *pages* pages of *path* to JPEG and return as base64.

        Returns an empty list on any error and sets debug['vision_status']
        to an error code.  The caller should return early when this is empty.
        """
        import base64
        import tempfile

        debug["vision_status"] = "rendering"
        try:
            tmpdir_ctx = tempfile.TemporaryDirectory(prefix="papis_import_vis_")
        except Exception as exc:
            debug["vision_status"] = "tempdir_error"
            debug["vision_error"] = clean_text(str(exc))
            return []

        image_b64s: list[str] = []
        try:
            with tmpdir_ctx as td:
                prefix = Path(td) / "page"
                try:
                    cp = subprocess.run(
                        ["pdftoppm", "-jpeg", "-r", str(dpi),
                         "-f", "1", "-l", str(pages),
                         str(path), str(prefix)],
                        check=False, capture_output=True, text=True, timeout=60,
                    )
                except FileNotFoundError:
                    debug["vision_status"] = "pdftoppm_missing"
                    return []
                except subprocess.TimeoutExpired:
                    debug["vision_status"] = "render_timeout"
                    return []
                if cp.returncode != 0:
                    debug["vision_status"] = "render_failed"
                    stderr = clean_text(cp.stderr or "")
                    if stderr:
                        debug["vision_error"] = stderr
                    return []
                # pdftoppm produces page-1.jpg, page-01.jpg, etc. depending on
                # zero-padding.  Glob handles both naming schemes.
                imgs = sorted(Path(td).glob("page-*.jpg"))[:pages]
                if not imgs:
                    debug["vision_status"] = "no_rendered_images"
                    return []
                for img in imgs:
                    try:
                        image_b64s.append(
                            base64.b64encode(img.read_bytes()).decode("ascii")
                        )
                    except Exception:
                        continue
        except Exception as exc:
            debug["vision_status"] = "render_exception"
            debug["vision_error"] = clean_text(str(exc))
            return []

        if not image_b64s:
            debug["vision_status"] = "no_image_bytes"
        return image_b64s

    def vision_llm_candidate(
        self,
        path: Path,
        filename_cand: Candidate | None,
        is_book: bool = False,
    ) -> tuple[list[Candidate], dict[str, str]]:
        endpoint = getattr(self.args, "vision_llm_endpoint", "").strip()
        model    = getattr(self.args, "vision_llm_model",    "").strip()
        api_key  = getattr(self.args, "vision_llm_api_key",  "").strip()
        pages    = max(1, int(getattr(self.args, "vision_pages", 4)))
        dpi      = max(72, int(getattr(self.args, "vision_dpi",   120)))

        # Fall back to text-LLM credentials when vision-specific ones
        # aren't configured.
        if not endpoint:
            endpoint = getattr(self.args, "llm_endpoint", "").strip()
        if not api_key:
            api_key = getattr(self.args, "llm_api_key", "").strip()

        debug: dict[str, str] = {
            "vision_status": "not_configured",
            "vision_model": model,
            "vision_pages": str(pages),
            "vision_dpi": str(dpi),
            "vision_escalated": "no",
        }

        if not endpoint or not model:
            return [], debug

        # Two-tier strategy:
        # - Books go straight to the full page count so that a title on page
        #   4-5 (after cover, half-title, and series page) is still visible.
        # - Non-books (papers, articles) try 1 page at 100 DPI first — the
        #   title is almost always on page 1 and this is 3-4× cheaper on
        #   tokens.  Only escalate to the full count if page 1 returned
        #   nothing useful (no title, no authors, no identifiers).
        if is_book:
            tiers = [(pages, dpi)]
        else:
            cheap_dpi = min(100, dpi)
            tiers = [(1, cheap_dpi), (pages, dpi)] if pages > 1 else [(pages, dpi)]

        url = endpoint.rstrip("/") + "/chat/completions"
        extra_headers: dict[str, str] = {}
        if api_key:
            extra_headers["Authorization"] = f"Bearer {api_key}"
        filename_guess = dataclasses.asdict(filename_cand) if filename_cand else {}

        for tier_idx, (tier_pages, tier_dpi) in enumerate(tiers):
            is_last_tier = tier_idx == len(tiers) - 1

            image_b64s = self._render_pages_to_b64(path, tier_pages, tier_dpi, debug)
            if not image_b64s:
                return [], debug

            debug["vision_pages"]  = str(tier_pages)
            debug["vision_dpi"]    = str(tier_dpi)
            debug["vision_images"] = str(len(image_b64s))
            debug["vision_status"] = "requesting"

            user_text = (
                "Extract bibliographic metadata from these PDF pages (typically "
                "the cover, title page, and copyright page of a book or paper). "
                "Return ONLY what you can directly read — do not guess, infer, "
                "or hallucinate. "
                f"Filename (for context only, may be unhelpful): {path.name}. "
                f"Filename-based guess (may be wrong): "
                f"{json.dumps(filename_guess, ensure_ascii=False)}"
            )
            user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            for b64 in image_b64s:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a bibliographic metadata extractor for an "
                        "academic reference manager. Respond ONLY with a valid "
                        "JSON object matching the schema: "
                        '{"title": string, "authors": [string], "year": string, '
                        '"doi": string, "isbn": string, "arxiv": string, '
                        '"publisher": string, "notes": string}. '
                        "Return empty strings for fields you cannot read directly. "
                        "The title is the main title of the work, not the series "
                        "or imprint name. Authors should be individual people "
                        "listed on the title page, not editors (unless there is "
                        "no author). ISBNs and DOIs typically appear on the "
                        "copyright page."
                    ),
                },
                {"role": "user", "content": user_content},
            ]
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": 600,
                "response_format": {"type": "json_object"},
            }
            cache_key = (
                f"{path.resolve()}::{path.stat().st_mtime_ns}::"
                f"{endpoint}::{model}::vision::p{tier_pages}::r{tier_dpi}"
            )

            try:
                resp = self.http.post_json(
                    url, payload, "vision_llm", cache_key,
                    extra_headers=extra_headers,
                    bucket="vision_llm", min_interval=0.5,
                    max_retry_wait=60.0,
                    min_remaining_tokens=10_000,
                )
                http_trace = self.http.take_last_request_trace("vision_llm")
                if http_trace:
                    debug["vision_http_json"] = json.dumps(
                        http_trace,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    debug["vision_cache_hit"] = "yes" if http_trace.get("cache_hit") else "no"
                    debug["vision_attempts"] = str(http_trace.get("attempt_count", ""))
                    debug["vision_http_s"] = f"{float(http_trace.get('elapsed_s') or 0.0):.6f}"
                    debug["vision_retry_sleep_s"] = f"{float(http_trace.get('retry_sleep_s') or 0.0):.6f}"
            except Exception as exc:
                debug["vision_status"] = "request_error"
                debug["vision_error"] = clean_text(str(exc))
                return [], debug

            try:
                usage = resp.get("usage") if isinstance(resp, dict) else None
                if isinstance(usage, dict):
                    for src, dst in (
                        ("prompt_tokens", "vision_prompt_tokens"),
                        ("completion_tokens", "vision_completion_tokens"),
                        ("total_tokens", "vision_total_tokens"),
                        ("total_time", "vision_provider_total_s"),
                        ("queue_time", "vision_provider_queue_s"),
                    ):
                        if src in usage and usage[src] is not None:
                            debug[dst] = str(usage[src])
                if isinstance(resp, dict) and "choices" in resp:
                    content = resp["choices"][0]["message"]["content"]
                elif isinstance(resp, dict) and "message" in resp:
                    content = resp["message"]["content"]
                else:
                    debug["vision_status"] = "bad_response_shape"
                    return [], debug
                data = json.loads(content) if isinstance(content, str) else content
            except Exception as exc:
                debug["vision_status"] = "parse_error"
                debug["vision_error"] = clean_text(str(exc))
                return [], debug

            title   = clean_text(_coerce_str(data.get("title")))
            authors = [clean_text(a) for a in (data.get("authors") or []) if clean_text(str(a))]
            year    = clean_text(_coerce_str(data.get("year")))
            doi     = clean_text(_coerce_str(data.get("doi")))
            isbn    = validate_isbn(_coerce_str(data.get("isbn")))
            arxiv   = clean_text(_coerce_str(data.get("arxiv")))
            notes_s = clean_text(_coerce_str(data.get("notes")))

            if not title and not authors and not (doi or isbn or arxiv):
                if not is_last_tier:
                    debug["vision_escalated"] = "yes"
                    continue  # try next tier with more pages
                debug["vision_status"] = "empty_result"
                return [], debug

            notes = [f"vision-extracted from {len(image_b64s)} page(s)"]
            if notes_s:
                notes.append(notes_s)

            debug["vision_status"] = "ok"
            debug["vision_title"]   = title
            debug["vision_authors"] = "; ".join(authors)
            debug["vision_year"]    = year
            debug["vision_doi"]     = doi
            debug["vision_isbn"]    = isbn
            debug["vision_arxiv"]   = arxiv
            return [Candidate(
                title=title, authors=authors, year=year,
                doi=doi, isbn=isbn, arxiv=arxiv,
                source=f"vision_llm:{model}",
                priority=25,
                notes=notes,
            )], debug

        # Unreachable: the loop always returns or continues until the last tier.
        debug["vision_status"] = "empty_result"
        return [], debug
