"""Embedded PDF metadata and pdfinfo extraction."""
from __future__ import annotations

from pathlib import Path

from papis_import.core.identifiers import extract_identifiers
from papis_import.core.process import read_cmd
from papis_import.core.text import clean_text, first_year, split_authors, strip_footnote_markers
from papis_import.core.title_quality import is_garbage_pdfinfo_author, is_garbage_pdfinfo_title
from papis_import.models import Candidate

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore


class PdfMetadataExtractor:
    """Extract embedded metadata from PDF metadata streams and pdfinfo."""

    def embedded_metadata(self, path: Path) -> list[Candidate]:
        if PdfReader is None:
            return []
        out: list[Candidate] = []
        try:
            reader = PdfReader(str(path))
        except Exception:
            return []

        try:
            meta = reader.metadata or {}
        except Exception:
            meta = {}
        title = clean_text(str(meta.get("/Title", "")))
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
            cand.doi = dois[0] if dois else ""
            cand.isbn = isbns[0] if isbns else ""
            cand.arxiv = arxivs[0] if arxivs else ""
            out.append(cand)

        try:
            xmp = reader.xmp_metadata
        except Exception:
            xmp = None
        if xmp is not None:
            xmp_title = ""
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
                cand.doi = dois[0] if dois else ""
                cand.isbn = isbns[0] if isbns else ""
                cand.arxiv = arxivs[0] if arxivs else ""
                out.append(cand)
        return out

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
        title = fields.get("title", "")
        author = fields.get("author", "")
        if is_garbage_pdfinfo_title(title):
            title = ""
        if is_garbage_pdfinfo_author(author):
            author = ""
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
        cand.doi = dois[0] if dois else ""
        cand.isbn = isbns[0] if isbns else ""
        cand.arxiv = arxivs[0] if arxivs else ""
        return [cand]

