"""Identifier parsing and validation helpers."""
from __future__ import annotations

import re

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.I)

# Exclude JSTOR stable/* paths from being matched as arXiv IDs.
ARXIV_RE = re.compile(
    r"(?<!stable/)(?<![A-Za-z0-9])"
    r"(?:arXiv\s*:?\s*)?"
    r"((?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)\b",
    re.I,
)
ISBN_CANDIDATE_RE = re.compile(r"(?<!\d)(?:97[89][\-\s]?)?\d[\d\-\s]{8,20}[\dXx](?!\d)")
JSTOR_STABLE_RE = re.compile(r"\bstable/\d+\b", re.I)

# ScienceDirect download format: 1-s2.0-S<rawPII>-*.pdf
# Raw PII: S + 4 digits + 4 chars (ISSN, may end in X) + 2 year + 5 seq + 1 check
ELSEVIER_SCIDIR_RE = re.compile(
    r"(?:^|[\\/\s])1-s2\.0-(S[\dA-Z]{14,18})(?:-\w+)?(?:\.pdf)?",
    re.I,
)

# Formatted PII already contains hyphens and parentheses: S0049-237X(08)71107-8
ELSEVIER_PII_FMT_RE = re.compile(
    r"\b(S\d{4}-\d{3}[\dX]\(\d{2}\)\d{5}-\d)\b",
    re.I,
)


def pii_to_doi(pii: str) -> str:
    """Convert an Elsevier PII to a DOI string, or '' if format is unrecognised."""
    pii = pii.upper().strip()
    if re.fullmatch(r"S\d{4}-\d{3}[\dX]\(\d{2}\)\d{5}-\d", pii):
        return f"10.1016/{pii}"
    m = re.fullmatch(r"S(\d{4})([\dX]{4})(\d{2})(\d{5})(\d)", pii)
    if m:
        issn = f"{m.group(1)}-{m.group(2)}"
        return f"10.1016/S{issn}({m.group(3)}){m.group(4)}-{m.group(5)}"
    m2 = re.fullmatch(r"(\d{4})([\dX]{4})(\d{2})(\d{5})(\d)", pii)
    if m2:
        issn = f"{m2.group(1)}-{m2.group(2)}"
        return f"10.1016/{issn}({m2.group(3)}){m2.group(4)}-{m2.group(5)}"
    return ""


def jstor_filename_doi(stem: str) -> str:
    """Return a JSTOR DOI for pure numeric stable-id filename stems."""
    if re.fullmatch(r"\d{7,10}", stem):
        return f"10.2307/{stem}"
    return ""


def numeric_filename_dois(stem: str) -> list[str]:
    """Return plausible DOI candidates for pure numeric filename stems."""
    if not re.fullmatch(r"\d{7,10}", stem):
        return []
    return [
        f"10.2307/{stem}",
        f"10.1214/aop/{stem}",
        f"10.1214/aos/{stem}",
        f"10.1214/aoms/{stem}",
        f"10.1214/aoap/{stem}",
        f"10.1214/ss/{stem}",
        f"10.1214/lnms/{stem}",
        f"10.1214/{stem}",
    ]


def validate_isbn(raw: str) -> str:
    """Return a normalised ISBN-10 or ISBN-13 string, or '' if invalid."""
    s = re.sub(r"[^0-9Xx]", "", raw)
    if len(s) == 10:
        total = 0
        for i, ch in enumerate(s[:9], start=1):
            if not ch.isdigit():
                return ""
            total += i * int(ch)
        check = 10 if s[9] in "Xx" else (int(s[9]) if s[9].isdigit() else -1)
        if check < 0:
            return ""
        total += 10 * check
        return s.upper() if total % 11 == 0 else ""
    if len(s) == 13 and s.isdigit():
        total = 0
        for i, ch in enumerate(s[:12]):
            total += int(ch) * (1 if i % 2 == 0 else 3)
        check = (10 - (total % 10)) % 10
        return s if check == int(s[12]) else ""
    return ""


def extract_identifiers(
    *chunks: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (dois, isbns, arxivs, jstor_stable_ids) found across all chunks."""
    dois: list[str] = []
    isbns: list[str] = []
    arxivs: list[str] = []
    stable_ids: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        for m in DOI_RE.finditer(chunk):
            doi = m.group(1).rstrip(").,;:")
            if doi not in dois:
                dois.append(doi)
        for m in ISBN_CANDIDATE_RE.finditer(chunk):
            isbn = validate_isbn(m.group(0))
            if isbn and isbn not in isbns:
                isbns.append(isbn)
        for m in JSTOR_STABLE_RE.finditer(chunk):
            sid = m.group(0)
            if sid not in stable_ids:
                stable_ids.append(sid)
        for m in ARXIV_RE.finditer(chunk):
            arx = m.group(1)
            if arx and not arx.lower().startswith("stable/") and arx not in arxivs:
                arxivs.append(arx)
        for m in ELSEVIER_SCIDIR_RE.finditer(chunk):
            doi = pii_to_doi(m.group(1))
            if doi and doi not in dois:
                dois.append(doi)
        for m in re.finditer(
            r"(?:^|[\\/\s])1-s2\.0-(\d[\dA-X]{15})(?:-\w+)?(?:\.pdf)?",
            chunk,
            re.I,
        ):
            doi = pii_to_doi(m.group(1))
            if doi and doi not in dois:
                dois.append(doi)
        for m in ELSEVIER_PII_FMT_RE.finditer(chunk):
            doi = pii_to_doi(m.group(1))
            if doi and doi not in dois:
                dois.append(doi)
    return dois, isbns, arxivs, stable_ids

