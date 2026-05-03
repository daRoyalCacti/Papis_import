"""Text normalization, title cleanup, and author parsing helpers."""
from __future__ import annotations

import re

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
MULTISPACE_RE = re.compile(r"\s+")
AUTHOR_ROLE_MARKER_RE = re.compile(
    r"\s*[\[(]\s*(?:"
    r"auth\.?|author|authors|"
    r"ed\.?|eds\.?|editor|editors"
    r")\s*[\])]\s*$",
    re.I,
)


def strip_footnote_markers(s: str) -> str:
    """Remove footnote/affiliation markers from author strings."""
    s = re.sub(r"[∗†‡§¶✝✦⋆]", "", s)
    s = s.replace("*", "")
    s = re.sub(r"(?<=[a-zA-Z])\d+", "", s)
    s = re.sub(r"\s+\d+(?=\s|$)", " ", s)
    return s.strip()


def clean_author_name(s: str) -> str:
    """Normalize a single author name without removing meaningful names."""
    s = clean_text(s)
    s = s.replace("Author(s):", "").replace("author(s):", "")
    prev = ""
    while s and s != prev:
        prev = s
        s = AUTHOR_ROLE_MARKER_RE.sub("", s)
    return clean_text(s).strip(" ,;")


def clean_author_list(authors: list[str]) -> list[str]:
    """Clean author names and drop empty entries, preserving order."""
    out: list[str] = []
    for author in authors:
        cleaned = clean_author_name(author)
        if cleaned:
            out.append(cleaned)
    return out


def repair_ligature_splits(text: str) -> str:
    """Fix PDF text where ligature artifacts inject spaces into words."""
    pattern = re.compile(r"\b([B-HJ-Z])\s+([A-Z][a-z]{2,})\b")
    matches = pattern.findall(text)
    if len(matches) < 3:
        return text
    return pattern.sub(lambda m: m.group(1) + m.group(2).lower(), text)


def repair_title_ligatures(title: str) -> str:
    """Repair ligature splits in short candidate titles."""
    pattern = re.compile(r"\b([B-HJ-Z])\s+([A-Z][a-z]{1,})\b")
    matches = pattern.findall(title)
    if len(matches) >= 2:
        title = pattern.sub(lambda m: m.group(1) + m.group(2).lower(), title)
    if len(matches) >= 2:
        frag_pattern = re.compile(r"\b([A-Za-z][a-z]{1,4})\s+([A-Z][a-z]{2,})\b")
        common = frozenset({
            "the", "and", "for", "with", "from", "that", "this",
            "are", "was", "were", "has", "have", "had", "but",
            "not", "can", "its", "our", "their", "via", "per",
            "on", "at", "in", "of", "to", "by", "an", "or",
            "as", "is", "it", "be", "do", "so", "no", "if",
            "edge", "rule", "loss", "deep", "data", "step",
            "new", "non", "all", "one", "two", "how", "why",
        })

        def _rejoin(m: re.Match) -> str:
            left = m.group(1)
            if left.lower() in common:
                return m.group(0)
            return left + m.group(2).lower()

        prev = title
        for _ in range(5):
            title = frag_pattern.sub(_rejoin, title)
            if title == prev:
                break
            prev = title
    return title


def clean_text(s: str) -> str:
    s = CONTROL_RE.sub(" ", s)
    s = s.replace("\u00a0", " ")
    s = MULTISPACE_RE.sub(" ", s)
    return s.strip()


def decamelize(s: str) -> str:
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    s = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)
    s = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s)
    return s


def clean_filename_text(s: str) -> str:
    s = clean_text(s)
    s = decamelize(s)
    s = s.replace("_", " ")
    s = re.sub(r"^\[[^\]]{1,200}\]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    if s.count("-") >= 4 and " " not in s:
        s = s.replace("-", " ")
    s = re.sub(r"\b(?:paper|main|supplementary material|supplement|accepted|final draft)\b$", "", s, flags=re.I)
    s = re.sub(r"^(?:nips|neurips|icml|aistats|jmlr|uai|aaai|cvpr|iclr)[-_ ]+\d{4}[-_ ]+", "", s, flags=re.I)
    s = re.sub(r"^\d{4}[_ -]+book[_ -]+", "", s, flags=re.I)
    s = re.sub(r"\s*\((\d+)\)$", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -_.")
    return clean_text(s)


def strip_trailing_title_metadata(title: str) -> tuple[str, str]:
    """Remove trailing publisher/year junk like '(2020, Chapman & Hall...)'."""
    raw = clean_text(title)
    year = first_year(raw)
    trimmed = re.sub(r"\s*\((\d{4})(?:\s*,[^)]*)?\)?$", "", raw)
    if trimmed != raw:
        return clean_text(trimmed).strip(" -_.,;:"), year
    m = re.match(r"^(.*?)(?:\s*[\[(](\d{4})(?:\s*,.*)?)$", raw)
    if m:
        return clean_text(m.group(1)).strip(" -_.,;:"), m.group(2)
    return raw, year


def normalize_title(s: str) -> str:
    s = clean_text(s).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\b(edition|ed\.?|vol\.?|volume)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def normalize_author_token(s: str) -> str:
    s = clean_author_name(s).lower()
    s = s.replace("author(s):", "")
    s = re.sub(r"[^a-z\s,.-]", " ", s)
    s = MULTISPACE_RE.sub(" ", s).strip(" ,.-")
    return s


def split_authors(text: str) -> list[str]:
    text = clean_text(text)
    text = text.replace("Author(s):", "").replace("author(s):", "")
    text = text.replace("_", " ")
    if not text:
        return []
    if ";" in text:
        parts = [clean_text(p) for p in text.split(";")]
    elif " and " in text.lower():
        parts = [clean_text(p) for p in re.split(r"\band\b", text, flags=re.I)]
    elif " · " in text:
        parts = [clean_text(p) for p in text.split(" · ")]
    elif text.count(",") >= 3:
        raw = [clean_text(p) for p in text.split(",") if clean_text(p)]
        parts = []
        i = 0
        while i < len(raw):
            if i + 1 < len(raw):
                parts.append(clean_text(raw[i + 1] + " " + raw[i]))
                i += 2
            else:
                parts.append(raw[i])
                i += 1
    else:
        parts = [clean_text(p) for p in text.split(",") if len(text.split(",")) <= 4]
        if len(parts) <= 1:
            return [text]
    return clean_author_list([p for p in parts if p])


def first_year(*chunks: str) -> str:
    for chunk in chunks:
        if not chunk:
            continue
        m = YEAR_RE.search(chunk)
        if m:
            return m.group(1)
    return ""
