"""Title and author matching helpers."""
from __future__ import annotations

import difflib

from papis_import.core.text import normalize_author_token, normalize_title


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    sa, sb = set(na.split()), set(nb.split())
    jaccard = len(sa & sb) / max(1, len(sa | sb))
    return max(ratio, jaccard)


def author_overlap(a: list[str], b: list[str]) -> float:
    def surnames(names: list[str]) -> set[str]:
        out: set[str] = set()
        for x in names:
            norm = normalize_author_token(x)
            if not norm:
                continue
            surname = norm.split(",", 1)[0].strip() if "," in norm else (norm.split()[-1] if norm.split() else norm)
            if surname:
                out.add(surname)
        return out

    sa, sb = surnames(list(a)), surnames(list(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, min(len(sa), len(sb)))

