from __future__ import annotations

from papis_import.models import Metadata
from papis_import.utils import MIN_SANITY_SCORE, sanity_score_for_match


def apply_sanity(meta: Metadata, text: str, filename: str = "") -> Metadata:
    """Attach sanity score/pass fields without changing confidence policy."""
    score = sanity_score_for_match(
        meta.title, meta.authors, meta.source, text, filename=filename
    )
    meta.sanity_score = round(score, 3)
    meta.sanity_passed = meta.sanity_score >= MIN_SANITY_SCORE
    return meta
