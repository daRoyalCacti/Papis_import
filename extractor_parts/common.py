from __future__ import annotations


def coerce_str(val: object) -> str:
    """Coerce an LLM output field to a plain string."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x)
    return str(val)
