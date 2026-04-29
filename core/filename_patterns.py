"""Compiled regexes used by filename-based extraction."""
from __future__ import annotations

import re

HEX32_RE = re.compile(r"^[0-9a-f]{32}$", re.I)
LEADING_SERIES_RE = re.compile(r"^[\[(].{0,160}?[\])]\s*")
ANNA_SPLIT_RE = re.compile(r"\s+--\s+")
ANNA_NAME_RE = re.compile(r"^Anna['']?s Archive(?:-\d+)?$", re.I)

