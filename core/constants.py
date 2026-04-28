"""Shared constants for external services and defaults."""
from __future__ import annotations

import os

CROSSREF_BASE = "https://api.crossref.org"
OPENLIBRARY_BOOKS_URL = "https://openlibrary.org/api/books"
OPENLIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1"
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
OPENALEX_BASE = "https://api.openalex.org"  # free, no key needed
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

USER_AGENT = "papis-import/2026.04 (https://github.com/)"

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/papis_import")

