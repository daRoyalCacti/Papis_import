"""Canonical output schema definitions for all result, profile, and debug files."""
from __future__ import annotations

RESULT_COLUMNS: list[str] = [
    "File Path", "Tags", "Source", "Confidence", "Verified",
    "Title", "Authors", "Year", "DOI", "ISBN", "arXiv",
    "Sanity Passed", "Sanity Score", "Auto Safe", "Needs OCR",
    "Notes", "Imported", "Error", "Suggested Command",
    "Vision Used", "Vision Trigger", "Vision Status", "Vision Error",
    "Final Source",
    "Soft Auto", "Soft Auto Reasons",
]

PROFILE_COLUMNS: list[str] = [
    "File Path", "Status", "Error", "Final Source", "Confidence", "Verified",
    "File Wall s", "Resolve Total s",
    "Text Extract s", "Embedded Metadata s", "GROBID s",
    "Text LLM s", "Text LLM HTTP s", "Text LLM Retry Sleep s",
    "Text LLM Pacing s", "Text LLM Provider Total s",
    "Text LLM Provider Queue s", "Text LLM Attempts", "Text LLM Cache Hit",
    "Vision LLM s", "Vision HTTP s", "Vision Retry Sleep s",
    "Vision Provider Total s", "Vision Provider Queue s",
    "Vision Attempts", "Vision Cache Hit",
    "Vision Pacing s", "Identifier Lookups s",
    "Title Search s", "OCR Retry s", "OCRMyPDF s", "OCR Reresolve s",
    "Best Local s", "Header Candidate s",
    "Identifier Lookup Count", "Identifier Lookups JSON",
    "Crossref Search s", "Crossref Search Candidates Tried",
    "Crossref Search Matches", "Crossref Search Errors",
    "OpenAlex Search s", "OpenAlex Search Candidates Tried",
    "OpenAlex Search Matches", "OpenAlex Search Errors",
    "Semantic Scholar Search s", "Semantic Scholar Search Candidates Tried",
    "Semantic Scholar Search Matches", "Semantic Scholar Search Errors",
    "OpenLibrary Search s", "OpenLibrary Search Candidates Tried",
    "OpenLibrary Search Matches", "OpenLibrary Search Errors",
    "Google Books Search s", "Google Books Search Candidates Tried",
    "Google Books Search Matches", "Google Books Search Errors",
    "Title Searches JSON",
]

# Result rows go to TSV files; debug rows go to a JSONL file.
TSV_FILES: dict[str, str] = {
    "auto":    "papis_import.tsv",
    "review":  "papis_import_review.tsv",
    "soft":    "papis_import_soft.tsv",
    "profile": "papis_import_profile.tsv",
}

JSONL_FILES: dict[str, str] = {
    "debug": "papis_import_debug.jsonl",
}
