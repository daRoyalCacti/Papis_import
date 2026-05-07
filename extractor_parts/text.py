"""PDF text extraction helpers."""
from __future__ import annotations

import argparse
from pathlib import Path

from papis_import.core.process import read_cmd
from papis_import.core.text import repair_ligature_splits


class TextExtractor:
    """Extract front-matter text windows from PDF files."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def get_text(self, path: Path) -> str:
        """Extract text from the first N pages and repair common artifacts."""
        pages = max(1, int(self.args.text_pages))
        raw = read_cmd(["pdftotext", "-f", "1", "-l", str(pages), str(path), "-"])
        return repair_ligature_splits(raw)

    def get_identifier_text(self, path: Path) -> str:
        """Extract a larger front-matter window for ISBN / DOI scanning."""
        pages = max(8, int(self.args.text_pages))
        return read_cmd(["pdftotext", "-f", "1", "-l", str(pages), str(path), "-"])

    def get_sanity_text(self, path: Path) -> str:
        """Extract text from enough pages to support external-match sanity checks."""
        pages = max(5, int(self.args.text_pages))
        raw = read_cmd(["pdftotext", "-f", "1", "-l", str(pages), str(path), "-"])
        return repair_ligature_splits(raw)

    def get_wide_text(self, path: Path) -> str:
        """Extract a wider window for text-LLM escalation on hard cases."""
        pages = max(5, int(getattr(self.args, "text_llm_pages_escalated", 5)))
        raw = read_cmd(["pdftotext", "-f", "1", "-l", str(pages), str(path), "-"])
        return repair_ligature_splits(raw)

