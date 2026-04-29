"""Compatibility facade for local metadata extraction strategies.

The focused implementations live under :mod:`papis_import.extractor_parts`.
This class keeps the historical public methods in place while delegating each
strategy to a smaller component.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from papis_import.extractor_parts.filename import FilenameExtractor
from papis_import.extractor_parts.grobid import GrobidExtractor
from papis_import.extractor_parts.pdf_metadata import PdfMetadataExtractor
from papis_import.extractor_parts.text import TextExtractor
from papis_import.extractor_parts.text_header import TextHeaderExtractor
from papis_import.extractor_parts.text_llm import TextLlmExtractor
from papis_import.extractor_parts.vision_llm import VisionLlmExtractor
from papis_import.http_client import HttpClient
from papis_import.models import Candidate


class Extractor:
    """Compatibility wrapper over focused extraction components."""

    def __init__(self, args: argparse.Namespace, http: HttpClient) -> None:
        self.args = args
        self.http = http
        self.text = TextExtractor(args)
        self.pdf_metadata = PdfMetadataExtractor()
        self.filename = FilenameExtractor()
        self.text_header = TextHeaderExtractor()
        self.grobid = GrobidExtractor(args, http)
        self.text_llm = TextLlmExtractor(args, http)
        self.vision_llm = VisionLlmExtractor(args, http)
        self.last_llm_debug: dict[str, Any] = {}

    def get_text(self, path: Path) -> str:
        return self.text.get_text(path)

    def get_identifier_text(self, path: Path) -> str:
        return self.text.get_identifier_text(path)

    def get_sanity_text(self, path: Path) -> str:
        return self.text.get_sanity_text(path)

    def embedded_metadata(self, path: Path) -> list[Candidate]:
        return self.pdf_metadata.embedded_metadata(path)

    def pdfinfo_metadata(self, path: Path) -> list[Candidate]:
        return self.pdf_metadata.pdfinfo_metadata(path)

    def filename_candidate(self, path: Path) -> list[Candidate]:
        return self.filename.candidate(path)

    def text_header_candidate(self, text: str) -> list[Candidate]:
        return self.text_header.candidate(text)

    def grobid_candidate(self, path: Path) -> list[Candidate]:
        return self.grobid.candidate(path)

    def llm_candidate(
        self,
        path: Path,
        text: str,
        filename_cand: Candidate | None,
    ) -> list[Candidate]:
        candidates = self.text_llm.candidate(path, text, filename_cand)
        self.last_llm_debug = dict(self.text_llm.last_debug)
        return candidates

    def _render_pages_to_b64(
        self,
        path: Path,
        pages: int,
        dpi: int,
        debug: dict[str, str],
    ) -> list[str]:
        return self.vision_llm.render_pages_to_b64(path, pages, dpi, debug)

    def vision_llm_candidate(
        self,
        path: Path,
        filename_cand: Candidate | None,
        is_book: bool = False,
    ) -> tuple[list[Candidate], dict[str, str]]:
        return self.vision_llm.candidate(path, filename_cand, is_book=is_book)
