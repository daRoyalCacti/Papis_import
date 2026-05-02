"""Implementation components for local metadata extraction."""
from __future__ import annotations

import dataclasses

from papis_import.extractor_parts.filename import FilenameExtractor
from papis_import.extractor_parts.grobid import GrobidExtractor
from papis_import.extractor_parts.pdf_metadata import PdfMetadataExtractor
from papis_import.extractor_parts.text import TextExtractor
from papis_import.extractor_parts.text_header import TextHeaderExtractor
from papis_import.extractor_parts.text_llm import TextLlmExtractor
from papis_import.extractor_parts.vision_llm import VisionLlmExtractor


@dataclasses.dataclass
class ExtractorSet:
    """All extraction components needed by the resolution pipeline."""

    text: TextExtractor
    pdf: PdfMetadataExtractor
    filename: FilenameExtractor
    text_header: TextHeaderExtractor
    grobid: GrobidExtractor
    text_llm: TextLlmExtractor
    vision_llm: VisionLlmExtractor

    @classmethod
    def build(cls, args, http) -> "ExtractorSet":
        return cls(
            text=TextExtractor(args),
            pdf=PdfMetadataExtractor(),
            filename=FilenameExtractor(),
            text_header=TextHeaderExtractor(),
            grobid=GrobidExtractor(args, http),
            text_llm=TextLlmExtractor(args, http),
            vision_llm=VisionLlmExtractor(args, http),
        )


__all__ = [
    "ExtractorSet",
    "FilenameExtractor",
    "GrobidExtractor",
    "PdfMetadataExtractor",
    "TextExtractor",
    "TextHeaderExtractor",
    "TextLlmExtractor",
    "VisionLlmExtractor",
]
