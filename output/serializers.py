"""Record → serialized-row converters for each output format."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from papis_import.models import Record, TimingBreakdown


def _fmt_s(value: float) -> str:
    return f"{max(0.0, value):.6f}"


def _identifier_lookups_json(timing: TimingBreakdown) -> str:
    return json.dumps(
        [asdict(item) for item in timing.identifier_lookups],
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    )


def _title_searches_json(timing: TimingBreakdown) -> str:
    return json.dumps(
        [{k: v for k, v in asdict(item).items() if k != "query_traces"}
         for item in timing.title_searches],
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    )


def _title_search_by_source(timing: TimingBreakdown) -> dict:
    return {item.source: item for item in timing.title_searches}


def record_to_result_dict(rec: Record) -> dict[str, str]:
    """Produce a flat string dict keyed by RESULT_COLUMNS for csv.DictWriter."""
    m = rec.result
    return {
        "File Path":         str(rec.path),
        "Tags":              ", ".join(rec.tags),
        "Source":            m.source,
        "Confidence":        m.confidence,
        "Verified":          "yes" if m.verified else "no",
        "Title":             m.title,
        "Authors":           "; ".join(m.authors),
        "Year":              m.year,
        "DOI":               m.doi,
        "ISBN":              m.isbn,
        "arXiv":             m.arxiv,
        "Sanity Passed":     "yes" if m.sanity_passed else "no",
        "Sanity Score":      f"{m.sanity_score:.3f}",
        "Auto Safe":         "yes" if m.auto_safe else "no",
        "Needs OCR":         "yes" if m.needs_ocr else "no",
        "Notes":             " | ".join(m.notes),
        "Imported":          "yes" if rec.imported else "no",
        "Error":             rec.error,
        "Suggested Command": rec.suggested_command,
        "Final Source":      rec.debug.get("final_source", m.source),
        "Soft Auto":         "yes" if m.soft_auto else "no",
        "Soft Auto Reasons": " | ".join(m.soft_auto_reasons),
    }


def record_to_debug_obj(rec: Record) -> dict[str, Any]:
    """Produce a JSON-serialisable dict for the debug JSONL file."""
    m = rec.result
    return {
        "file_path":         str(rec.path),
        "tags":              rec.tags,
        "imported":          rec.imported,
        "error":             rec.error,
        "suggested_command": rec.suggested_command,
        "result": {
            "title":              m.title,
            "authors":            m.authors,
            "year":               m.year,
            "doi":                m.doi,
            "isbn":               m.isbn,
            "arxiv":              m.arxiv,
            "source":             m.source,
            "confidence":         m.confidence,
            "verified":           m.verified,
            "sanity_passed":      m.sanity_passed,
            "sanity_score":       m.sanity_score,
            "auto_safe":          m.auto_safe,
            "soft_auto":          m.soft_auto,
            "soft_auto_reasons":  m.soft_auto_reasons,
            "needs_ocr":          m.needs_ocr,
            "notes":              m.notes,
        },
        "debug":  dict(rec.debug),
        "timing": asdict(rec.timing),
    }


def record_to_profile_dict(rec: Record, status: str = "processed") -> dict[str, str]:
    """Produce a flat string dict keyed by PROFILE_COLUMNS for csv.DictWriter."""
    m = rec.result
    t = rec.timing
    d = rec.debug
    ts_by_source = _title_search_by_source(t)

    def title_cols(source: str) -> tuple[str, str, str, str]:
        item = ts_by_source.get(source)
        if item is None:
            return ("0.000000", "0", "0", "0")
        return (
            _fmt_s(item.elapsed_s),
            str(item.candidates_tried),
            str(item.matches_returned),
            str(item.errors),
        )

    cr = title_cols("crossref")
    oa = title_cols("openalex")
    ss = title_cols("semanticscholar")
    ol = title_cols("openlibrary")
    gb = title_cols("google_books")

    return {
        "File Path":   str(rec.path),
        "Status":      status,
        "Error":       rec.error,
        "Final Source": d.get("final_source", m.source),
        "Confidence":  m.confidence,
        "Verified":    "yes" if m.verified else "no",
        "File Wall s":          _fmt_s(t.file_wall_s),
        "Resolve Total s":      _fmt_s(t.resolve_total_s),
        "Text Extract s":       _fmt_s(t.text_extract_s),
        "Embedded Metadata s":  _fmt_s(t.embedded_metadata_s),
        "GROBID s":             _fmt_s(t.grobid_s),
        "Text LLM s":           _fmt_s(t.text_llm_s),
        "Text LLM HTTP s":              d.get("text_llm_http_s", "0.000000"),
        "Text LLM Retry Sleep s":       d.get("text_llm_retry_sleep_s", "0.000000"),
        "Text LLM Pacing s":            d.get("text_llm_pacing_s", "0.000000"),
        "Text LLM Provider Total s":    d.get("text_llm_provider_total_s", ""),
        "Text LLM Provider Queue s":    d.get("text_llm_provider_queue_s", ""),
        "Text LLM Attempts":            d.get("text_llm_attempts", ""),
        "Text LLM Cache Hit":           d.get("text_llm_cache_hit", ""),
        "Vision LLM s":         _fmt_s(t.vision_llm_s),
        "Vision HTTP s":                d.get("vision_http_s", "0.000000"),
        "Vision Retry Sleep s":         d.get("vision_retry_sleep_s", "0.000000"),
        "Vision Provider Total s":      d.get("vision_provider_total_s", ""),
        "Vision Provider Queue s":      d.get("vision_provider_queue_s", ""),
        "Vision Attempts":              d.get("vision_attempts", ""),
        "Vision Cache Hit":             d.get("vision_cache_hit", ""),
        "Vision Pacing s":      _fmt_s(t.vision_pacing_s),
        "Identifier Lookups s": _fmt_s(t.identifier_lookups_s),
        "Title Search s":       _fmt_s(t.title_search_s),
        "OCR Retry s":          _fmt_s(t.ocr_retry_s),
        "OCRMyPDF s":           _fmt_s(t.ocrmypdf_s),
        "OCR Reresolve s":      _fmt_s(t.ocr_reresolve_s),
        "Best Local s":         _fmt_s(t.best_local_s),
        "Header Candidate s":   _fmt_s(t.header_candidate_s),
        "Identifier Lookup Count": str(len(t.identifier_lookups)),
        "Identifier Lookups JSON": _identifier_lookups_json(t),
        "Crossref Search s":                     cr[0],
        "Crossref Search Candidates Tried":      cr[1],
        "Crossref Search Matches":               cr[2],
        "Crossref Search Errors":                cr[3],
        "OpenAlex Search s":                     oa[0],
        "OpenAlex Search Candidates Tried":      oa[1],
        "OpenAlex Search Matches":               oa[2],
        "OpenAlex Search Errors":                oa[3],
        "Semantic Scholar Search s":                 ss[0],
        "Semantic Scholar Search Candidates Tried":  ss[1],
        "Semantic Scholar Search Matches":           ss[2],
        "Semantic Scholar Search Errors":            ss[3],
        "OpenLibrary Search s":                  ol[0],
        "OpenLibrary Search Candidates Tried":   ol[1],
        "OpenLibrary Search Matches":            ol[2],
        "OpenLibrary Search Errors":             ol[3],
        "Google Books Search s":                 gb[0],
        "Google Books Search Candidates Tried":  gb[1],
        "Google Books Search Matches":           gb[2],
        "Google Books Search Errors":            gb[3],
        "Title Searches JSON": _title_searches_json(t),
    }
