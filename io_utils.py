from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = "~/.config/papis-import/config.json"

DEFAULT_REVIEW_TSV  = PROJECT_ROOT / "out" / "papis_import_review.tsv"
DEFAULT_SOFT_TSV    = PROJECT_ROOT / "out" / "papis_import_soft.tsv"
DEFAULT_DEBUG_TSV   = PROJECT_ROOT / "out" / "papis_import_debug.jsonl"
DEFAULT_PROFILE_TSV = PROJECT_ROOT / "out" / "papis_import_profile.tsv"

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expanduser(str(value))).resolve()


def load_json_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    missing_ok: bool = False,
) -> dict[str, object]:
    config_path = expand_path(path)
    if not config_path.exists():
        if missing_ok:
            return {}
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse config JSON at {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Config at {config_path} must be a JSON object")
    return data


def read_tsv_dicts(path: str | Path, *, required: bool = True) -> list[dict[str, str]]:
    tsv_path = expand_path(path)
    if not tsv_path.exists():
        if required:
            raise FileNotFoundError(tsv_path)
        return []
    with tsv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv_dicts(
    path: str | Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    tsv_path = expand_path(path)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_yes(value: object) -> bool:
    return str(value).strip().lower() == "yes"


def _flatten_debug_obj(obj: dict[str, Any]) -> dict[str, str]:
    """Convert a debug JSONL record to a flat string dict matching old debug TSV column names."""
    result = obj.get("result", {})
    debug  = obj.get("debug",  {})

    def yn(v: object) -> str:
        if isinstance(v, bool):
            return "yes" if v else "no"
        return str(v).strip()

    def lst(v: object, sep: str = " | ") -> str:
        if isinstance(v, list):
            return sep.join(str(x) for x in v)
        return str(v or "")

    flat: dict[str, str] = {
        "File Path":         str(obj.get("file_path", "")),
        "Tags":              ", ".join(obj.get("tags", [])),
        "Imported":          "yes" if obj.get("imported") else "no",
        "Error":             str(obj.get("error", "")),
        "Suggested Command": str(obj.get("suggested_command", "")),
        "Title":             str(result.get("title", "")),
        "Authors":           "; ".join(result.get("authors", [])),
        "Year":              str(result.get("year", "")),
        "DOI":               str(result.get("doi", "")),
        "ISBN":              str(result.get("isbn", "")),
        "arXiv":             str(result.get("arxiv", "")),
        "Source":            str(result.get("source", "")),
        "Confidence":        str(result.get("confidence", "")),
        "Verified":          yn(result.get("verified", False)),
        "Sanity Passed":     yn(result.get("sanity_passed", False)),
        "Sanity Score":      f"{float(result.get('sanity_score', 0)):.3f}",
        "Auto Safe":         yn(result.get("auto_safe", False)),
        "Soft Auto":         yn(result.get("soft_auto", False)),
        "Soft Auto Reasons": lst(result.get("soft_auto_reasons", [])),
        "Needs OCR":         yn(result.get("needs_ocr", False)),
        "Notes":             lst(result.get("notes", [])),
    }
    _DEBUG_FIELD_MAP = {
        "final_source":              "Final Source",
        "vision_used":               "Vision Used",
        "vision_trigger":            "Vision Trigger",
        "vision_status":             "Vision Status",
        "vision_error":              "Vision Error",
        "vision_cache_hit":          "Vision Cache Hit",
        "vision_attempts":           "Vision Attempts",
        "vision_http_s":             "Vision HTTP s",
        "vision_retry_sleep_s":      "Vision Retry Sleep s",
        "vision_http_json":          "Vision HTTP JSON",
        "vision_prompt_tokens":      "Vision Prompt Tokens",
        "vision_completion_tokens":  "Vision Completion Tokens",
        "vision_total_tokens":       "Vision Total Tokens",
        "vision_provider_total_s":   "Vision Provider Total s",
        "vision_provider_queue_s":   "Vision Provider Queue s",
        "vision_pacing_s":           "Vision Pacing s",
        "vision_tokens_remaining":   "Vision Tokens Remaining",
        "vision_escalated":          "Vision Escalated",
        "vision_model":              "Vision Model",
        "vision_pages":              "Vision Pages",
        "vision_dpi":                "Vision DPI",
        "vision_title":              "Vision Title",
        "vision_authors":            "Vision Authors",
        "vision_year":               "Vision Year",
        "text_llm_used":             "Text LLM Used",
        "text_llm_model":            "Text LLM Model",
        "text_llm_status":           "Text LLM Status",
        "text_llm_error":            "Text LLM Error",
        "text_llm_cache_hit":        "Text LLM Cache Hit",
        "text_llm_attempts":         "Text LLM Attempts",
        "text_llm_http_s":           "Text LLM HTTP s",
        "text_llm_retry_sleep_s":    "Text LLM Retry Sleep s",
        "text_llm_pacing_s":         "Text LLM Pacing s",
        "text_llm_tokens_remaining": "Text LLM Tokens Remaining",
        "text_llm_prompt_tokens":    "Text LLM Prompt Tokens",
        "text_llm_completion_tokens":"Text LLM Completion Tokens",
        "text_llm_total_tokens":     "Text LLM Total Tokens",
        "text_llm_provider_total_s": "Text LLM Provider Total s",
        "text_llm_provider_queue_s": "Text LLM Provider Queue s",
        "text_llm_title":            "Text LLM Title",
        "text_llm_authors":          "Text LLM Authors",
        "text_llm_year":             "Text LLM Year",
        "text_llm_doi":              "Text LLM DOI",
        "text_llm_isbn":             "Text LLM ISBN",
        "text_llm_arxiv":            "Text LLM arXiv",
        "text_llm_http_json":        "Text LLM HTTP JSON",
        "grobid_used":               "GROBID Used",
        "grobid_title":              "GROBID Title",
        "grobid_authors":            "GROBID Authors",
        "grobid_year":               "GROBID Year",
        "local_best_source":         "Local Best Source",
        "identifier_dois":           "Identifier DOIs",
        "identifier_isbns":          "Identifier ISBNs",
        "identifier_arxivs":         "Identifier arXivs",
        "candidate_sources":         "Candidate Sources",
        "candidates_json":           "Candidates JSON",
        "title_search_queries_json": "Title Search Queries JSON",
    }
    for debug_key, col_name in _DEBUG_FIELD_MAP.items():
        flat[col_name] = str(debug.get(debug_key, ""))
    return flat


def read_debug_jsonl(
    path: str | Path,
    *,
    required: bool = True,
) -> list[dict[str, str]]:
    """Read a debug JSONL file and return flat string dicts matching old debug TSV columns."""
    jsonl_path = expand_path(path)
    if not jsonl_path.exists():
        if required:
            raise FileNotFoundError(jsonl_path)
        return []
    rows: list[dict[str, str]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_flatten_debug_obj(json.loads(line)))
    return rows


def parse_json_list_cell(value: object) -> list[dict[str, Any]]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]
