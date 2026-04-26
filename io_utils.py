from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = "~/.config/papis-import/config.json"

DEFAULT_REVIEW_TSV = PROJECT_ROOT / "out" / "papis_import_review.tsv"
DEFAULT_DEBUG_TSV = PROJECT_ROOT / "out" / "papis_import_debug.tsv"
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
