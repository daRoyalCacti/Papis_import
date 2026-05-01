"""Streaming output writers: result TSVs, profile TSV, debug JSONL."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from papis_import.models import Record
from papis_import.output.schema import RESULT_COLUMNS, PROFILE_COLUMNS
from papis_import.output.serializers import (
    record_to_result_dict,
    record_to_debug_obj,
    record_to_profile_dict,
)


class ResultWriter:
    """Streams result records to auto, review, and soft TSVs.

    auto_safe rows  → auto only
    soft_auto rows  → auto + soft
    everything else → review
    """

    def __init__(self, auto: Path, review: Path, soft: Path) -> None:
        self._auto = auto
        self._review = review
        self._soft = soft

    def init(self) -> None:
        for p in (self._auto, self._review, self._soft):
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t").writeheader()

    def append(self, rec: Record) -> None:
        m = rec.result
        row = record_to_result_dict(rec)
        if m.auto_safe:
            self._append_to(self._auto, row)
        elif m.soft_auto:
            self._append_to(self._auto, row)
            self._append_to(self._soft, row)
        else:
            self._append_to(self._review, row)

    def _append_to(self, path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t").writerow(row)


class ProfileWriter:
    """Streams timing records to a profile TSV."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def init(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=PROFILE_COLUMNS, delimiter="\t").writeheader()

    def append(self, rec: Record, status: str = "processed") -> None:
        row = record_to_profile_dict(rec, status)
        with self._path.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=PROFILE_COLUMNS, delimiter="\t").writerow(row)


class DebugWriter:
    """Streams per-record pipeline diagnostics as JSON Lines."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def init(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.open("w").close()

    def append(self, rec: Record) -> None:
        obj = record_to_debug_obj(rec)
        line = json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
