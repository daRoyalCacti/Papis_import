"""Streaming output writers: result TSVs, profile TSV, debug JSONL."""
from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from papis_import.models import Record
from papis_import.output.schema import RESULT_COLUMNS, PROFILE_COLUMNS
from papis_import.output.serializers import (
    record_to_result_dict,
    record_to_debug_obj,
    record_to_profile_dict,
)


def _now_iso() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


_ALLOWED_LIVE_KEYS = {
    "status",
    "idx",
    "total",
    "file_path",
    "file_name",
    "phase",
    "bucket",
    "event",
    "message",
    "started_at",
    "updated_at",
    "wait_s",
    "sleep_reason",
    "attempt",
    "max_retries",
    "http_status",
    "method",
    "url",
    "remaining_tokens",
    "needed_tokens",
    "rate_limit",
    "rate_limit_error",
}


_TRANSIENT_LIVE_KEYS = {
    "bucket",
    "method",
    "url",
    "wait_s",
    "sleep_reason",
    "attempt",
    "max_retries",
    "http_status",
    "remaining_tokens",
    "needed_tokens",
    "rate_limit",
    "rate_limit_error",
}


def _clean_live_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_live_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_live_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class LiveStatusReporter:
    """Atomically rewrites a compact, mutable JSON status file.

    The reporter keeps in-flight state separate from completed per-file debug
    JSONL.  Only sanitized, whitelisted fields are written.
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {}
        self._run_started_at = _now_iso()

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def file_started(self, idx: int, total: int, path: Path) -> None:
        self.update(
            status="processing",
            idx=idx,
            total=total,
            file_path=str(path),
            file_name=path.name,
            phase="start",
            event="file_started",
            message=f"processing {path}",
        )

    def file_completed(self, idx: int, total: int, path: Path) -> None:
        self.update(
            status="completed",
            idx=idx,
            total=total,
            file_path=str(path),
            file_name=path.name,
            phase="finalize",
            event="file_completed",
            message=f"completed {path}",
        )

    def file_error(self, idx: int, total: int, path: Path, message: str) -> None:
        self.update(
            status="error",
            idx=idx,
            total=total,
            file_path=str(path),
            file_name=path.name,
            phase="finalize",
            event="file_error",
            message=message,
        )

    def run_finished(self, total: int) -> None:
        self.update(
            status="finished",
            idx=total,
            total=total,
            phase="finalize",
            event="run_finished",
            message="run finished",
        )

    def update(self, **kwargs: Any) -> None:
        if self._path is None:
            return
        now = _now_iso()
        clean = {
            k: _clean_live_value(v)
            for k, v in kwargs.items()
            if k in _ALLOWED_LIVE_KEYS and v not in ("", None)
        }
        with self._lock:
            if not self._state:
                self._state["started_at"] = self._run_started_at
            for key in _TRANSIENT_LIVE_KEYS - set(clean):
                self._state.pop(key, None)
            self._state.update(clean)
            self._state["updated_at"] = now
            self._write_atomic(dict(self._state))

    def _write_atomic(self, state: dict[str, Any]) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            dir=str(self._path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=True, sort_keys=True, indent=2)
                f.write("\n")
            os.replace(tmp_name, self._path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


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
