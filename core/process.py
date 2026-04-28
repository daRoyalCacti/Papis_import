"""Process and stderr helpers shared across papis_import."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def quote_shell(s: str) -> str:
    return shlex.quote(s)


def command_exists(name: str) -> bool:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(p) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return True
    return False


def read_cmd(cmd: list[str], timeout: int = 40) -> str:
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception:
        return ""
    return cp.stdout or "" if cp.returncode == 0 else ""

