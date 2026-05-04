"""Helpers for starting/stopping a local Ollama service on demand."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from papis_import.llm_config import local_llm_enabled, local_llm_model, ollama_host
from papis_import.utils import command_exists, eprint


@dataclass
class OllamaSession:
    url: str
    model: str
    started_here: bool = False
    process: subprocess.Popen[str] | None = None


def _is_local_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _ollama_env(url: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = url
    return env


def ollama_is_alive(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=timeout) as resp:
            resp.read()
        return resp.status == 200
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


def wait_for_ollama(base_url: str, timeout: int = 60, interval: float = 1.0, verbose: bool = False) -> None:
    deadline = time.time() + max(1, int(timeout))
    while time.time() < deadline:
        if ollama_is_alive(base_url):
            return
        if verbose:
            eprint(f"[ollama] waiting for {base_url} ...")
        time.sleep(interval)
    raise RuntimeError(f"Ollama did not become ready at {base_url} within {timeout}s")


def installed_models(base_url: str, timeout: float = 10.0) -> set[str]:
    with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    models = data.get("models") if isinstance(data, dict) else []
    names: set[str] = set()
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict) and item.get("name"):
                names.add(str(item["name"]))
    return names


def ensure_model(base_url: str, model: str, *, pull_missing: bool, verbose: bool = False) -> None:
    names = installed_models(base_url)
    if model in names:
        return
    # Ollama may report "name:latest" while users type "name".
    if ":" not in model and f"{model}:latest" in names:
        return
    if not pull_missing:
        raise RuntimeError(
            f"Ollama model '{model}' is not installed. "
            f"Run `ollama pull {model}` or pass --ollama-pull-missing."
        )
    cmd = ["ollama", "pull", model]
    if verbose:
        eprint("[ollama] pulling:", " ".join(cmd))
    cp = subprocess.run(
        cmd,
        env=_ollama_env(base_url),
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError((cp.stderr or cp.stdout or f"failed to pull Ollama model '{model}'").strip())


def start_ollama(base_url: str, verbose: bool = False) -> subprocess.Popen[str]:
    if not command_exists("ollama"):
        raise RuntimeError("'ollama' executable was not found in PATH")
    if not _is_local_url(base_url):
        raise RuntimeError("--local-llm can only auto-start Ollama for localhost/127.0.0.1 URLs")
    cmd = ["ollama", "serve"]
    if verbose:
        eprint("[ollama] starting:", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        env=_ollama_env(base_url),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )


def stop_ollama(process: subprocess.Popen[str] | None, verbose: bool = False) -> None:
    if process is None or process.poll() is not None:
        return
    if verbose:
        eprint("[ollama] stopping service started by this run")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


@contextlib.contextmanager
def local_ollama_session(args: argparse.Namespace):
    """Yield an OllamaSession when --local-llm is enabled, otherwise no-op."""
    if not local_llm_enabled(args):
        yield OllamaSession(url="", model="")
        return

    url = ollama_host(args)
    model = local_llm_model(args)
    timeout = int(getattr(args, "ollama_start_timeout", 60) or 60)
    pull_missing = bool(getattr(args, "ollama_pull_missing", False))
    verbose = bool(getattr(args, "verbose", False))
    session = OllamaSession(url=url, model=model)

    if ollama_is_alive(url):
        if verbose:
            eprint(f"[ollama] reusing running service at {url}")
        ensure_model(url, model, pull_missing=pull_missing, verbose=verbose)
        yield session
        return

    process = start_ollama(url, verbose=verbose)
    session.process = process
    session.started_here = True
    try:
        wait_for_ollama(url, timeout=timeout, verbose=verbose)
        ensure_model(url, model, pull_missing=pull_missing, verbose=verbose)
        yield session
    finally:
        stop_ollama(process, verbose=verbose)
