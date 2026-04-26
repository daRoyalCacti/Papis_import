"""Helpers for starting/stopping a local GROBID service on demand."""
from __future__ import annotations

import argparse
import contextlib
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from papis_import.utils import command_exists, eprint


@dataclass
class GrobidSession:
    url: str
    runtime: str = ""
    container_name: str = ""
    started_here: bool = False


def _is_local_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in {"", "localhost", "127.0.0.1"}


def _pick_runtime(preferred: str) -> str:
    pref = (preferred or "auto").strip().lower()
    if pref in {"docker", "podman"}:
        if command_exists(pref):
            return pref
        raise RuntimeError(f"requested GROBID runtime '{pref}' was not found in PATH")
    for candidate in ("docker", "podman"):
        if command_exists(candidate):
            return candidate
    raise RuntimeError("neither docker nor podman was found in PATH")


def grobid_is_alive(base_url: str, timeout: float = 3.0) -> bool:
    url = base_url.rstrip("/") + "/api/isalive"
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace").strip().lower()
        return resp.status == 200 and body == "true"
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


def wait_for_grobid(base_url: str, timeout: int = 180, interval: float = 2.0, verbose: bool = False) -> None:
    deadline = time.time() + max(1, int(timeout))
    last_status = "service not yet alive"
    while time.time() < deadline:
        if grobid_is_alive(base_url):
            return
        if verbose:
            eprint(f"[grobid] waiting for {base_url} …")
        time.sleep(interval)
    raise RuntimeError(f"GROBID did not become ready at {base_url} within {timeout}s ({last_status})")


def _start_container(runtime: str, image: str, port: int, verbose: bool = False) -> str:
    name = f"papis-import-grobid-{int(time.time())}-{port}"
    cmd = [
        runtime,
        "run",
        "--rm",
        "--detach",
        "--init",
        "--name",
        name,
        "-p",
        f"127.0.0.1:{port}:8070",
        image,
    ]
    if verbose:
        eprint("[grobid] starting:", " ".join(cmd))
    cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        raise RuntimeError((cp.stderr or cp.stdout or "failed to start GROBID container").strip())
    return name


def stop_container(runtime: str, container_name: str, verbose: bool = False) -> None:
    if not runtime or not container_name:
        return
    cmd = [runtime, "stop", container_name]
    if verbose:
        eprint("[grobid] stopping:", " ".join(cmd))
    subprocess.run(cmd, capture_output=True, text=True, check=False)


@contextlib.contextmanager
def local_grobid_session(args: argparse.Namespace):
    """Yield a GrobidSession, optionally starting a temporary local container.

    Rules:
    - If --grobid-url already points to a live service, reuse it.
    - If --start-local-grobid is absent, do nothing.
    - If --start-local-grobid is present and no URL is supplied, use
      http://127.0.0.1:<grobid_port>.
    - Only auto-start a container for localhost/127.0.0.1 URLs.
    """
    explicit_url = getattr(args, "grobid_url", "").strip()
    start_local = bool(getattr(args, "start_local_grobid", False))
    port = int(getattr(args, "grobid_port", 8070))
    timeout = int(getattr(args, "grobid_start_timeout", 180))
    image = getattr(args, "grobid_image", "grobid/grobid:0.9.0-full").strip() or "grobid/grobid:0.9.0-full"
    runtime_pref = getattr(args, "grobid_runtime", "auto")
    verbose = bool(getattr(args, "verbose", False))

    url = explicit_url or (f"http://127.0.0.1:{port}" if start_local else "")
    session = GrobidSession(url=url)

    if url and grobid_is_alive(url):
        if verbose:
            eprint(f"[grobid] reusing running service at {url}")
        args.grobid_url = url
        yield session
        return

    if not start_local:
        args.grobid_url = explicit_url
        yield session
        return

    if not url:
        url = f"http://127.0.0.1:{port}"
        session.url = url

    if not _is_local_url(url):
        raise RuntimeError("--start-local-grobid only supports localhost/127.0.0.1 grobid URLs")

    runtime = _pick_runtime(runtime_pref)
    container_name = _start_container(runtime=runtime, image=image, port=port, verbose=verbose)
    session.runtime = runtime
    session.container_name = container_name
    session.started_here = True
    args.grobid_url = url
    try:
        wait_for_grobid(url, timeout=timeout, verbose=verbose)
        yield session
    finally:
        stop_container(runtime, container_name, verbose=verbose)
