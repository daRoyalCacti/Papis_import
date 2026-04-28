#!/usr/bin/env python3
"""Run all papis_import debug helper scripts in a stable order."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DebugScript:
    key: str
    path: Path
    description: str
    supports_config: bool = True
    supports_force: bool = False
    supports_no_pdfs: bool = False
    supports_link: bool = False
    supports_limit: bool = False
    supports_top: bool = False


SCRIPTS = [
    DebugScript(
        key="profile",
        path=_PROJECT_ROOT / "debug" / "summarize_profile.py",
        description="Summarize profile timing and resolver behavior",
        supports_top=True,
    ),
    DebugScript(
        key="review",
        path=_PROJECT_ROOT / "debug" / "prepare_review_bundle.py",
        description="Prepare manual-review bundle",
        supports_force=True,
        supports_no_pdfs=True,
        supports_link=True,
        supports_limit=True,
    ),
    DebugScript(
        key="soft",
        path=_PROJECT_ROOT / "debug" / "prepare_soft_accept_bundle.py",
        description="Prepare soft-accept audit bundle",
        supports_force=True,
        supports_no_pdfs=True,
        supports_link=True,
        supports_limit=True,
    ),
]


def split_arg_string(value: str) -> list[str]:
    return shlex.split(value) if value else []


def extend_repeated_arg_strings(values: list[str] | None) -> list[str]:
    args: list[str] = []
    for value in values or []:
        args.extend(split_arg_string(value))
    return args


def selected_scripts(args: argparse.Namespace) -> list[DebugScript]:
    only = set(args.only or [])
    skip = set(args.skip or [])
    scripts = [script for script in SCRIPTS if not only or script.key in only]
    return [script for script in scripts if script.key not in skip]


def script_extra_args(args: argparse.Namespace, script: DebugScript) -> list[str]:
    extras: list[str] = []
    extras.extend(extend_repeated_arg_strings(args.common_args))
    if script.key == "profile":
        extras.extend(extend_repeated_arg_strings(args.profile_args))
    elif script.key == "review":
        extras.extend(extend_repeated_arg_strings(args.review_args))
    elif script.key == "soft":
        extras.extend(extend_repeated_arg_strings(args.soft_args))
    return extras


def build_command(args: argparse.Namespace, script: DebugScript) -> list[str]:
    cmd = [args.python, str(script.path)]
    if script.supports_config and args.config:
        cmd.extend(["--config", args.config])
    if script.supports_force and args.force:
        cmd.append("--force")
    if script.supports_no_pdfs and args.no_pdfs:
        cmd.append("--no-pdfs")
    if script.supports_link and args.link:
        cmd.append("--link")
    if script.supports_limit and args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    if script.supports_top and args.top > 0:
        cmd.extend(["--top", str(args.top)])
    cmd.extend(script_extra_args(args, script))
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all debug helper scripts for the current papis_import outputs.",
        epilog=(
            "Examples:\n"
            "  python debug/run_all_debug.py --force --no-pdfs\n"
            "  python debug/run_all_debug.py --only review soft --force --limit 25\n"
            "  python debug/run_all_debug.py --profile-args '--dest out/profile.md --top 20'\n"
            "  python debug/run_all_debug.py --soft-args '--sort tsv --dest out/soft_check'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="", help="Config JSON path passed to each debug script")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[script.key for script in SCRIPTS],
        help="Run only the named debug scripts",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=[script.key for script in SCRIPTS],
        help="Skip the named debug scripts",
    )
    parser.add_argument("--list", action="store_true", help="List scripts and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--continue-on-error", action="store_true", help="Run remaining scripts after a failure")

    parser.add_argument("--force", action="store_true", help="Pass --force to bundle scripts")
    parser.add_argument("--no-pdfs", action="store_true", help="Pass --no-pdfs to bundle scripts")
    parser.add_argument("--link", action="store_true", help="Pass --link to bundle scripts")
    parser.add_argument("--limit", type=int, default=0, help="Pass --limit to bundle scripts")
    parser.add_argument("--top", type=int, default=0, help="Pass --top to summarize_profile")

    parser.add_argument(
        "--common-args",
        action="append",
        default=[],
        metavar="ARGS",
        help="Extra shell-style args appended to every script; may be repeated",
    )
    parser.add_argument(
        "--profile-args",
        action="append",
        default=[],
        metavar="ARGS",
        help="Extra shell-style args appended to summarize_profile.py; may be repeated",
    )
    parser.add_argument(
        "--review-args",
        action="append",
        default=[],
        metavar="ARGS",
        help="Extra shell-style args appended to prepare_review_bundle.py; may be repeated",
    )
    parser.add_argument(
        "--soft-args",
        action="append",
        default=[],
        metavar="ARGS",
        help="Extra shell-style args appended to prepare_soft_accept_bundle.py; may be repeated",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scripts = selected_scripts(args)

    if args.list:
        for script in scripts:
            print(f"{script.key}: {script.path} - {script.description}")
        return 0

    if not scripts:
        print("[error] no debug scripts selected", file=sys.stderr)
        return 2

    failures: list[tuple[str, int]] = []
    for index, script in enumerate(scripts, start=1):
        cmd = build_command(args, script)
        print(f"[{index}/{len(scripts)}] {script.key}: {subprocess.list2cmdline(cmd)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=False)
        if result.returncode != 0:
            failures.append((script.key, result.returncode))
            if not args.continue_on_error:
                break

    if failures:
        print("", file=sys.stderr)
        print("[error] debug script failures:", file=sys.stderr)
        for key, code in failures:
            print(f"  {key}: exit {code}", file=sys.stderr)
        return failures[0][1] or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
