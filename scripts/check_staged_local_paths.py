#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


TOP_LEVEL_TEMP_DIRECTORY = b"temp"


def git_bytes(cwd: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(message or "Git staged-path query failed")
    return completed.stdout


def is_local_path(path: bytes) -> bool:
    return (
        path == b"loom-local-config.json"
        or path == b"build"
        or path.startswith(b"build/")
    )


def is_top_level_temp_path(path: bytes) -> bool:
    return path == TOP_LEVEL_TEMP_DIRECTORY or path.startswith(
        TOP_LEVEL_TEMP_DIRECTORY + b"/"
    )


def main() -> int:
    try:
        root = Path(
            os.fsdecode(git_bytes(Path.cwd(), "rev-parse", "--show-toplevel").strip())
        )
        raw = git_bytes(
            root,
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--no-renames",
            "--diff-filter=ACMT",
            "--",
        )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    staged_paths = {path for path in raw.split(b"\0") if path}
    try:
        tracked_ignored_raw = git_bytes(
            root, "ls-files", "-ci", "--exclude-standard", "-z", "--"
        )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    tracked_ignored_paths = {
        path for path in tracked_ignored_raw.split(b"\0") if path
    }
    violations = sorted(
        (path, "staged local output")
        for path in staged_paths
        if is_local_path(path) or is_top_level_temp_path(path)
    )
    violations.extend(
        (path, "tracked ignored file")
        for path in sorted(tracked_ignored_paths)
        if path not in {entry[0] for entry in violations}
    )
    for path, reason in violations:
        print(
            f"{reason} is not publishable: {os.fsdecode(path)!r}",
            file=sys.stderr,
        )
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
