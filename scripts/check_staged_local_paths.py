#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
        or path == b"temp"
        or path.startswith(b"temp/")
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

    violations = sorted(
        path for path in raw.split(b"\0") if path and is_local_path(path)
    )
    for path in violations:
        print(
            f"staged local output is not publishable: {os.fsdecode(path)!r}",
            file=sys.stderr,
        )
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
