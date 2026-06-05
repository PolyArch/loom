#!/usr/bin/env python3
"""Ensure repo tests and runners do not default to system temp directories."""

from __future__ import annotations

import re
import sys
from pathlib import Path


TEMP_DIR_RE = re.compile(r"tempfile\.TemporaryDirectory\((?P<args>[^)]*)\)")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    offenders: list[str] = []
    for relative in (
        "scripts/test_make_worktree.py",
        "test/techmap/perf/perf_runner.py",
    ):
        path = repo / relative
        text = path.read_text()
        if "tempfile.gettempdir()" in text:
            offenders.append(relative)
            continue
        for match in TEMP_DIR_RE.finditer(text):
            if "dir=" not in match.group("args"):
                offenders.append(relative)
                break
    if offenders:
        raise AssertionError("system temp defaults are not allowed: " + ", ".join(offenders))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
