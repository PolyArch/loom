#!/usr/bin/env python3
"""Ensure artifact-ladder tests keep scratch directories under repo temp."""

from __future__ import annotations

import re
import sys
from pathlib import Path


CALL_RE = re.compile(r"tempfile\.TemporaryDirectory\((?P<args>[^)]*)\)")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    candidates = sorted((repo / "test" / "artifacts").glob("test_*.py"))
    candidates.append(repo / "test" / "dataflow" / "primitive_coverage_summary.py")
    offenders: list[str] = []
    for path in candidates:
        text = path.read_text()
        for match in CALL_RE.finditer(text):
            if "dir=" not in match.group("args"):
                offenders.append(str(path.relative_to(repo)))
                break
    if offenders:
        raise AssertionError("artifact scratch dirs must use repo temp: " + ", ".join(offenders))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
