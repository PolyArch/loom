#!/usr/bin/env python3
"""Check that CMSIS negative runners use repo-local temp directories."""

from __future__ import annotations

import pathlib
import sys


def main() -> int:
    root = pathlib.Path(sys.argv[1]).resolve()
    offenders = []
    for path in sorted(root.glob("cmsis-*/test_runner*.sh")):
        text = path.read_text()
        if "mktemp -d -t" in text:
            offenders.append(str(path.relative_to(root)))
    if offenders:
        raise AssertionError("system tempdir mktemp usage: " + ", ".join(offenders))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
