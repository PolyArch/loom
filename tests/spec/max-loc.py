#!/usr/bin/env python3
"""Max lines-of-code check.

Scans tests/, lib/, and include/ for source files and flags any file
exceeding 1800 lines.

Exit code: 0 if all files pass, 1 if any fail.
"""

import sys
from pathlib import Path

SCAN_DIRS = ["tests", "lib", "include"]

EXTENSIONS = {
    ".h", ".hpp", ".c", ".cpp", ".sv", ".td",
    ".py", ".rs", ".cu", ".cuh", ".mlir", ".md", ".sh",
}

MAX_LINES = 1800

# Directory names to skip (generated artifacts, not project source)
SKIP_DIRS = {"Output", "__pycache__"}


def main() -> int:
    root = Path(__file__).resolve().parent.parent.parent

    violations: list[tuple[str, int]] = []
    scanned = 0

    for scan_dir in SCAN_DIRS:
        base = root / scan_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
                continue
            if path.suffix not in EXTENSIONS:
                continue
            scanned += 1
            try:
                line_count = sum(1 for _ in path.open())
            except (OSError, UnicodeDecodeError):
                continue
            if line_count > MAX_LINES:
                rel = path.relative_to(root)
                violations.append((str(rel), line_count))

    total = 1  # single test point
    if violations:
        print(f"[FAIL] {len(violations)} file(s) exceed {MAX_LINES} lines:")
        for path, count in violations:
            print(f"  {path}: {count} lines")
        passed = 0
    else:
        print(f"[PASS] All {scanned} files are within {MAX_LINES} lines")
        passed = 1

    failed = total - passed
    print()
    print(f"MAXLOC_RESULT total={total} pass={passed} fail={failed}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
