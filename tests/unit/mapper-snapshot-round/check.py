#!/usr/bin/env python3

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check.py <case-name> <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(sys.argv[2])
    snapshot_dir = output_dir / "mapper-snapshots"
    if not snapshot_dir.is_dir():
        print("mapper-snapshots directory missing", file=sys.stderr)
        return 1

    html_files = sorted(snapshot_dir.glob("*.viz.html"))
    map_files = sorted(snapshot_dir.glob("*.map.json"))
    if not html_files:
        print("snapshot HTML missing", file=sys.stderr)
        return 1
    if not map_files:
        print("snapshot map.json missing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
