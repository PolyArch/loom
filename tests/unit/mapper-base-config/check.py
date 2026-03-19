#!/usr/bin/env python3

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check.py <case-name> <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(sys.argv[2])
    run_err = output_dir / "run.err"
    text = run_err.read_text(encoding="utf-8")
    expected = (
        "CLI option --mapper-routing-heuristic-weight overrides mapper base "
        "config value"
    )
    if expected not in text:
        print("missing mapper override warning", file=sys.stderr)
        return 1

    map_json = output_dir / "dfg.fabric.map.json"
    if not map_json.exists():
        print("mapping artifact missing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
