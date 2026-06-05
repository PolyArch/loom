#!/usr/bin/env python3
"""Emit a CSV inventory for a legacy Loom app corpus."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


HEADER = [
    "case",
    "main_source",
    "implementation_sources",
    "headers",
    "source_count",
    "header_count",
    "status",
    "diagnostic",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def case_directories(source_root: Path) -> list[Path]:
    return sorted(path for path in source_root.iterdir() if path.is_dir())


def inventory_row(case_dir: Path) -> dict[str, str]:
    sources = sorted(path.name for path in case_dir.glob("*.cpp"))
    headers = sorted(path.name for path in case_dir.glob("*.h"))
    main_source = "main.cpp" if "main.cpp" in sources else ""
    implementation_sources = [source for source in sources if source != "main.cpp"]

    diagnostics: list[str] = []
    if not main_source:
        diagnostics.append("missing main.cpp")
    if not implementation_sources:
        diagnostics.append("missing implementation source")
    if not headers:
        diagnostics.append("missing header")

    status = "blocked" if diagnostics else "ready"
    diagnostic = "; ".join(diagnostics) if diagnostics else "ready for migration"
    return {
        "case": case_dir.name,
        "main_source": main_source,
        "implementation_sources": ";".join(implementation_sources),
        "headers": ";".join(headers),
        "source_count": str(len(sources)),
        "header_count": str(len(headers)),
        "status": status,
        "diagnostic": diagnostic,
    }


def write_inventory(source_root: Path, output: Path) -> int:
    if not source_root.is_dir():
        print(f"missing source root: {source_root}", file=sys.stderr)
        return 1
    rows = [inventory_row(case_dir) for case_dir in case_directories(source_root)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        print(f"no app case directories under {source_root}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return write_inventory(Path(args.source_root), Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
