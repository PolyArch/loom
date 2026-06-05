#!/usr/bin/env python3
"""Emit migration status rows for a legacy app corpus inventory."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "app"))

import app_manifest  # noqa: E402


HEADER = [
    "case",
    "import_state",
    "manifest_case",
    "reason",
    "owner",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def read_inventory(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    diagnostics: list[str] = []
    if not path.is_file():
        return [], [f"missing inventory: {path}"]
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        diagnostics.append("inventory is empty")
    seen: set[str] = set()
    for row in rows:
        case = row.get("case", "")
        if not case:
            diagnostics.append("inventory row has blank case")
            continue
        if case in seen:
            diagnostics.append(f"duplicate inventory case: {case}")
        seen.add(case)
    return rows, diagnostics


def manifest_cases(path: Path) -> tuple[set[str], list[str]]:
    data, diagnostics = app_manifest.validate_manifest(path)
    if diagnostics:
        return set(), diagnostics
    cases = data["cases"]
    assert isinstance(cases, list)
    return {str(entry["case"]) for entry in cases if isinstance(entry, dict)}, []


def import_rows(inventory: list[dict[str, str]], accepted_cases: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in inventory:
        case = row["case"]
        if case in accepted_cases:
            rows.append(
                {
                    "case": case,
                    "import_state": "accepted",
                    "manifest_case": case,
                    "reason": "listed in app manifest",
                    "owner": "test_migration",
                }
            )
        else:
            rows.append(
                {
                    "case": case,
                    "import_state": "deferred",
                    "manifest_case": "",
                    "reason": "not listed in app manifest",
                    "owner": "test_migration",
                }
            )
    return rows


def write_rows(output: Path, rows: list[dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    inventory, inventory_diagnostics = read_inventory(Path(args.inventory))
    accepted_cases, manifest_diagnostics = manifest_cases(Path(args.manifest))
    diagnostics = inventory_diagnostics + manifest_diagnostics
    if diagnostics:
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
        return 1
    write_rows(Path(args.output), import_rows(inventory, accepted_cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
