#!/usr/bin/env python3
"""Regression test for legacy app corpus import status evidence."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import artifact_test_common


HEADER = [
    "case",
    "import_state",
    "manifest_case",
    "reason",
    "owner",
]


def write_inventory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "main_source",
                "implementation_sources",
                "headers",
                "source_count",
                "header_count",
                "status",
                "diagnostic",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case": "legacy_missing",
                "main_source": "main.cpp",
                "implementation_sources": "legacy_missing.cpp",
                "headers": "legacy_missing.h",
                "source_count": "2",
                "header_count": "1",
                "status": "ready",
                "diagnostic": "ready for migration",
            }
        )
        writer.writerow(
            {
                "case": "vecadd",
                "main_source": "main.cpp",
                "implementation_sources": "vecadd.cpp",
                "headers": "vecadd.h",
                "source_count": "2",
                "header_count": "1",
                "status": "ready",
                "diagnostic": "ready for migration",
            }
        )


def read_rows(output: Path) -> list[dict[str, str]]:
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise AssertionError(f"unexpected app import status header {reader.fieldnames}")
        return list(reader)


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-app-import-status-") as tmp:
        temp_root = Path(tmp)
        inventory = temp_root / "inventory.csv"
        output = temp_root / "import-status.csv"
        write_inventory(inventory)
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/app/app_import_status.py",
                "--inventory",
                str(inventory),
                "--manifest",
                "test/app/manifest.json",
                "--output",
                str(output),
            ],
            "app import status",
        )
        rows = read_rows(output)
        if [row["case"] for row in rows] != ["legacy_missing", "vecadd"]:
            raise AssertionError(f"import status must preserve inventory case order: {rows}")
        by_case = {row["case"]: row for row in rows}
        if by_case["vecadd"]["import_state"] != "accepted" or by_case["vecadd"]["manifest_case"] != "vecadd":
            raise AssertionError(f"vecadd should be accepted by manifest: {by_case['vecadd']}")
        missing = by_case["legacy_missing"]
        if missing["import_state"] != "deferred":
            raise AssertionError(f"unimported legacy case should be deferred: {missing}")
        if "not listed in app manifest" not in missing["reason"]:
            raise AssertionError(f"deferred case should explain manifest gap: {missing}")

        bad_inventory = temp_root / "bad-inventory.csv"
        bad_inventory.write_text("case,status\nvecadd,ready\nvecadd,ready\n")
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/app/app_import_status.py",
                "--inventory",
                str(bad_inventory),
                "--manifest",
                "test/app/manifest.json",
                "--output",
                str(temp_root / "bad-status.csv"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("duplicate inventory case unexpectedly passed")
        if "duplicate inventory case" not in result.stderr:
            raise AssertionError(f"duplicate inventory diagnostic missing: {result.stderr}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
