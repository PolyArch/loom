#!/usr/bin/env python3
"""Regression test for legacy app corpus inventory evidence."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import artifact_test_common


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


def read_rows(output: Path) -> list[dict[str, str]]:
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise AssertionError(f"unexpected inventory header {reader.fieldnames}")
        return list(reader)


def write_case(root: Path, name: str, *, with_header: bool = True) -> None:
    case_dir = root / name
    case_dir.mkdir(parents=True)
    (case_dir / "main.cpp").write_text("int main() { return 0; }\n")
    (case_dir / f"{name}.cpp").write_text(f'#include "{name}.h"\n')
    if with_header:
        (case_dir / f"{name}.h").write_text("#pragma once\n")


def run_inventory(repo: Path, source_root: Path, output: Path) -> list[dict[str, str]]:
    artifact_test_common.require_success(
        repo,
        [
            "python3",
            "test/app/old_app_corpus_inventory.py",
            "--source-root",
            str(source_root),
            "--output",
            str(output),
        ],
        "old app corpus inventory",
    )
    return read_rows(output)


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="loom-old-app-inventory-") as tmp:
        source_root = Path(tmp) / "old-app"
        write_case(source_root, "alpha")
        write_case(source_root, "beta", with_header=False)
        output = Path(tmp) / "inventory.csv"
        rows = run_inventory(repo, source_root, output)

        if [row["case"] for row in rows] != ["alpha", "beta"]:
            raise AssertionError(f"inventory rows are not sorted by case: {rows}")
        alpha = rows[0]
        if alpha["status"] != "ready" or alpha["source_count"] != "2" or alpha["header_count"] != "1":
            raise AssertionError(f"alpha should be a ready two-source case: {alpha}")
        beta = rows[1]
        if beta["status"] != "blocked" or "missing header" not in beta["diagnostic"]:
            raise AssertionError(f"beta should report its missing header: {beta}")

    local_old = repo / "temp" / "old_implementation_loom" / "loom" / "tests" / "app"
    if local_old.is_dir():
        with tempfile.TemporaryDirectory(prefix="loom-old-app-local-") as tmp:
            output = Path(tmp) / "inventory.csv"
            rows = run_inventory(repo, local_old, output)
            if len(rows) != 127:
                raise AssertionError(f"expected 127 local old app cases, got {len(rows)}")
            cases = {row["case"] for row in rows}
            for required in ("axpy", "matmul", "spmv", "vecadd"):
                if required not in cases:
                    raise AssertionError(f"missing expected local old app case {required}")
            if any(row["status"] != "ready" for row in rows):
                raise AssertionError(f"local old app inventory contains blocked rows: {rows}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
