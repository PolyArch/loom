#!/usr/bin/env python3
"""Regression test for legacy app corpus inventory evidence."""

from __future__ import annotations

import csv
import argparse
import sys
from pathlib import Path

import artifact_test_common


HEADER = [
    "case",
    "main_source",
    "implementation_sources",
    "headers",
    "source_count",
    "header_count",
    "feature_tags",
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("--legacy-root")
    parser.add_argument("--expect-count", type=int)
    parser.add_argument("--expect-case", action="append", default=[])
    return parser.parse_args(argv)


def validate_synthetic_inventory(repo: Path) -> None:
    with artifact_test_common.repo_temp_dir(repo, "loom-old-app-inventory-") as tmp:
        source_root = Path(tmp) / "old-app"
        write_case(source_root, "matmul")
        write_case(source_root, "popcount")
        write_case(source_root, "sort_quick")
        write_case(source_root, "spmv")
        write_case(source_root, "string_hash")
        write_case(source_root, "bitonic_stage")
        write_case(source_root, "beta", with_header=False)
        output = Path(tmp) / "inventory.csv"
        rows = run_inventory(repo, source_root, output)

        if [row["case"] for row in rows] != [
            "beta",
            "bitonic_stage",
            "matmul",
            "popcount",
            "sort_quick",
            "spmv",
            "string_hash",
        ]:
            raise AssertionError(f"inventory rows are not sorted by case: {rows}")
        by_case = {row["case"]: row for row in rows}
        matmul = by_case["matmul"]
        if matmul["status"] != "ready" or matmul["source_count"] != "2" or matmul["header_count"] != "1":
            raise AssertionError(f"matmul should be a ready two-source case: {matmul}")
        expected_tags = {
            "matmul": {"matrix", "numeric"},
            "popcount": {"bit", "integer"},
            "sort_quick": {"sort", "integer"},
            "spmv": {"sparse", "matrix"},
            "string_hash": {"string", "hash"},
        }
        for case, tags in expected_tags.items():
            actual = set(filter(None, by_case[case]["feature_tags"].split(";")))
            if not tags.issubset(actual):
                raise AssertionError(f"{case} tags {actual} do not include {tags}")
        bitonic_tags = set(filter(None, by_case["bitonic_stage"]["feature_tags"].split(";")))
        if "bit" in bitonic_tags:
            raise AssertionError(f"bitonic_stage tags should not include bit: {bitonic_tags}")
        beta = by_case["beta"]
        if beta["status"] != "blocked" or "missing header" not in beta["diagnostic"]:
            raise AssertionError(f"beta should report its missing header: {beta}")


def validate_legacy_inventory(
    repo: Path,
    legacy_root: Path,
    expected_count: int | None,
    expected_cases: list[str],
) -> None:
    if not legacy_root.is_dir():
        raise AssertionError(f"legacy app corpus root does not exist: {legacy_root}")
    with artifact_test_common.repo_temp_dir(repo, "loom-old-app-local-") as tmp:
        output = Path(tmp) / "inventory.csv"
        rows = run_inventory(repo, legacy_root, output)
        if expected_count is not None and len(rows) != expected_count:
            raise AssertionError(f"expected {expected_count} legacy app cases, got {len(rows)}")
        cases = {row["case"] for row in rows}
        for required in expected_cases:
            if required not in cases:
                raise AssertionError(f"missing expected legacy app case {required}")
        if any(row["status"] != "ready" for row in rows):
            raise AssertionError(f"legacy app inventory contains blocked rows: {rows}")


def main() -> int:
    args = parse_args(sys.argv[1:])
    repo = Path(args.repo).resolve()
    validate_synthetic_inventory(repo)
    if args.legacy_root:
        validate_legacy_inventory(repo, Path(args.legacy_root), args.expect_count, args.expect_case)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
