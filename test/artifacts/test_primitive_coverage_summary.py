#!/usr/bin/env python3
"""Regression test for dataflow primitive coverage summary evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import artifact_test_common


HEADER = ["workload", "primitive", "op_count", "dfg_sim_status", "diagnostic"]
EXPECTED_POSITIVE = {"stream", "carry", "load"}
EXPECTED_VECSUM_SIMULATED = {"stream", "carry", "load", "sync"}
EXPECTED_DOTPRODUCT_SIMULATED = {"stream", "carry", "load", "sync"}
EXPECTED_XOR_BLOCK_SIMULATED = {"load", "store", "sync"}
EXPECTED_XOR_BLOCK_STATIC_ONLY = {"stream", "carry"}


def run_summary(repo: Path, output: Path, *args: str) -> list[dict[str, str]]:
    return artifact_test_common.run_csv_summary(
        repo,
        "test/dataflow/run_primitive_coverage.sh",
        output,
        HEADER,
        *args,
        label="primitive coverage summary",
    )


def assert_vecadd_rows(rows: list[dict[str, str]]) -> None:
    by_primitive = {row["primitive"]: row for row in rows if row["workload"] == "vecadd"}
    missing = sorted(EXPECTED_POSITIVE - set(by_primitive))
    if missing:
        raise AssertionError(f"missing vecadd primitive rows: {missing}; rows={rows}")
    for primitive in sorted(EXPECTED_POSITIVE):
        row = by_primitive[primitive]
        if int(row["op_count"]) <= 0:
            raise AssertionError(f"vecadd {primitive} count is not positive: {row}")
        if row["dfg_sim_status"] != "blocked":
            raise AssertionError(f"vecadd {primitive} simulator status should be blocked: {row}")
        if "DFG-sim report is unavailable" not in row["diagnostic"]:
            raise AssertionError(f"unexpected diagnostic for {primitive}: {row}")


def assert_vecsum_simulated_rows(rows: list[dict[str, str]]) -> None:
    by_primitive = {row["primitive"]: row for row in rows if row["workload"] == "vecsum"}
    missing = sorted(EXPECTED_VECSUM_SIMULATED - set(by_primitive))
    if missing:
        raise AssertionError(f"missing vecsum primitive rows: {missing}; rows={rows}")
    for primitive in sorted(EXPECTED_VECSUM_SIMULATED):
        row = by_primitive[primitive]
        if int(row["op_count"]) <= 0:
            raise AssertionError(f"vecsum {primitive} count is not positive: {row}")
        if row["dfg_sim_status"] != "pass":
            raise AssertionError(f"vecsum {primitive} should have DFG-sim pass evidence: {row}")
        if "DFG-sim report" not in row["diagnostic"]:
            raise AssertionError(f"unexpected vecsum diagnostic for {primitive}: {row}")


def assert_dotproduct_simulated_rows(rows: list[dict[str, str]]) -> None:
    by_primitive = {row["primitive"]: row for row in rows if row["workload"] == "dotproduct"}
    missing = sorted(EXPECTED_DOTPRODUCT_SIMULATED - set(by_primitive))
    if missing:
        raise AssertionError(f"missing dotproduct primitive rows: {missing}; rows={rows}")
    for primitive in sorted(EXPECTED_DOTPRODUCT_SIMULATED):
        row = by_primitive[primitive]
        if int(row["op_count"]) <= 0:
            raise AssertionError(f"dotproduct {primitive} count is not positive: {row}")
        if row["dfg_sim_status"] != "pass":
            raise AssertionError(f"dotproduct {primitive} should have DFG-sim pass evidence: {row}")
        if "DFG-sim report" not in row["diagnostic"]:
            raise AssertionError(f"unexpected dotproduct diagnostic for {primitive}: {row}")


def assert_xor_block_simulated_rows(rows: list[dict[str, str]]) -> None:
    by_primitive = {row["primitive"]: row for row in rows if row["workload"] == "xor_block"}
    expected_primitives = EXPECTED_XOR_BLOCK_SIMULATED | EXPECTED_XOR_BLOCK_STATIC_ONLY
    missing = sorted(expected_primitives - set(by_primitive))
    if missing:
        raise AssertionError(f"missing xor_block primitive rows: {missing}; rows={rows}")
    for primitive in sorted(EXPECTED_XOR_BLOCK_SIMULATED):
        row = by_primitive[primitive]
        if int(row["op_count"]) <= 0:
            raise AssertionError(f"xor_block {primitive} count is not positive: {row}")
        if row["dfg_sim_status"] != "pass":
            raise AssertionError(f"xor_block {primitive} should have DFG-sim pass evidence: {row}")
        if "DFG-sim report" not in row["diagnostic"]:
            raise AssertionError(f"unexpected xor_block diagnostic for {primitive}: {row}")
    for primitive in sorted(EXPECTED_XOR_BLOCK_STATIC_ONLY):
        row = by_primitive[primitive]
        if int(row["op_count"]) <= 0:
            raise AssertionError(f"xor_block {primitive} static count is not positive: {row}")
        if row["dfg_sim_status"] != "blocked":
            raise AssertionError(f"xor_block {primitive} should stay static-only evidence: {row}")
        if "op-count coverage only" not in row["diagnostic"]:
            raise AssertionError(f"unexpected xor_block static-only diagnostic for {primitive}: {row}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-primitive-coverage-") as tmp:
        output = Path(tmp) / "dataflow-primitive-coverage.csv"
        assert_vecadd_rows(run_summary(repo, output, "--case", "vecadd"))

        vecsum_output = Path(tmp) / "dataflow-primitive-coverage-vecsum.csv"
        assert_vecsum_simulated_rows(run_summary(repo, vecsum_output, "--case", "vecsum"))

        dotproduct_output = Path(tmp) / "dataflow-primitive-coverage-dotproduct.csv"
        assert_dotproduct_simulated_rows(run_summary(repo, dotproduct_output, "--case", "dotproduct"))

        xor_block_output = Path(tmp) / "dataflow-primitive-coverage-xor-block.csv"
        assert_xor_block_simulated_rows(run_summary(repo, xor_block_output, "--case", "xor_block"))

        default_output = Path(tmp) / "dataflow-primitive-coverage-default.csv"
        rows = run_summary(repo, default_output)
        expected_cases = {
            path.name
            for path in (repo / "test" / "app").iterdir()
            if (path / "dfg_check.sh").is_file()
        }
        actual_cases = {row["workload"] for row in rows}
        if expected_cases != actual_cases:
            raise AssertionError(f"default cases {actual_cases} do not match {expected_cases}")
        assert_vecadd_rows(rows)
        assert_vecsum_simulated_rows(rows)
        assert_dotproduct_simulated_rows(rows)
        assert_xor_block_simulated_rows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
