#!/usr/bin/env python3
"""Regression test for unsupported-scope ledger extraction."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import artifact_test_common


HEADER = ["stage", "case", "artifact", "reason", "owner", "blocking_input"]


def run(repo: Path, argv: list[str]) -> None:
    result = subprocess.run(
        argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed with {result.returncode}: {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-unsupported-ledger-") as tmp:
        out_dir = Path(tmp)
        primitive = out_dir / "dataflow-primitive-coverage.csv"
        ledger = out_dir / "unsupported-scope-ledger.csv"

        run(
            repo,
            [
                "bash",
                "test/dataflow/run_primitive_coverage.sh",
                "--case",
                "vecadd",
                "--output",
                str(primitive),
            ],
        )
        run(
            repo,
            [
                "bash",
                "test/e2e/run_unsupported_scope_ledger.sh",
                "--artifact",
                str(primitive),
                "--output",
                str(ledger),
            ],
        )

        with ledger.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if reader.fieldnames[: len(HEADER)] != HEADER:
                raise AssertionError(f"unexpected header: {reader.fieldnames}")
        if not rows:
            raise AssertionError("expected blocked primitive rows in unsupported ledger")
        vecadd_stream = [
            row for row in rows
            if row["stage"] == "dfg_sim_status"
            and row["case"] == "vecadd:stream"
            and row["artifact"] == "dataflow_primitive_coverage"
        ]
        if len(vecadd_stream) != 1:
            raise AssertionError(f"expected one vecadd stream row, got {rows}")
        row = vecadd_stream[0]
        if "blocked" not in row["reason"]:
            raise AssertionError(f"reason should record blocked status: {row}")
        if row["owner"] != "implementation":
            raise AssertionError(f"owner should be implementation: {row}")
        if not row["blocking_input"].endswith("dataflow-primitive-coverage.csv"):
            raise AssertionError(f"blocking input should name source artifact: {row}")

        blocked = out_dir / "blocked-dataflow-primitive-coverage.csv"
        blocked.write_text(
            "workload,primitive,op_count,dfg_sim_status,diagnostic\n"
            "vecsum,stream,2,blocked,simulator evidence missing\n"
        )
        passed = out_dir / "passed-dataflow-primitive-coverage.csv"
        passed.write_text(
            "workload,primitive,op_count,dfg_sim_status,diagnostic\n"
            "vecsum,stream,2,pass,DFG-sim report produced operation fire counts\n"
        )
        superseded = out_dir / "superseded-unsupported-scope-ledger.csv"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_unsupported_scope_ledger.sh",
                "--artifact",
                str(blocked),
                "--artifact",
                str(passed),
                "--output",
                str(superseded),
            ],
        )
        with superseded.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        stale_rows = [
            row for row in rows
            if row["stage"] == "dfg_sim_status"
            and row["case"] == "vecsum:stream"
            and row["artifact"] == "dataflow_primitive_coverage"
        ]
        if stale_rows:
            raise AssertionError(f"later pass evidence should close earlier unsupported rows: {rows}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
