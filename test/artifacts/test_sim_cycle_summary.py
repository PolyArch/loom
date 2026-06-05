#!/usr/bin/env python3
"""Regression test for simulator cycle summary blocked workload rows."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


HEADER = ["kernel", "dfg_sim_cycles", "cgra_sim_cycles"]


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
    with tempfile.TemporaryDirectory(prefix="loom-sim-cycle-") as tmp:
        out_dir = Path(tmp)
        primitive = out_dir / "dataflow-primitive-coverage.csv"
        sim = out_dir / "sim-cycle-summary.csv"
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
                "test/app/run_sim_cycle_summary.sh",
                "--primitive-coverage",
                str(primitive),
                "--output",
                str(sim),
            ],
        )
        with sim.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if reader.fieldnames[: len(HEADER)] != HEADER:
                raise AssertionError(f"unexpected header: {reader.fieldnames}")
        vecadd_rows = [row for row in rows if row["kernel"] == "vecadd"]
        if len(vecadd_rows) != 1:
            raise AssertionError(f"expected one vecadd row, got {rows}")
        row = vecadd_rows[0]
        if row["dfg_sim_cycles"] != "" or row["cgra_sim_cycles"] != "":
            raise AssertionError(f"missing simulator evidence must stay empty, not numeric: {row}")
        if row.get("status") != "blocked":
            raise AssertionError(f"sim cycle row should be blocked: {row}")
        if "DFG-sim and CGRA-sim cycle evidence is not available yet" not in row.get("diagnostic", ""):
            raise AssertionError(f"unexpected diagnostic: {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
