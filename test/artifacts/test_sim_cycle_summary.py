#!/usr/bin/env python3
"""Regression test for simulator cycle summary workload rows."""

from __future__ import annotations

import sys
from pathlib import Path

import artifact_test_common


HEADER = ["kernel", "dfg_sim_cycles", "cgra_sim_cycles"]


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-sim-cycle-") as tmp:
        out_dir = Path(tmp)
        primitive = out_dir / "dataflow-primitive-coverage.csv"
        sim = out_dir / "sim-cycle-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/dataflow/run_primitive_coverage.sh",
                "--case",
                "vecadd",
                "--output",
                str(primitive),
            ],
            "primitive coverage summary",
        )
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_sim_cycle_summary.sh",
                "--primitive-coverage",
                str(primitive),
                "--output",
                str(sim),
            ],
            "sim cycle summary",
        )
        rows = artifact_test_common.read_csv_rows(sim, HEADER)
        vecadd_rows = [row for row in rows if row["kernel"] == "vecadd"]
        if len(vecadd_rows) != 1:
            raise AssertionError(f"expected one vecadd row, got {rows}")
        row = vecadd_rows[0]
        if row["dfg_sim_cycles"] != "":
            raise AssertionError(f"DFG-sim cycles require a DFG-sim report: {row}")
        if row["cgra_sim_cycles"] != "":
            raise AssertionError(f"CGRA-sim cycles require mapping and Fabric evidence: {row}")
        if row.get("status") != "blocked":
            raise AssertionError(f"sim cycle row should stay blocked until simulator reports exist: {row}")
        if "primitive-count proxy only; DFG-sim report unavailable" not in row.get("diagnostic", ""):
            raise AssertionError(f"unexpected diagnostic: {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
