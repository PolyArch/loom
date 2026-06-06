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
        default_sim = out_dir / "sim-cycle-summary-default.csv"
        primitive = out_dir / "dataflow-primitive-coverage.csv"
        sim = out_dir / "sim-cycle-summary.csv"
        dfg_report = out_dir / "dfg-sim-report.json"
        sim_from_dfg = out_dir / "sim-cycle-summary-from-dfg.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_sim_cycle_summary.sh",
                "--output",
                str(default_sim),
            ],
            "default sim cycle summary",
        )
        default_rows = artifact_test_common.read_csv_rows(default_sim, HEADER)
        vecsum_default_rows = [row for row in default_rows if row["kernel"] == "vecsum"]
        if len(vecsum_default_rows) != 1:
            raise AssertionError(f"expected one default vecsum row, got {default_rows}")
        default_row = vecsum_default_rows[0]
        if default_row.get("status") != "pass":
            raise AssertionError(f"default sim cycle row should be pass evidence: {default_row}")
        if default_row["dfg_sim_cycles"] == "" or default_row["cgra_sim_cycles"] == "":
            raise AssertionError(f"default sim cycle row should include both simulators: {default_row}")
        if int(default_row["cgra_sim_cycles"]) < int(default_row["dfg_sim_cycles"]):
            raise AssertionError(f"CGRA-sim should not be more optimistic than DFG-sim: {default_row}")

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

        dfg_tool = repo / "build/tools/loom-dfg-sim/loom-dfg-sim"
        if not dfg_tool.is_file():
            dfg_tool = repo / "build/bin/loom-dfg-sim"
        artifact_test_common.require_success(
            repo,
            [
                str(dfg_tool),
                "test/simulator/dfg_basic.mlir",
                "--graph",
                "sum4",
                "--arg",
                "0=none",
                "--arg",
                "1=0",
                "--arg",
                "2=4",
                "--arg",
                "3=1",
                "--arg",
                "4=0.000000e+00",
                "--output",
                str(dfg_report),
            ],
            "DFG simulation report",
        )
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_sim_cycle_summary.sh",
                "--dfg-report",
                str(dfg_report),
                "--output",
                str(sim_from_dfg),
            ],
            "sim cycle summary from DFG report",
        )
        dfg_rows = artifact_test_common.read_csv_rows(sim_from_dfg, HEADER)
        sum4_rows = [row for row in dfg_rows if row["kernel"] == "sum4"]
        if len(sum4_rows) != 1:
            raise AssertionError(f"expected one sum4 row, got {dfg_rows}")
        dfg_row = sum4_rows[0]
        if dfg_row["dfg_sim_cycles"] != "28":
            raise AssertionError(f"DFG report should fill DFG cycles: {dfg_row}")
        if dfg_row["cgra_sim_cycles"] != "":
            raise AssertionError(f"CGRA-sim cycles require mapping and Fabric evidence: {dfg_row}")
        if dfg_row.get("status") != "blocked":
            raise AssertionError(f"row should stay blocked until CGRA-sim exists: {dfg_row}")
        if "CGRA-sim requires Fabric ADG and mapping artifact evidence" not in dfg_row.get("diagnostic", ""):
            raise AssertionError(f"unexpected DFG diagnostic: {dfg_row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
