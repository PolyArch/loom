#!/usr/bin/env python3
"""Regression test for simulator cycle summary workload rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


HEADER = ["kernel", "dfg_sim_cycles", "cgra_sim_cycles"]


def write_blocked_mapping_artifact(path: Path, workload: str) -> None:
    artifact = {
        "schema_version": 1,
        "kind": "pnr_mapping",
        "workload": workload,
        "graph": f"g_{workload}",
        "hardware": "blocked_adg",
        "mapping_id": f"{workload}__blocked_adg",
        "status": "fail",
        "placed_records": 0,
        "routed_edges": 0,
        "unrouted_edges": 1,
        "unplaced_records": 0,
        "config_records": 0,
        "placements": [],
        "routes": [],
        "config_bitstream": [],
        "diagnostics": ["structured test mapping blocks CGRA-sim"],
    }
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def run_discovered_report_pair(repo: Path, evidence_dir: Path, workload: str, upper_bound: str) -> list[Path]:
    dfg_tool = repo / "build/tools/loom-dfg-sim/loom-dfg-sim"
    if not dfg_tool.is_file():
        dfg_tool = repo / "build/bin/loom-dfg-sim"
    cgra_tool = repo / "build/tools/loom-cgra-sim/loom-cgra-sim"
    if not cgra_tool.is_file():
        cgra_tool = repo / "build/bin/loom-cgra-sim"

    dfg_report = evidence_dir / f"{workload}-dfg-sim-report.json"
    mapping_artifact = evidence_dir / f"{workload}-pnr-mapping.json"
    cgra_report = evidence_dir / f"{workload}-cgra-sim-report.json"
    artifact_test_common.require_success(
        repo,
        [
            str(dfg_tool),
            "test/simulator/dfg_basic.mlir",
            "--graph",
            "sum4",
            "--workload",
            workload,
            "--arg",
            "0=none",
            "--arg",
            "1=0",
            "--arg",
            f"2={upper_bound}",
            "--arg",
            "3=1",
            "--arg",
            "4=0.000000e+00",
            "--output",
            str(dfg_report),
        ],
        f"{workload} DFG simulation report",
    )
    write_blocked_mapping_artifact(mapping_artifact, workload)
    artifact_test_common.require_success(
        repo,
        [
            str(cgra_tool),
            "--dfg-report",
            str(dfg_report),
            "--mapping-artifact",
            str(mapping_artifact),
            "--output",
            str(cgra_report),
        ],
        f"{workload} CGRA simulation report",
    )
    return [dfg_report, mapping_artifact, cgra_report]


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
            raise AssertionError(f"default sim cycle row should pass with routed CGRA evidence: {default_row}")
        if default_row["dfg_sim_cycles"] != "579":
            raise AssertionError(f"default sim cycle row should include vecsum DFG-sim evidence: {default_row}")
        if default_row["cgra_sim_cycles"] != "591":
            raise AssertionError(f"default sim cycle row should include vecsum CGRA-sim evidence: {default_row}")
        if "DFG-sim and CGRA-sim reports available" not in default_row.get("diagnostic", ""):
            raise AssertionError(f"default sim cycle row should report available simulator evidence: {default_row}")

        evidence_dir = out_dir / "current-sim-cycle"
        evidence_dir.mkdir()
        discovered_artifacts = [
            *run_discovered_report_pair(repo, evidence_dir, "sum4", "4"),
            *run_discovered_report_pair(repo, evidence_dir, "sum8", "8"),
        ]
        discovered_sim = out_dir / "discovered-sim-cycle-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_sim_cycle_summary.sh",
                "--output",
                str(discovered_sim),
            ],
            "discovered sim cycle summary",
        )
        discovered_rows = artifact_test_common.read_csv_rows(discovered_sim, HEADER)
        discovered_by_kernel = {row["kernel"]: row for row in discovered_rows}
        if set(discovered_by_kernel) != {"sum4", "sum8"}:
            raise AssertionError(f"expected discovered evidence rows only, got {discovered_rows}")
        if discovered_by_kernel["sum4"]["dfg_sim_cycles"] != "28":
            raise AssertionError(f"sum4 should keep DFG cycles from report: {discovered_by_kernel['sum4']}")
        if discovered_by_kernel["sum8"]["dfg_sim_cycles"] != "48":
            raise AssertionError(f"sum8 should keep DFG cycles from report: {discovered_by_kernel['sum8']}")
        for kernel, discovered_row in discovered_by_kernel.items():
            if discovered_row["cgra_sim_cycles"] != "":
                raise AssertionError(f"{kernel} must not expose blocked CGRA cycles: {discovered_row}")
            if discovered_row.get("status") != "blocked":
                raise AssertionError(f"{kernel} should keep structured blocked status: {discovered_row}")
            if "mapping artifact status fail blocks CGRA-sim" not in discovered_row.get("diagnostic", ""):
                raise AssertionError(f"{kernel} should carry CGRA blocked diagnostic: {discovered_row}")
        discovered_audit = out_dir / "sim-cycle-summary-discovered-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(discovered_audit),
                str(discovered_sim),
                *(str(path) for path in discovered_artifacts),
            ],
            "discovered sim cycle summary artifact audit",
        )
        discovered_audit_data = json.loads(discovered_audit.read_text())
        if discovered_audit_data.get("verdict") != "pass":
            raise AssertionError(f"discovered sim cycle audit should pass: {discovered_audit_data}")

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
