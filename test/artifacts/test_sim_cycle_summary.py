#!/usr/bin/env python3
"""Regression test for simulator cycle summary workload rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


HEADER = ["kernel", "dfg_sim_cycles", "cgra_sim_cycles"]


def config_fingerprint(repo: Path) -> str:
    tool = repo / "build/tools/loom-config-test/loom-config-test"
    if not tool.is_file():
        tool = repo / "build/bin/loom-config-test"
    result = artifact_test_common.require_success(
        repo,
        [str(tool), "--resolved-fingerprint"],
        "resolved config fingerprint",
    )
    return result.stdout.strip()


def component_config_fingerprint(repo: Path, view: str) -> str:
    tool = repo / "build/tools/loom-config-test/loom-config-test"
    if not tool.is_file():
        tool = repo / "build/bin/loom-config-test"
    result = artifact_test_common.require_success(
        repo,
        [
            str(tool),
            "--component-fingerprint",
            "--component-view",
            view,
        ],
        f"{view} component config fingerprint",
    )
    return result.stdout.strip()


def write_blocked_mapping_artifact(path: Path, repo: Path, workload: str) -> None:
    artifact = {
        "schema_version": 1,
        "kind": "pnr_mapping",
        "workload": workload,
        "graph": f"g_{workload}",
        "hardware": "blocked_adg",
        "mapping_id": f"{workload}__blocked_adg",
        "config_id": "loom.default",
        "config_fingerprint": config_fingerprint(repo),
        "component_config_view": "pnr.mapping.v1",
        "component_config_fingerprint": component_config_fingerprint(
            repo, "pnr.mapping.v1"
        ),
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
    write_blocked_mapping_artifact(mapping_artifact, repo, workload)
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

        stale_evidence_dir = out_dir / "stale-current-sim-cycle" / "current-sim-cycle"
        stale_evidence_dir.mkdir(parents=True)
        stale_dfg = stale_evidence_dir / "stale-dfg-sim-report.json"
        stale_cgra = stale_evidence_dir / "stale-cgra-sim-report.json"
        stale_dfg.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "dfg_sim_report",
                    "workload": "stale",
                    "graph": "g_stale",
                    "status": "pass",
                    "metric_definition": "fixture",
                    "operation_semantics_source": "loom.sim.operation_semantics.v1",
                    "operation_cost_model_source": "loom.sim.operation_cost.v1",
                    "optimistic_cycles": 10,
                    "wavefront_steps": 1,
                    "event_count": 1,
                    "dynamic_work_items": 1,
                    "operation_fire_counts": {"arith.addi": 1},
                    "final_outputs": ["i32:3"],
                    "final_memory_state": {},
                    "diagnostics": [],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        stale_cgra.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "cgra_sim_report",
                    "workload": "stale",
                    "hardware": "shared_reduction_adg",
                    "mapping_id": "stale__shared_reduction_adg",
                    "status": "pass",
                    "fidelity_level": "mapping_constraint_estimate",
                    "metric_definition": "mapping_constraint_estimate",
                    "operation_semantics_source": "loom.sim.operation_semantics.v1",
                    "operation_cost_model_source": "loom.sim.operation_cost.v1",
                    "difference_classification": "expected_hardware_constraint",
                    "hardware_bound_classification": "within_modeled_bounds",
                    "dfg_cycles": 10,
                    "modeled_lower_bound_cycles": 12,
                    "performance_delta_cycles": 2,
                    "route_latency_cycles": 2,
                    "memory_latency_cycles": 0,
                    "temporal_penalty_cycles": 0,
                    "hardware_aware_cycles": 12,
                    "placed_records": 1,
                    "routed_edges": 1,
                    "route_segments": 1,
                    "config_records": 1,
                    "spatial_placements": 1,
                    "temporal_placements": 0,
                    "cycle_breakdown": [],
                    "unmodeled_constraints": [],
                    "first_principles_checks": [],
                    "diagnostics": [],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        stale_sim = stale_evidence_dir.parent / "stale-sim-cycle-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_sim_cycle_summary.sh",
                "--output",
                str(stale_sim),
            ],
            "stale discovered sim cycle summary",
        )
        stale_rows = artifact_test_common.read_csv_rows(stale_sim, HEADER)
        stale_by_kernel = {row["kernel"]: row for row in stale_rows}
        if set(stale_by_kernel) != {"stale"}:
            raise AssertionError(f"expected one stale evidence row, got {stale_rows}")
        stale_row = stale_by_kernel["stale"]
        if (
            stale_row.get("status") != "blocked"
            or stale_row["dfg_sim_cycles"] != ""
            or stale_row["cgra_sim_cycles"] != ""
        ):
            raise AssertionError(f"invalid discovered evidence must not be summarized as CGRA pass: {stale_row}")
        if "discovered simulator evidence failed artifact audit" not in stale_row.get("diagnostic", ""):
            raise AssertionError(f"stale evidence diagnostic should name artifact audit failure: {stale_row}")
        stale_audit = stale_evidence_dir.parent / "stale-sim-cycle-summary-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_audit),
                str(stale_sim),
            ],
            "stale sim cycle summary artifact audit",
        )

        malformed_evidence_dir = out_dir / "malformed-current-sim-cycle" / "current-sim-cycle"
        malformed_evidence_dir.mkdir(parents=True)
        malformed_dfg = malformed_evidence_dir / "malformed-dfg-sim-report.json"
        malformed_cgra = malformed_evidence_dir / "malformed-cgra-sim-report.json"
        malformed_dfg.write_text(stale_dfg.read_text().replace('"stale"', '"malformed"'))
        malformed_cgra.write_text("{not-json\n")
        malformed_sim = malformed_evidence_dir.parent / "malformed-sim-cycle-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_sim_cycle_summary.sh",
                "--output",
                str(malformed_sim),
            ],
            "malformed discovered sim cycle summary",
        )
        malformed_rows = artifact_test_common.read_csv_rows(malformed_sim, HEADER)
        malformed_by_kernel = {row["kernel"]: row for row in malformed_rows}
        if set(malformed_by_kernel) != {"malformed"}:
            raise AssertionError(f"expected one malformed evidence row, got {malformed_rows}")
        malformed_row = malformed_by_kernel["malformed"]
        if malformed_row.get("status") != "blocked" or malformed_row["cgra_sim_cycles"] != "":
            raise AssertionError(f"malformed discovered evidence must become blocked: {malformed_row}")
        if "discovered simulator evidence failed artifact audit" not in malformed_row.get("diagnostic", ""):
            raise AssertionError(f"malformed evidence diagnostic should name artifact audit failure: {malformed_row}")

        missing_kind_evidence_dir = out_dir / "missing-kind-current-sim-cycle" / "current-sim-cycle"
        missing_kind_evidence_dir.mkdir(parents=True)
        missing_kind_cgra = missing_kind_evidence_dir / "missing-kind-cgra-sim-report.json"
        missing_kind_cgra.write_text(
            json.dumps({"schema_version": 1, "status": "pass"}, indent=2, sort_keys=True) + "\n"
        )
        missing_kind_sim = missing_kind_evidence_dir.parent / "missing-kind-sim-cycle-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_sim_cycle_summary.sh",
                "--output",
                str(missing_kind_sim),
            ],
            "missing-kind discovered sim cycle summary",
        )
        missing_kind_rows = artifact_test_common.read_csv_rows(missing_kind_sim, HEADER)
        missing_kind_by_kernel = {row["kernel"]: row for row in missing_kind_rows}
        if set(missing_kind_by_kernel) != {"missing-kind"}:
            raise AssertionError(f"fallback workload should strip report suffixes: {missing_kind_rows}")
        missing_kind_row = missing_kind_by_kernel["missing-kind"]
        if missing_kind_row.get("status") != "blocked" or missing_kind_row["cgra_sim_cycles"] != "":
            raise AssertionError(f"missing-kind discovered evidence must become blocked: {missing_kind_row}")

        cgra_only_evidence_dir = out_dir / "cgra-only-current-sim-cycle" / "current-sim-cycle"
        cgra_only_evidence_dir.mkdir(parents=True)
        cgra_only_report = cgra_only_evidence_dir / "cgra-only-cgra-sim-report.json"
        cgra_only_data = json.loads(stale_cgra.read_text())
        cgra_only_data["workload"] = "cgra_only"
        cgra_only_data["final_outputs"] = ["i32:3"]
        cgra_only_data["final_memory_state"] = {}
        cgra_only_data["functional_state_source"] = "carried_from_dfg_sim_report"
        cgra_only_data["cycle_breakdown"] = [
            {
                "category": "route_latency",
                "cycles": 2,
                "evidence": "mapping.route_segments",
                "modeled": True,
            }
        ]
        cgra_only_data["first_principles_checks"] = [
            {
                "name": "cgra_not_more_optimistic_than_dfg",
                "status": "pass",
                "evidence": "hardware_aware_cycles >= dfg_cycles",
            }
        ]
        cgra_only_report.write_text(json.dumps(cgra_only_data, indent=2, sort_keys=True) + "\n")
        cgra_only_sim = cgra_only_evidence_dir.parent / "cgra-only-sim-cycle-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_sim_cycle_summary.sh",
                "--output",
                str(cgra_only_sim),
            ],
            "CGRA-only discovered sim cycle summary",
        )
        cgra_only_rows = artifact_test_common.read_csv_rows(cgra_only_sim, HEADER)
        cgra_only_by_kernel = {row["kernel"]: row for row in cgra_only_rows}
        if set(cgra_only_by_kernel) != {"cgra_only"}:
            raise AssertionError(f"CGRA-only evidence should keep workload identity: {cgra_only_rows}")
        cgra_only_row = cgra_only_by_kernel["cgra_only"]
        if cgra_only_row.get("status") != "blocked" or cgra_only_row["cgra_sim_cycles"] != "":
            raise AssertionError(f"CGRA-only evidence must block until DFG report exists: {cgra_only_row}")
        if "lacks matching DFG-sim report evidence" not in cgra_only_row.get("diagnostic", ""):
            raise AssertionError(f"CGRA-only diagnostic should name missing DFG report: {cgra_only_row}")

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
