#!/usr/bin/env python3
"""Regression test for DSE candidate summary rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


HEADER = [
    "candidate",
    "workload",
    "hardware",
    "mapping_id",
    "objective",
    "cgra_sim_cycles",
    "frequency_mhz",
    "area_um2",
    "dynamic_power_mw",
    "leakage_power_mw",
    "energy_nj",
    "selection_status",
]


def artifact_id(path: Path) -> str:
    for suffix in (".csv", ".json"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def write_mapping_artifact(
    path: Path,
    workload: str,
    graph: str,
    mapping_id: str,
    hardware: str = "fabric0",
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pnr_mapping",
                "workload": workload,
                "hardware": hardware,
                "graph": graph,
                "mapping_id": mapping_id,
                "status": "pass",
                "placed_records": 1,
                "routed_edges": 1,
                "unrouted_edges": 0,
                "unplaced_records": 0,
                "config_records": 0,
                "placements": [{"software": f"{graph}#op0", "hardware": f"{hardware}::op0"}],
                "routes": [
                    {
                        "record_id": "route#0",
                        "producer_binding": f"placement:{graph}#op0",
                        "consumer_binding": f"placement:{graph}#op1",
                        "payload_kind": "data",
                        "segments": [
                            {
                                "segment_id": "seg0",
                                "segment_kind": "module_path",
                                "source_endpoint": f"{hardware}::op0.out",
                                "sink_endpoint": f"{hardware}::op1.in",
                            }
                        ],
                    }
                ],
                "config_bitstream": [],
            }
        )
    )


def write_cgra_report(
    path: Path,
    workload: str,
    mapping_id: str,
    dfg_cycles: int,
    cgra_cycles: int,
    hardware: str = "fabric0",
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "cgra_sim_report",
                "workload": workload,
                "hardware": hardware,
                "mapping_id": mapping_id,
                "status": "pass",
                "fidelity_level": "mapping_constraint_estimate",
                "metric_definition": "mapping_constraint_estimate",
                "operation_semantics_source": "loom.sim.operation_semantics.v1",
                "operation_cost_model_source": "loom.sim.operation_cost.v1",
                "difference_classification": "expected_hardware_constraint",
                "hardware_bound_classification": "within_modelled_bounds",
                "dfg_cycles": dfg_cycles,
                "modeled_lower_bound_cycles": cgra_cycles,
                "performance_delta_cycles": cgra_cycles - dfg_cycles,
                "route_latency_cycles": 1,
                "memory_latency_cycles": cgra_cycles - dfg_cycles - 1,
                "temporal_penalty_cycles": 0,
                "hardware_aware_cycles": cgra_cycles,
                "cycle_breakdown": [],
                "unmodeled_constraints": [],
                "first_principles_checks": [],
                "diagnostics": ["synthetic CGRA report for DSE producer test"],
            }
        )
    )


def write_mapping_set_manifest(
    path: Path,
    objective: str,
    mapping_artifacts: list[Path],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mapping_set_manifest",
                "objective": objective,
                "policy_id": f"deterministic_{objective}_v1",
                "candidates": [
                    {"mapping_artifact": str(mapping_artifact)}
                    for mapping_artifact in mapping_artifacts
                ],
                "diagnostics": [],
            }
        )
    )


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-dse-candidate-") as tmp:
        out_dir = Path(tmp)
        primitive, hardware = artifact_test_common.prepare_candidate_inputs(repo, out_dir)
        mapping = out_dir / "pnr-mapping-summary.csv"
        sim = out_dir / "sim-cycle-summary.csv"
        rtl_fpa = out_dir / "rtl-fpa-summary.csv"
        dse = out_dir / "dse-candidate-summary.csv"

        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/pnr/run_mapping_summary.sh",
                "--primitive-coverage",
                str(primitive),
                "--hardware-summary",
                str(hardware),
                "--output",
                str(mapping),
            ],
            "PnR mapping summary",
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
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_fpa_summary.sh",
                "--primitive-coverage",
                str(primitive),
                "--hardware-summary",
                str(hardware),
                "--output",
                str(rtl_fpa),
            ],
            "RTL/FPA summary",
        )
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            dse,
            HEADER,
            "--artifact",
            str(mapping),
            "--artifact",
            str(sim),
            "--artifact",
            str(rtl_fpa),
            label="DSE candidate summary",
        )

        matches = [
            row
            for row in rows
            if row["workload"] == "vecadd" and row["hardware"].endswith("::pe_two_pes")
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one vecadd pe_two_pes candidate row, got {rows}")
        row = matches[0]
        if not row["candidate"].startswith("candidate::vecadd::"):
            raise AssertionError(f"unexpected candidate id: {row}")
        for column in (
            "mapping_id",
            "cgra_sim_cycles",
            "frequency_mhz",
            "area_um2",
            "dynamic_power_mw",
            "leakage_power_mw",
            "energy_nj",
        ):
            if row[column] != "":
                raise AssertionError(f"blocked candidate must not fake {column}: {row}")
        if row["objective"] != "minimize_runtime":
            raise AssertionError(f"unexpected objective: {row}")
        if row["selection_status"] != "blocked":
            raise AssertionError(f"candidate should be blocked: {row}")
        if "missing mapping, simulator, or FPA evidence" not in row.get("diagnostic", ""):
            raise AssertionError(f"unexpected diagnostic: {row}")

        selected_like_mapping = out_dir / "selected-like-pnr-mapping-summary.csv"
        selected_like_mapping.write_text(
            "workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic\n"
            "vecadd,fabric0,map0,1,1,0,0,pass,synthetic complete mapping\n"
        )
        selected_like_sim = out_dir / "selected-like-sim-cycle-summary.csv"
        selected_like_sim.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,12,pass,synthetic cycle evidence\n"
        )
        selected_like_fpa = out_dir / "selected-like-rtl-fpa-summary.csv"
        selected_like_fpa.write_text(
            "hardware,workload,rtl_lint_status,rtl_sim_status,synth_status,frequency_mhz,area_um2,"
            "dynamic_power_mw,leakage_power_mw,fidelity_level,frequency_source,area_source,power_source,"
            "activity_source,status,diagnostic\n"
            "fabric0,vecadd,pass,pass,pass,100,200,3,1,analytic,analytic_fpa_model,analytic_fpa_model,"
            "analytic_fpa_model,default_toggle,pass,synthetic FPA evidence without energy\n"
        )
        summary_only_output = out_dir / "summary-only-dse-candidate-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            summary_only_output,
            HEADER,
            "--artifact",
            str(selected_like_mapping),
            "--artifact",
            str(selected_like_sim),
            "--artifact",
            str(selected_like_fpa),
            label="summary-only DSE candidate summary",
        )
        if len(rows) != 1:
            raise AssertionError(f"expected one summary-only row, got {rows}")
        row = rows[0]
        for column in (
            "mapping_id",
            "cgra_sim_cycles",
            "frequency_mhz",
            "area_um2",
            "dynamic_power_mw",
            "leakage_power_mw",
            "energy_nj",
        ):
            if row[column] != "":
                raise AssertionError(f"summary-only DSE candidate must not fake {column}: {row}")
        if row["selection_status"] != "blocked":
            raise AssertionError(f"summary-only DSE candidate should be blocked: {row}")

        selected_like_mapping_artifact = out_dir / "selected-like-pnr-mapping.json"
        write_mapping_artifact(selected_like_mapping_artifact, "vecadd", "g_vecadd", "map0")
        selected_like_cgra_report = out_dir / "selected-like-cgra-sim-report.json"
        write_cgra_report(selected_like_cgra_report, "vecadd", "map0", 10, 12)
        selected_like_output = out_dir / "selected-like-dse-candidate-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            selected_like_output,
            HEADER,
            "--artifact",
            str(selected_like_mapping),
            "--artifact",
            str(selected_like_mapping_artifact),
            "--artifact",
            str(selected_like_sim),
            "--artifact",
            str(selected_like_cgra_report),
            "--artifact",
            str(selected_like_fpa),
            label="selected-like DSE candidate summary",
        )
        if len(rows) != 1:
            raise AssertionError(f"expected one selected-like row, got {rows}")
        row = rows[0]
        expected = {
            "mapping_id": "map0",
            "cgra_sim_cycles": "12",
            "frequency_mhz": "100",
            "area_um2": "200",
            "dynamic_power_mw": "3",
            "leakage_power_mw": "1",
            "energy_nj": "0.480",
            "selection_status": "selected",
        }
        for column, value in expected.items():
            if row[column] != value:
                raise AssertionError(f"unexpected selected-like {column}: {row}")
        if "cycle-frequency-power-area energy estimate" not in row.get("diagnostic", ""):
            raise AssertionError(f"unexpected selected-like diagnostic: {row}")
        provenance_expected = {
            "candidate_kind": "combined_full_stack_candidate",
            "hardware_evidence_kind": "analytic_model_only",
            "objective_record": "objective::minimize_runtime",
            "policy_id": "deterministic_minimize_runtime_v1",
            "ordering_rule": "runtime_score_then_candidate_id",
        }
        for column, value in provenance_expected.items():
            if row.get(column) != value:
                raise AssertionError(f"unexpected selected-like {column}: {row}")
        leaked_out_dir = str(out_dir)
        for provenance_column in (
            "input_artifacts",
            "input_artifact_fingerprints",
            "output_artifacts",
        ):
            if leaked_out_dir in row.get(provenance_column, ""):
                raise AssertionError(f"selected-like {provenance_column} leaked local path: {row}")
        input_artifacts = {entry for entry in row.get("input_artifacts", "").split(";") if entry}
        expected_input_artifacts = {
            "selected-like-pnr-mapping-summary",
            "selected-like-pnr-mapping",
            "selected-like-sim-cycle-summary",
            "selected-like-cgra-sim-report",
            "selected-like-rtl-fpa-summary",
        }
        if input_artifacts != expected_input_artifacts:
            raise AssertionError(f"selected-like input artifacts missed identity provenance: {row}")
        input_fingerprints = artifact_test_common.semicolon_map(row.get("input_artifact_fingerprints", ""))
        expected_fingerprints = {
            "selected-like-pnr-mapping-summary": artifact_test_common.fingerprint(selected_like_mapping),
            "selected-like-pnr-mapping": artifact_test_common.fingerprint(selected_like_mapping_artifact),
            "selected-like-sim-cycle-summary": artifact_test_common.fingerprint(selected_like_sim),
            "selected-like-cgra-sim-report": artifact_test_common.fingerprint(selected_like_cgra_report),
            "selected-like-rtl-fpa-summary": artifact_test_common.fingerprint(selected_like_fpa),
        }
        if input_fingerprints != expected_fingerprints:
            raise AssertionError(f"selected-like input fingerprints are incomplete: {row}")
        if row.get("output_artifacts", "") != "selected-like-dse-candidate-summary":
            raise AssertionError(f"selected-like output artifacts missed summary identity: {row}")
        metric_records = row.get("metric_records", "")
        for metric in (
            "cgra_sim_cycles=12",
            "frequency_mhz=100",
            "area_um2=200",
            "dynamic_power_mw=3",
            "leakage_power_mw=1",
            "energy_nj=0.480",
        ):
            if metric not in metric_records:
                raise AssertionError(f"selected-like metric records missed {metric}: {row}")
        fidelity_records = row.get("feedback_fidelity_records", "")
        for record in (
            "cgra_sim_cycles=mapping_constraint_estimate:selected-like-cgra-sim-report",
            "frequency_mhz=analytic:analytic_fpa_model",
            "area_um2=analytic:analytic_fpa_model",
            "dynamic_power_mw=analytic:analytic_fpa_model:default_toggle",
            "leakage_power_mw=analytic:analytic_fpa_model:default_toggle",
            "energy_nj=analytic:derived_from_fpa_and_cgra_sim",
        ):
            if record not in fidelity_records:
                raise AssertionError(f"selected-like fidelity records missed {record}: {row}")

        artifact_only_output = out_dir / "artifact-only-dse-candidate-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            artifact_only_output,
            HEADER,
            "--artifact",
            str(selected_like_mapping_artifact),
            "--artifact",
            str(selected_like_sim),
            "--artifact",
            str(selected_like_cgra_report),
            "--artifact",
            str(selected_like_fpa),
            label="artifact-only DSE candidate summary",
        )
        if len(rows) != 1:
            raise AssertionError(f"expected one artifact-only row, got {rows}")
        row = rows[0]
        if row["selection_status"] != "selected":
            raise AssertionError(f"mapping artifact alone should seed a selected candidate: {row}")
        if row["candidate"] != "candidate::vecadd::fabric0::map0":
            raise AssertionError(f"artifact-only candidate id should include mapping id: {row}")
        if row["energy_nj"] != "0.480":
            raise AssertionError(f"artifact-only row should compute energy from FPA evidence: {row}")

        two_candidate_mapping = out_dir / "two-candidate-pnr-mapping-summary.csv"
        two_candidate_mapping.write_text(
            "workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic\n"
            "vecadd,fabric0,map_fast,1,1,0,0,pass,synthetic fast mapping\n"
            "vecadd,fabric1,map_slow,1,1,0,0,pass,synthetic slow mapping\n"
        )
        fast_mapping_artifact = out_dir / "fast-pnr-mapping.json"
        slow_mapping_artifact = out_dir / "slow-pnr-mapping.json"
        write_mapping_artifact(fast_mapping_artifact, "vecadd", "g_vecadd", "map_fast", "fabric0")
        write_mapping_artifact(slow_mapping_artifact, "vecadd", "g_vecadd", "map_slow", "fabric1")
        two_candidate_sim = out_dir / "two-candidate-sim-cycle-summary.csv"
        two_candidate_sim.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,12,pass,selected candidate cycle summary\n"
        )
        fast_cgra_report = out_dir / "fast-cgra-sim-report.json"
        slow_cgra_report = out_dir / "slow-cgra-sim-report.json"
        write_cgra_report(fast_cgra_report, "vecadd", "map_fast", 10, 12, "fabric0")
        write_cgra_report(slow_cgra_report, "vecadd", "map_slow", 10, 20, "fabric1")
        two_candidate_fpa = out_dir / "two-candidate-rtl-fpa-summary.csv"
        two_candidate_fpa.write_text(
            "hardware,workload,rtl_lint_status,rtl_sim_status,synth_status,frequency_mhz,area_um2,"
            "dynamic_power_mw,leakage_power_mw,fidelity_level,frequency_source,area_source,power_source,"
            "activity_source,status,diagnostic\n"
            "fabric0,vecadd,skipped,skipped,skipped,100,200,3,1,analytic,analytic_fpa_model,"
            "analytic_fpa_model,analytic_fpa_model,default_toggle,pass,synthetic fast FPA evidence\n"
            "fabric1,vecadd,skipped,skipped,skipped,100,300,4,1,analytic,analytic_fpa_model,"
            "analytic_fpa_model,analytic_fpa_model,default_toggle,pass,synthetic slow FPA evidence\n"
        )
        two_candidate_output = out_dir / "two-candidate-dse-candidate-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            two_candidate_output,
            HEADER,
            "--artifact",
            str(two_candidate_mapping),
            "--artifact",
            str(fast_mapping_artifact),
            "--artifact",
            str(slow_mapping_artifact),
            "--artifact",
            str(two_candidate_sim),
            "--artifact",
            str(fast_cgra_report),
            "--artifact",
            str(slow_cgra_report),
            "--artifact",
            str(two_candidate_fpa),
            label="two-candidate DSE candidate summary",
        )
        statuses = {row["hardware"]: row for row in rows}
        if set(statuses) != {"fabric0", "fabric1"}:
            raise AssertionError(f"expected two hardware candidates, got {rows}")
        if statuses["fabric0"]["selection_status"] != "selected":
            raise AssertionError(f"fast candidate should be selected: {statuses['fabric0']}")
        if statuses["fabric1"]["selection_status"] != "rejected":
            raise AssertionError(f"slow candidate should be rejected: {statuses['fabric1']}")
        if statuses["fabric1"]["cgra_sim_cycles"] != "20":
            raise AssertionError(f"slow candidate should keep its own CGRA cycles: {statuses['fabric1']}")
        if statuses["fabric1"]["energy_nj"] != "1.000":
            raise AssertionError(f"slow candidate should keep its own energy: {statuses['fabric1']}")
        for hardware, row in statuses.items():
            if row.get("candidate_kind") != "combined_full_stack_candidate":
                raise AssertionError(f"{hardware} candidate missed kind provenance: {row}")
            if artifact_id(two_candidate_output) not in row.get("output_artifacts", ""):
                raise AssertionError(f"{hardware} candidate missed output provenance: {row}")
            if "rejected by minimize_runtime deterministic ordering" in row["diagnostic"]:
                continue
            if "cycle-frequency-power-area energy estimate" not in row["diagnostic"]:
                raise AssertionError(f"{hardware} candidate has unexpected diagnostic: {row}")

        mapping_set_manifest = out_dir / "mapping-set-manifest.json"
        write_mapping_set_manifest(
            mapping_set_manifest,
            "minimize_runtime",
            [fast_mapping_artifact, slow_mapping_artifact],
        )
        manifest_output = out_dir / "manifest-dse-candidate-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            manifest_output,
            HEADER,
            "--artifact",
            str(mapping_set_manifest),
            "--artifact",
            str(two_candidate_sim),
            "--artifact",
            str(fast_cgra_report),
            "--artifact",
            str(slow_cgra_report),
            "--artifact",
            str(two_candidate_fpa),
            label="mapping-set manifest DSE candidate summary",
        )
        statuses = {row["hardware"]: row for row in rows}
        if set(statuses) != {"fabric0", "fabric1"}:
            raise AssertionError(f"manifest should expand two mapping candidates, got {rows}")
        if statuses["fabric0"]["selection_status"] != "selected":
            raise AssertionError(f"manifest fast candidate should be selected: {statuses['fabric0']}")
        if statuses["fabric1"]["selection_status"] != "rejected":
            raise AssertionError(f"manifest slow candidate should be rejected: {statuses['fabric1']}")
        for hardware, row in statuses.items():
            if artifact_id(mapping_set_manifest) not in row.get("input_artifacts", ""):
                raise AssertionError(f"{hardware} candidate missed mapping-set manifest provenance: {row}")
            if row.get("policy_id") != "deterministic_minimize_runtime_v1":
                raise AssertionError(f"{hardware} candidate missed manifest policy id: {row}")

        energy_mapping = out_dir / "energy-pnr-mapping-summary.csv"
        energy_mapping.write_text(
            "workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic\n"
            "vecadd,fabric_fast,map_energy_fast,1,1,0,0,pass,fast but power-hungry mapping\n"
            "vecadd,fabric_efficient,map_energy_efficient,1,1,0,0,pass,slower but lower-energy mapping\n"
        )
        fast_energy_mapping_artifact = out_dir / "fast-energy-pnr-mapping.json"
        efficient_energy_mapping_artifact = out_dir / "efficient-energy-pnr-mapping.json"
        write_mapping_artifact(
            fast_energy_mapping_artifact,
            "vecadd",
            "g_vecadd",
            "map_energy_fast",
            "fabric_fast",
        )
        write_mapping_artifact(
            efficient_energy_mapping_artifact,
            "vecadd",
            "g_vecadd",
            "map_energy_efficient",
            "fabric_efficient",
        )
        energy_manifest = out_dir / "energy-mapping-set-manifest.json"
        write_mapping_set_manifest(
            energy_manifest,
            "minimize_energy",
            [fast_energy_mapping_artifact, efficient_energy_mapping_artifact],
        )
        energy_sim = out_dir / "energy-sim-cycle-summary.csv"
        energy_sim.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,12,pass,selected candidate cycle summary\n"
        )
        fast_energy_cgra_report = out_dir / "fast-energy-cgra-sim-report.json"
        efficient_energy_cgra_report = out_dir / "efficient-energy-cgra-sim-report.json"
        write_cgra_report(fast_energy_cgra_report, "vecadd", "map_energy_fast", 10, 12, "fabric_fast")
        write_cgra_report(
            efficient_energy_cgra_report,
            "vecadd",
            "map_energy_efficient",
            10,
            20,
            "fabric_efficient",
        )
        energy_fpa = out_dir / "energy-rtl-fpa-summary.csv"
        energy_fpa.write_text(
            "hardware,workload,rtl_lint_status,rtl_sim_status,synth_status,frequency_mhz,area_um2,"
            "dynamic_power_mw,leakage_power_mw,fidelity_level,frequency_source,area_source,power_source,"
            "activity_source,status,diagnostic\n"
            "fabric_fast,vecadd,skipped,skipped,skipped,100,300,20,0,analytic,analytic_fpa_model,"
            "analytic_fpa_model,analytic_fpa_model,default_toggle,pass,fast high-power evidence\n"
            "fabric_efficient,vecadd,skipped,skipped,skipped,100,200,1,0,analytic,analytic_fpa_model,"
            "analytic_fpa_model,analytic_fpa_model,default_toggle,pass,slower low-power evidence\n"
        )
        energy_output = out_dir / "energy-dse-candidate-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            energy_output,
            HEADER,
            "--artifact",
            str(energy_manifest),
            "--artifact",
            str(energy_sim),
            "--artifact",
            str(fast_energy_cgra_report),
            "--artifact",
            str(efficient_energy_cgra_report),
            "--artifact",
            str(energy_fpa),
            label="energy-objective mapping-set manifest DSE candidate summary",
        )
        statuses = {row["hardware"]: row for row in rows}
        if set(statuses) != {"fabric_fast", "fabric_efficient"}:
            raise AssertionError(f"energy manifest should expand two mapping candidates, got {rows}")
        if statuses["fabric_efficient"]["selection_status"] != "selected":
            raise AssertionError(
                f"energy manifest should select the lower-energy candidate: {statuses['fabric_efficient']}"
            )
        if statuses["fabric_fast"]["selection_status"] != "rejected":
            raise AssertionError(f"energy manifest should reject the higher-energy candidate: {statuses['fabric_fast']}")
        for hardware, row in statuses.items():
            if row.get("objective") != "minimize_energy":
                raise AssertionError(f"{hardware} candidate missed manifest objective: {row}")
            if row.get("objective_record") != "objective::minimize_energy":
                raise AssertionError(f"{hardware} candidate missed manifest objective record: {row}")
            if row.get("policy_id") != "deterministic_minimize_energy_v1":
                raise AssertionError(f"{hardware} candidate missed manifest energy policy id: {row}")
            if row.get("ordering_rule") != "energy_score_then_candidate_id":
                raise AssertionError(f"{hardware} candidate missed energy ordering rule: {row}")

        dynamic_power_manifest = out_dir / "dynamic-power-mapping-set-manifest.json"
        write_mapping_set_manifest(
            dynamic_power_manifest,
            "minimize_dynamic_power",
            [fast_energy_mapping_artifact, efficient_energy_mapping_artifact],
        )
        dynamic_power_output = out_dir / "dynamic-power-dse-candidate-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            dynamic_power_output,
            HEADER,
            "--artifact",
            str(dynamic_power_manifest),
            "--artifact",
            str(energy_sim),
            "--artifact",
            str(fast_energy_cgra_report),
            "--artifact",
            str(efficient_energy_cgra_report),
            "--artifact",
            str(energy_fpa),
            label="dynamic-power-objective mapping-set manifest DSE candidate summary",
        )
        statuses = {row["hardware"]: row for row in rows}
        if set(statuses) != {"fabric_fast", "fabric_efficient"}:
            raise AssertionError(f"dynamic-power manifest should expand two mapping candidates, got {rows}")
        if statuses["fabric_efficient"]["selection_status"] != "selected":
            raise AssertionError(
                f"dynamic-power manifest should select the lower-power candidate: {statuses['fabric_efficient']}"
            )
        if statuses["fabric_fast"]["selection_status"] != "rejected":
            raise AssertionError(
                f"dynamic-power manifest should reject the higher-power candidate: {statuses['fabric_fast']}"
            )
        for hardware, row in statuses.items():
            if row.get("objective") != "minimize_dynamic_power":
                raise AssertionError(f"{hardware} candidate missed manifest dynamic-power objective: {row}")
            if row.get("objective_record") != "objective::minimize_dynamic_power":
                raise AssertionError(f"{hardware} candidate missed manifest dynamic-power objective record: {row}")
            if row.get("policy_id") != "deterministic_minimize_dynamic_power_v1":
                raise AssertionError(f"{hardware} candidate missed manifest dynamic-power policy id: {row}")
            if row.get("ordering_rule") != "dynamic_power_score_then_candidate_id":
                raise AssertionError(f"{hardware} candidate missed dynamic-power ordering rule: {row}")

        leakage_manifest = out_dir / "leakage-power-mapping-set-manifest.json"
        write_mapping_set_manifest(
            leakage_manifest,
            "minimize_leakage_power",
            [fast_energy_mapping_artifact, efficient_energy_mapping_artifact],
        )
        leakage_fpa = out_dir / "leakage-rtl-fpa-summary.csv"
        leakage_fpa.write_text(
            "hardware,workload,rtl_lint_status,rtl_sim_status,synth_status,frequency_mhz,area_um2,"
            "dynamic_power_mw,leakage_power_mw,fidelity_level,frequency_source,area_source,power_source,"
            "activity_source,status,diagnostic\n"
            "fabric_fast,vecadd,skipped,skipped,skipped,100,300,1,20,analytic,analytic_fpa_model,"
            "analytic_fpa_model,analytic_fpa_model,default_toggle,pass,fast high-leakage evidence\n"
            "fabric_efficient,vecadd,skipped,skipped,skipped,100,200,20,1,analytic,analytic_fpa_model,"
            "analytic_fpa_model,analytic_fpa_model,default_toggle,pass,slower low-leakage evidence\n"
        )
        leakage_output = out_dir / "leakage-power-dse-candidate-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            leakage_output,
            HEADER,
            "--artifact",
            str(leakage_manifest),
            "--artifact",
            str(energy_sim),
            "--artifact",
            str(fast_energy_cgra_report),
            "--artifact",
            str(efficient_energy_cgra_report),
            "--artifact",
            str(leakage_fpa),
            label="leakage-power-objective mapping-set manifest DSE candidate summary",
        )
        statuses = {row["hardware"]: row for row in rows}
        if set(statuses) != {"fabric_fast", "fabric_efficient"}:
            raise AssertionError(f"leakage-power manifest should expand two mapping candidates, got {rows}")
        if statuses["fabric_efficient"]["selection_status"] != "selected":
            raise AssertionError(
                f"leakage-power manifest should select the lower-leakage candidate: {statuses['fabric_efficient']}"
            )
        if statuses["fabric_fast"]["selection_status"] != "rejected":
            raise AssertionError(
                f"leakage-power manifest should reject the higher-leakage candidate: {statuses['fabric_fast']}"
            )
        for hardware, row in statuses.items():
            if row.get("objective") != "minimize_leakage_power":
                raise AssertionError(f"{hardware} candidate missed manifest leakage-power objective: {row}")
            if row.get("objective_record") != "objective::minimize_leakage_power":
                raise AssertionError(f"{hardware} candidate missed manifest leakage-power objective record: {row}")
            if row.get("policy_id") != "deterministic_minimize_leakage_power_v1":
                raise AssertionError(f"{hardware} candidate missed manifest leakage-power policy id: {row}")
            if row.get("ordering_rule") != "leakage_power_score_then_candidate_id":
                raise AssertionError(f"{hardware} candidate missed leakage-power ordering rule: {row}")
            if row.get("leakage_power_mw") not in {"20", "1"}:
                raise AssertionError(f"{hardware} candidate missed leakage power evidence: {row}")

        same_hardware_fast_mapping_artifact = out_dir / "same-hardware-fast-pnr-mapping.json"
        same_hardware_slow_mapping_artifact = out_dir / "same-hardware-slow-pnr-mapping.json"
        write_mapping_artifact(
            same_hardware_fast_mapping_artifact,
            "vecadd",
            "g_vecadd",
            "map_same_fast",
            "fabric_same",
        )
        write_mapping_artifact(
            same_hardware_slow_mapping_artifact,
            "vecadd",
            "g_vecadd",
            "map_same_slow",
            "fabric_same",
        )
        same_hardware_manifest = out_dir / "same-hardware-mapping-set-manifest.json"
        write_mapping_set_manifest(
            same_hardware_manifest,
            "minimize_runtime",
            [same_hardware_fast_mapping_artifact, same_hardware_slow_mapping_artifact],
        )
        same_hardware_sim = out_dir / "same-hardware-sim-cycle-summary.csv"
        same_hardware_sim.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,10,12,pass,selected candidate cycle summary\n"
        )
        same_hardware_fast_cgra_report = out_dir / "same-hardware-fast-cgra-sim-report.json"
        same_hardware_slow_cgra_report = out_dir / "same-hardware-slow-cgra-sim-report.json"
        write_cgra_report(
            same_hardware_fast_cgra_report,
            "vecadd",
            "map_same_fast",
            10,
            12,
            "fabric_same",
        )
        write_cgra_report(
            same_hardware_slow_cgra_report,
            "vecadd",
            "map_same_slow",
            10,
            20,
            "fabric_same",
        )
        same_hardware_fpa = out_dir / "same-hardware-rtl-fpa-summary.csv"
        same_hardware_fpa.write_text(
            "hardware,workload,rtl_lint_status,rtl_sim_status,synth_status,frequency_mhz,area_um2,"
            "dynamic_power_mw,leakage_power_mw,fidelity_level,frequency_source,area_source,power_source,"
            "activity_source,status,diagnostic\n"
            "fabric_same,vecadd,skipped,skipped,skipped,100,200,3,1,analytic,analytic_fpa_model,"
            "analytic_fpa_model,analytic_fpa_model,default_toggle,pass,shared hardware FPA evidence\n"
        )
        same_hardware_output = out_dir / "same-hardware-dse-candidate-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            same_hardware_output,
            HEADER,
            "--artifact",
            str(same_hardware_manifest),
            "--artifact",
            str(same_hardware_sim),
            "--artifact",
            str(same_hardware_fast_cgra_report),
            "--artifact",
            str(same_hardware_slow_cgra_report),
            "--artifact",
            str(same_hardware_fpa),
            label="same-hardware mapping-set manifest DSE candidate summary",
        )
        if len(rows) != 2:
            raise AssertionError(f"same-hardware manifest should emit two rows, got {rows}")
        candidate_ids = {row["candidate"] for row in rows}
        if len(candidate_ids) != len(rows):
            raise AssertionError(f"same-hardware mapping candidates must have unique ids: {rows}")
        rows_by_mapping = {row["mapping_id"]: row for row in rows}
        if set(rows_by_mapping) != {"map_same_fast", "map_same_slow"}:
            raise AssertionError(f"same-hardware rows missed mapping ids: {rows}")
        for mapping_id, row in rows_by_mapping.items():
            if mapping_id not in row["candidate"]:
                raise AssertionError(f"candidate id must include mapping id {mapping_id}: {row}")

        unsupported_scope_manifest = out_dir / "unsupported-scope-mapping-set-manifest.json"
        write_mapping_set_manifest(
            unsupported_scope_manifest,
            "minimize_unsupported_scope_diagnostics",
            [same_hardware_fast_mapping_artifact, same_hardware_slow_mapping_artifact],
        )
        unsupported_scope_ledger = out_dir / "unsupported-scope-ledger.csv"
        unsupported_scope_ledger.write_text(
            "stage,case,artifact,reason,owner,blocking_input\n"
            "dse,candidate::vecadd::fabric_same::map_same_fast,dse-candidate-summary,"
            f"synthetic candidate diagnostic,implementation,{same_hardware_fast_mapping_artifact}\n"
        )
        unsupported_scope_output = out_dir / "unsupported-scope-dse-candidate-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            unsupported_scope_output,
            HEADER,
            "--artifact",
            str(unsupported_scope_manifest),
            "--artifact",
            str(same_hardware_sim),
            "--artifact",
            str(same_hardware_fast_cgra_report),
            "--artifact",
            str(same_hardware_slow_cgra_report),
            "--artifact",
            str(same_hardware_fpa),
            "--artifact",
            str(unsupported_scope_ledger),
            label="unsupported-scope-objective DSE candidate summary",
        )
        rows_by_mapping = {row["mapping_id"]: row for row in rows}
        if set(rows_by_mapping) != {"map_same_fast", "map_same_slow"}:
            raise AssertionError(f"unsupported-scope objective missed mapping candidates: {rows}")
        if rows_by_mapping["map_same_slow"]["selection_status"] != "selected":
            raise AssertionError(
                "unsupported-scope objective should select the candidate with fewer diagnostics: "
                f"{rows_by_mapping['map_same_slow']}"
            )
        if rows_by_mapping["map_same_fast"]["selection_status"] != "rejected":
            raise AssertionError(
                "unsupported-scope objective should reject the diagnosed candidate: "
                f"{rows_by_mapping['map_same_fast']}"
            )
        expected_counts = {"map_same_fast": "1", "map_same_slow": "0"}
        for mapping_id, row in rows_by_mapping.items():
            if row.get("objective") != "minimize_unsupported_scope_diagnostics":
                raise AssertionError(f"{mapping_id} candidate missed unsupported-scope objective: {row}")
            if row.get("policy_id") != "deterministic_minimize_unsupported_scope_diagnostics_v1":
                raise AssertionError(f"{mapping_id} candidate missed unsupported-scope policy id: {row}")
            if row.get("ordering_rule") != "unsupported_scope_diagnostics_score_then_candidate_id":
                raise AssertionError(f"{mapping_id} candidate missed unsupported-scope ordering rule: {row}")
            if row.get("unsupported_scope_diagnostics_count") != expected_counts[mapping_id]:
                raise AssertionError(f"{mapping_id} candidate missed diagnostic count: {row}")
            metric_records = row.get("metric_records", "")
            expected_metric = (
                "unsupported_scope_diagnostics_count="
                f"{expected_counts[mapping_id]}"
            )
            if expected_metric not in metric_records:
                raise AssertionError(f"{mapping_id} candidate missed diagnostic metric: {row}")
            if artifact_id(unsupported_scope_ledger) not in row.get("input_artifacts", ""):
                raise AssertionError(f"{mapping_id} candidate missed ledger provenance: {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
