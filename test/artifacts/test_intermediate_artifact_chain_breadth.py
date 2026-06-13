#!/usr/bin/env python3
"""Regression test for broader single-graph full-stack artifact chains."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import artifact_test_common


COMMON_FILES = [
    "old-app-corpus-inventory.csv",
    "app-corpus-import-status.csv",
    "source-compat-summary.csv",
    "compiler-pipeline-summary.csv",
    "cmsis-compiler-pipeline-summary.csv",
    "dataflow-primitive-coverage.csv",
    "adg-hardware-summary.csv",
    "pnr-mapping-summary.csv",
    "pnr-mapping.json",
    "sim-comparison-report.json",
    "runtime-package.json",
    "sim-cycle-summary.csv",
    "rtl-manifest.json",
    "rtl-eda-report.json",
    "rtl-sim-eda-report.json",
    "rtl-fpa-report.json",
    "rtl-fpa-summary.csv",
    "workload-report-bundle.json",
    "hardware-report-bundle.json",
    "dse-candidate-summary.csv",
    "dse-report-bundle.json",
    "full-stack-artifact-manifest.json",
    "e2e-demonstrator-summary.csv",
    "unsupported-scope-ledger.csv",
    "artifact-audit-summary.json",
]


CASES = {
    "reduction": {
        "graph": "g_t_reduce_sum_red_0_0",
        "mapping_id": "reduction__g_t_reduce_sum_red_0_0__shared_reduction_adg",
        "placed_records": "5",
        "route_edge_count": "6",
        "config_records": 97,
        "dfg_cycles": 1155,
        "dynamic_work_items": 128,
        "cgra_cycles": 1173,
        "byte_size": 512,
        "element_layout": "i32[128]",
        "mapping_status": "pass",
    },
    "mean": {
        "graph": "g_t_mean_kernel_red_0_0",
        "mapping_id": "mean__g_t_mean_kernel_red_0_0__shared_reduction_adg",
        "placed_records": "7",
        "route_edge_count": "9",
        "config_records": 142,
        "dfg_cycles": 904,
        "dynamic_work_items": 64,
        "cgra_cycles": 929,
        "byte_size": 256,
        "element_layout": "f32[64]",
        "mapping_status": "pass",
    },
    "vecnorm_l1": {
        "graph": "g_t_vecnorm_l1_red_0_0",
        "mapping_id": "vecnorm_l1__g_t_vecnorm_l1_red_0_0__shared_reduction_adg",
        "placed_records": "6",
        "route_edge_count": "7",
        "config_records": 116,
        "dfg_cycles": 643,
        "dynamic_work_items": 64,
        "cgra_cycles": 664,
        "byte_size": 256,
        "element_layout": "i32[64]",
        "mapping_status": "pass",
    },
    "vecnorm_l2": {
        "graph": "g_t_vecnorm_l2_red_0_0",
        "mapping_id": "vecnorm_l2__g_t_vecnorm_l2_red_0_0__shared_reduction_adg",
        "placed_records": "6",
        "route_edge_count": "8",
        "config_records": 123,
        "dfg_cycles": 771,
        "dynamic_work_items": 64,
        "cgra_cycles": 793,
        "byte_size": 256,
        "element_layout": "i32[64]",
        "mapping_status": "pass",
    },
    "correlation": {
        "graph": "g_t_correlation_kernel_red_0_0",
        "mapping_id": "correlation__g_t_correlation_kernel_red_0_0__shared_reduction_adg",
        "placed_records": "10",
        "route_edge_count": "12",
        "config_records": 148,
        "dfg_cycles": 346,
        "dynamic_work_items": 16,
        "cgra_cycles": 369,
        "byte_size": 1028,
        "element_layout": "f32[128];f32[16];f32[113]",
    },
    "prefix_sum": {
        "graph": "g_t_prefix_sum_red_0_0",
        "mapping_id": "prefix_sum__g_t_prefix_sum_red_0_0__shared_reduction_adg",
        "placed_records": "6",
        "route_edge_count": "9",
        "config_records": 146,
        "dfg_cycles": 835,
        "dynamic_work_items": 64,
        "cgra_cycles": 866,
        "byte_size": 512,
        "element_layout": "i32[64];i32[64]",
        "mapping_status": "pass",
    },
    "cumsum": {
        "graph": "g_t_cumsum_kernel_red_0_0",
        "mapping_id": "cumsum__g_t_cumsum_kernel_red_0_0__shared_reduction_adg",
        "placed_records": "6",
        "route_edge_count": "9",
        "config_records": 146,
        "dfg_cycles": 14339,
        "dynamic_work_items": 1024,
        "cgra_cycles": 14370,
        "byte_size": 8192,
        "element_layout": "f32[1024];f32[1024]",
        "mapping_status": "pass",
    },
    "prefix_sum_inclusive": {
        "graph": "g_t_prefix_sum_inclusive_kernel_red_0_0",
        "mapping_id": "prefix_sum_inclusive__g_t_prefix_sum_inclusive_kernel_red_0_0__shared_reduction_adg",
        "placed_records": "6",
        "route_edge_count": "9",
        "config_records": 146,
        "dfg_cycles": 13302,
        "dynamic_work_items": 1023,
        "cgra_cycles": 13333,
        "byte_size": 8192,
        "element_layout": "u32[1024];u32[1024]",
        "mapping_status": "pass",
    },
    "integrate_trapz": {
        "graph": "g_t_integrate_trapz_red_0_0",
        "mapping_id": "integrate_trapz__g_t_integrate_trapz_red_0_0__shared_reduction_adg",
        "placed_records": "15",
        "route_edge_count": "20",
        "config_records": 238,
        "dfg_cycles": 299,
        "dynamic_work_items": 8,
        "cgra_cycles": 340,
        "byte_size": 72,
        "element_layout": "f32[9];f32[9]",
    },
    "spmv": {
        "graph": "g_t_spmv_kernel_red_0_0",
        "mapping_id": "spmv__g_t_spmv_kernel_red_0_0__shared_reduction_adg",
        "placed_records": "9",
        "route_edge_count": "6",
        "config_records": 130,
        "dfg_cycles": 47,
        "dynamic_work_items": 2,
        "cgra_cycles": 72,
        "byte_size": 128,
        "element_layout": "u32[9];u32[9];u32[5];u32[5];u32[4]",
    },
    "convolve_1d": {
        "graph": "g_t_convolve_1d_kernel_red_0_0",
        "mapping_id": "convolve_1d__g_t_convolve_1d_kernel_red_0_0__shared_reduction_adg",
        "placed_records": "10",
        "route_edge_count": "12",
        "config_records": 148,
        "dfg_cycles": 157,
        "dynamic_work_items": 7,
        "cgra_cycles": 180,
        "byte_size": 1028,
        "element_layout": "f32[128];f32[7];f32[122]",
    },
    "matvec": {
        "graph": "g_t_matvec_kernel_0_0",
        "mapping_id": "matvec__g_t_matvec_kernel_0_0__shared_reduction_adg",
        "placed_records": "7",
        "route_edge_count": "3",
        "config_records": 101,
        "dfg_cycles": 83,
        "dynamic_work_items": 5,
        "cgra_cycles": 101,
        "byte_size": 116,
        "element_layout": "u32[20];u32[5];u32[4]",
    },
    "vecmul": {
        "graph": "g_t__ZN12_GLOBAL__N_116vecmul_candidateEPKfS1_Pfj_0_0",
        "hardware": "shared_vector_alu_adg",
        "mapping_id": "vecmul__g_t__ZN12_GLOBAL__N_116vecmul_candidateEPKfS1_Pfj_0_0__shared_vector_alu_adg",
        "placed_records": "5",
        "route_edge_count": "6",
        "config_records": 119,
        "dfg_cycles": 256,
        "dynamic_work_items": 16,
        "cgra_cycles": 288,
        "byte_size": 192,
        "element_layout": "f32[16];f32[16];f32[16]",
        "mapping_status": "pass",
    },
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json_object(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise AssertionError(f"expected JSON object in {path.name}: {data}")
    return data


def single_row(
    rows: list[dict[str, str]],
    *,
    key: str,
    value: str,
    label: str,
) -> dict[str, str]:
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise AssertionError(f"expected one {label} row, got {rows}")
    return matches[0]


def assert_fields(
    record: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"unexpected {label} {key}: {record}")


def assert_runtime_evidence(
    runtime_evidence: Mapping[str, object],
    *,
    case_name: str,
    expected: Mapping[str, object],
) -> None:
    expected_hardware = str(expected.get("hardware", "shared_reduction_adg"))
    runtime_report_identity = (
        f"runtime-report::{case_name}::{expected['mapping_id']}::report_only"
    )
    assert_fields(
        runtime_evidence,
        {
            "runtime_package_identity": "runtime-package",
            "runtime_report_identity": runtime_report_identity,
            "host_program_identity": f"test-app-host::{case_name}::default",
            "host_wrapper_identity": f"runtime-wrapper::{case_name}::{expected['mapping_id']}",
            "work_package_identity": f"work-package::{case_name}::{expected['mapping_id']}",
            "launch_descriptor_identity": (
                f"launch::{case_name}::{expected['mapping_id']}::"
                f"test-app-fixture::{case_name}::default"
            ),
            "mapping_artifact_identity": "pnr-mapping",
            "fabric_adg_identity": expected_hardware,
            "target_profile_id": "simulator::cgra_sim::mapping_constraint_estimate",
            "data_movement_policy": "simulated",
            "synchronization_mode": "host_wait",
            "launch_status": "not_run",
            "target_status": "not_run",
        },
        label=f"{case_name} runtime evidence",
    )
    if runtime_evidence.get("simulator_report_identities") != [
        f"{case_name}-cgra-sim-report",
        "sim-comparison-report",
    ]:
        raise AssertionError(f"unexpected {case_name} runtime simulator identities: {runtime_evidence}")
    argument_descriptors = runtime_evidence.get("argument_descriptors")
    expected_arguments = [
        {
            "name": "runtime_input",
            "descriptor_kind": "test_fixture",
            "identity": f"test-app-fixture::{case_name}::default",
        },
        {
            "name": "mapping_artifact",
            "descriptor_kind": "pnr_mapping_artifact",
            "identity": "pnr-mapping",
        },
        {
            "name": "cgra_sim_report",
            "descriptor_kind": "cgra_sim_report",
            "identity": f"{case_name}-cgra-sim-report",
        },
        {
            "name": "sim_comparison_report",
            "descriptor_kind": "sim_comparison_report",
            "identity": "sim-comparison-report",
        },
    ]
    if argument_descriptors != expected_arguments:
        raise AssertionError(f"unexpected {case_name} runtime arguments: {runtime_evidence}")
    report_configuration = runtime_evidence.get("report_output_configuration")
    if not isinstance(report_configuration, dict):
        raise AssertionError(f"missing {case_name} runtime report configuration: {runtime_evidence}")
    assert_fields(
        report_configuration,
        {
            "runtime_report_identity": runtime_report_identity,
            "diagnostic_output_enabled": True,
            "trace_output_enabled": False,
            "profiling_output_enabled": False,
        },
        label=f"{case_name} runtime report configuration",
    )
    memory_descriptors = runtime_evidence.get("memory_descriptors")
    if not isinstance(memory_descriptors, list) or len(memory_descriptors) != 1:
        raise AssertionError(f"{case_name} runtime evidence needs one memory descriptor: {runtime_evidence}")
    assert_fields(
        memory_descriptors[0],
        {
            "logical_argument": f"{case_name}.default_input",
            "host_buffer_identity": f"runtime-buffer::{case_name}::default_input",
            "policy": "simulated",
            "runtime_input_identity": f"test-app-fixture::{case_name}::default",
            "byte_size": expected["byte_size"],
            "element_layout": expected["element_layout"],
            "alignment_bytes": 4,
            "address_space": "simulator::memory_model",
            "coherence_requirement": "simulator_consistent",
            "transfer_policy": "simulated",
        },
        label=f"{case_name} runtime evidence memory descriptor",
    )


def assert_case(repo: Path, case_name: str, expected: Mapping[str, object]) -> None:
    with artifact_test_common.repo_temp_dir(repo, f"loom-{case_name}-chain-") as tmp:
        out_dir = Path(tmp)
        expected_hardware = str(expected.get("hardware", "shared_reduction_adg"))
        mapping_passes = expected.get("mapping_status") == "pass"
        expected_cgra_cycles = expected["cgra_cycles"] if mapping_passes else expected["dfg_cycles"]
        expected_difference = "expected_hardware_constraint" if mapping_passes else "unsupported_scope"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--case",
                case_name,
            ],
            f"{case_name} intermediate artifact chain",
        )

        expected_files = [
            *COMMON_FILES,
            f"{case_name}-dfg-sim-report.json",
            f"{case_name}-dfg-sim-cycle-summary.csv",
            f"{case_name}-cgra-sim-report.json",
        ]
        missing = [name for name in expected_files if not (out_dir / name).is_file()]
        if missing:
            raise AssertionError(f"missing {case_name} chain artifacts: {missing}")

        mapping = single_row(
            read_csv_rows(out_dir / "pnr-mapping-summary.csv"),
            key="workload",
            value=case_name,
            label=f"{case_name} mapping",
        )
        assert_fields(
            mapping,
            {
                "hardware": expected_hardware,
                "mapping_id": expected["mapping_id"],
                "placed_records": expected["placed_records"],
                "routed_edges": expected["route_edge_count"] if mapping_passes else "0",
                "unrouted_edges": "0" if mapping_passes else expected["route_edge_count"],
                "unplaced_records": "0",
                "status": "pass" if mapping_passes else "fail",
            },
            label=f"{case_name} mapping",
        )

        mapping_artifact = read_json_object(out_dir / "pnr-mapping.json")
        assert_fields(
            mapping_artifact,
            {
                "workload": case_name,
                "graph": expected["graph"],
                "mapping_id": expected["mapping_id"],
                "config_records": expected["config_records"] if mapping_passes else 0,
                "status": "pass" if mapping_passes else "fail",
            },
            label=f"{case_name} mapping artifact",
        )

        dfg_report = read_json_object(out_dir / f"{case_name}-dfg-sim-report.json")
        assert_fields(
            dfg_report,
            {
                "status": "pass",
                "workload": case_name,
                "graph": expected["graph"],
                "optimistic_cycles": expected["dfg_cycles"],
                "dynamic_work_items": expected["dynamic_work_items"],
            },
            label=f"{case_name} DFG-sim report",
        )

        cgra_report = read_json_object(out_dir / f"{case_name}-cgra-sim-report.json")
        assert_fields(
            cgra_report,
            {
                "status": "pass" if mapping_passes else "blocked",
                "workload": case_name,
                "mapping_id": expected["mapping_id"],
                "hardware_aware_cycles": expected_cgra_cycles,
                "difference_classification": expected_difference,
            },
            label=f"{case_name} CGRA-sim report",
        )
        if cgra_report["hardware_aware_cycles"] < dfg_report["optimistic_cycles"]:
            raise AssertionError(f"CGRA-sim must not be more optimistic than DFG-sim: {cgra_report}")

        sim_row = single_row(
            read_csv_rows(out_dir / "sim-cycle-summary.csv"),
            key="kernel",
            value=case_name,
            label=f"{case_name} sim",
        )
        assert_fields(
            sim_row,
            {
                "dfg_sim_cycles": str(expected["dfg_cycles"]),
                "cgra_sim_cycles": str(expected_cgra_cycles) if mapping_passes else "",
                "status": "pass" if mapping_passes else "blocked",
            },
            label=f"{case_name} sim row",
        )
        existing_dfg_cycles = {448, 579, 1027}
        if int(sim_row["dfg_sim_cycles"]) in existing_dfg_cycles:
            raise AssertionError(f"{case_name} cycles should add distinct workload evidence: {sim_row}")

        comparison = read_json_object(out_dir / "sim-comparison-report.json")
        assert_fields(
            comparison,
            {
                "status": "pass" if mapping_passes else "blocked",
                "workload": case_name,
                "dfg_sim_cycles": expected["dfg_cycles"],
                "cgra_sim_cycles": expected_cgra_cycles,
                "difference_classification": expected_difference,
            },
            label=f"{case_name} simulation comparison",
        )

        runtime_package = read_json_object(out_dir / "runtime-package.json")
        expected_runtime_status = "pass" if mapping_passes else "blocked"
        if runtime_package.get("status") != expected_runtime_status or runtime_package.get("workload") != case_name:
            raise AssertionError(f"unexpected {case_name} runtime package: {runtime_package}")
        if runtime_package.get("work_package_identity") != (
            f"work-package::{case_name}::{expected['mapping_id']}"
        ):
            raise AssertionError(f"unexpected {case_name} work package identity: {runtime_package}")
        memory_descriptors = runtime_package.get("memory_descriptors")
        if not isinstance(memory_descriptors, list) or len(memory_descriptors) != 1:
            raise AssertionError(f"{case_name} runtime package needs one memory descriptor: {runtime_package}")
        memory_descriptor = memory_descriptors[0]
        assert_fields(
            memory_descriptor,
            {
                "logical_argument": f"{case_name}.default_input",
                "host_buffer_identity": f"runtime-buffer::{case_name}::default_input",
                "policy": "simulated",
                "runtime_input_identity": f"test-app-fixture::{case_name}::default",
                "byte_size": expected["byte_size"],
                "element_layout": expected["element_layout"],
                "alignment_bytes": 4,
                "address_space": "simulator::memory_model",
                "coherence_requirement": "simulator_consistent",
                "transfer_policy": "simulated",
            },
            label=f"{case_name} memory descriptor",
        )

        dse_row = single_row(
            read_csv_rows(out_dir / "dse-candidate-summary.csv"),
            key="workload",
            value=case_name,
            label=f"{case_name} DSE",
        )
        assert_fields(
            dse_row,
            {
                "mapping_id": expected["mapping_id"],
                "cgra_sim_cycles": str(expected_cgra_cycles) if mapping_passes else "",
                "selection_status": "selected" if mapping_passes else "blocked",
            },
            label=f"{case_name} DSE",
        )
        if mapping_passes:
            if dse_row.get("hardware_evidence_kind") != "analytic_model_only":
                raise AssertionError(f"{case_name} DSE row should mark analytic hardware evidence: {dse_row}")
            if "energy_nj=analytic:derived_from_fpa_and_cgra_sim" not in dse_row.get(
                "feedback_fidelity_records", ""
            ):
                raise AssertionError(f"{case_name} DSE row should mark analytic energy fidelity: {dse_row}")

        workload_bundle = read_json_object(out_dir / "workload-report-bundle.json")
        expected_bundle_status = "pass" if mapping_passes else "blocked"
        if workload_bundle.get("report_status") != expected_bundle_status or workload_bundle.get("workload") != case_name:
            raise AssertionError(f"unexpected {case_name} workload report bundle: {workload_bundle}")
        runtime_evidence = workload_bundle.get("runtime_evidence")
        if not isinstance(runtime_evidence, dict):
            raise AssertionError(f"missing {case_name} runtime evidence: {workload_bundle}")
        assert_runtime_evidence(runtime_evidence, case_name=case_name, expected=expected)
        metric_ids = {
            metric.get("metric_id")
            for metric in workload_bundle.get("metric_records", [])
            if isinstance(metric, dict)
        }
        for metric_id in (
            f"metric::{case_name}::dfg_sim_cycles",
            f"metric::{case_name}::workload_size_items",
            f"metric::{expected_hardware}::frequency_mhz",
        ):
            if metric_id not in metric_ids:
                raise AssertionError(f"workload report bundle missed {metric_id}: {workload_bundle}")
        if mapping_passes and f"metric::{case_name}::cgra_sim_cycles" not in metric_ids:
            raise AssertionError(f"workload report bundle missed CGRA cycles metric: {workload_bundle}")

        hardware_bundle = read_json_object(out_dir / "hardware-report-bundle.json")
        if hardware_bundle.get("supported_workload_classes") != [case_name]:
            raise AssertionError(f"hardware report should cite {case_name} FPA support: {hardware_bundle}")

        audit = read_json_object(out_dir / "artifact-audit-summary.json")
        if audit.get("verdict") != "pass" or audit.get("cross_artifact_findings"):
            raise AssertionError(f"expected {case_name} chain audit pass, got {audit}")
        reviewed = {
            Path(review.get("artifact", "")).name
            for review in audit.get("artifact_reviews", [])
            if isinstance(review, dict)
        }
        expected_reviewed = set(expected_files) - {"artifact-audit-summary.json"}
        if reviewed != expected_reviewed:
            raise AssertionError(f"audit reviewed {reviewed}, expected {expected_reviewed}")
        cross_checks = {
            check.get("rule")
            for check in audit.get("cross_artifact_checks", [])
            if isinstance(check, dict)
        }
        expected_cross_checks = {"sim_cycle_dfg_report_evidence"}
        expected_cross_checks.add(
            "sim_cycle_report_mapping_evidence" if mapping_passes else "sim_cycle_blocked_mapping_evidence"
        )
        if not expected_cross_checks <= cross_checks:
            raise AssertionError(
                f"audit missed {case_name} cross-artifact checks {expected_cross_checks - cross_checks}: {audit}"
            )

        manifest = read_json_object(out_dir / "full-stack-artifact-manifest.json")
        manifest_artifacts = {
            artifact.get("logical_path")
            for artifact in manifest.get("artifacts", [])
            if isinstance(artifact, dict)
        }
        for logical_path in (
            f"{case_name}-dfg-sim-report.json",
            f"{case_name}-cgra-sim-report.json",
        ):
            if logical_path not in manifest_artifacts:
                raise AssertionError(f"manifest missed {logical_path}: {manifest}")
        edges = {(edge["from"], edge["to"]) for edge in manifest.get("edges", [])}
        required_edges = {
            (f"{case_name}-dfg-sim-report", "sim-cycle-summary"),
            (f"{case_name}-cgra-sim-report", "sim-cycle-summary"),
            ("pnr-mapping", "rtl-manifest"),
            ("rtl-manifest", "rtl-sim-eda-report"),
            ("rtl-sim-eda-report", "rtl-fpa-summary"),
            ("rtl-sim-eda-report", "rtl-fpa-report"),
            ("sim-comparison-report", "runtime-package"),
            ("dse-candidate-summary", "workload-report-bundle"),
        }
        missing_edges = required_edges - edges
        if missing_edges:
            raise AssertionError(f"manifest missed {case_name} dependency edges {missing_edges}: {edges}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    for case_name, expected in CASES.items():
        assert_case(repo, case_name, expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
