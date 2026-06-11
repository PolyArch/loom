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
    "source-compat-summary.csv",
    "compiler-pipeline-summary.csv",
    "dataflow-primitive-coverage.csv",
    "adg-hardware-summary.csv",
    "pnr-mapping-summary.csv",
    "pnr-mapping.json",
    "sim-comparison-report.json",
    "runtime-package.json",
    "sim-cycle-summary.csv",
    "rtl-manifest.json",
    "rtl-eda-report.json",
    "rtl-fpa-summary.csv",
    "workload-report-bundle.json",
    "hardware-report-bundle.json",
    "dse-candidate-summary.csv",
    "dse-report-bundle.json",
    "full-stack-artifact-manifest.json",
    "unsupported-scope-ledger.csv",
    "artifact-audit-summary.json",
]


CASES = {
    "prefix_sum": {
        "graph": "g_t_prefix_sum_red_0_0",
        "mapping_id": "prefix_sum__g_t_prefix_sum_red_0_0__shared_reduction_adg",
        "placed_records": "6",
        "routed_edges": "9",
        "config_records": 90,
        "dfg_cycles": 835,
        "dynamic_work_items": 64,
        "cgra_cycles": 852,
        "byte_size": 512,
        "element_layout": "i32[64];i32[64]",
    },
    "integrate_trapz": {
        "graph": "g_t_integrate_trapz_red_0_0",
        "mapping_id": "integrate_trapz__g_t_integrate_trapz_red_0_0__shared_reduction_adg",
        "placed_records": "15",
        "routed_edges": "25",
        "config_records": 238,
        "dfg_cycles": 299,
        "dynamic_work_items": 8,
        "cgra_cycles": 340,
        "byte_size": 72,
        "element_layout": "f32[9];f32[9]",
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


def assert_case(repo: Path, case_name: str, expected: Mapping[str, object]) -> None:
    with artifact_test_common.repo_temp_dir(repo, f"loom-{case_name}-chain-") as tmp:
        out_dir = Path(tmp)
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
                "hardware": "shared_reduction_adg",
                "mapping_id": expected["mapping_id"],
                "placed_records": expected["placed_records"],
                "routed_edges": expected["routed_edges"],
                "unrouted_edges": "0",
                "unplaced_records": "0",
                "status": "pass",
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
                "config_records": expected["config_records"],
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
                "status": "pass",
                "workload": case_name,
                "mapping_id": expected["mapping_id"],
                "hardware_aware_cycles": expected["cgra_cycles"],
                "difference_classification": "expected_hardware_constraint",
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
                "cgra_sim_cycles": str(expected["cgra_cycles"]),
                "status": "pass",
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
                "status": "pass",
                "workload": case_name,
                "dfg_sim_cycles": expected["dfg_cycles"],
                "cgra_sim_cycles": expected["cgra_cycles"],
                "difference_classification": "expected_hardware_constraint",
            },
            label=f"{case_name} simulation comparison",
        )

        runtime_package = read_json_object(out_dir / "runtime-package.json")
        if runtime_package.get("status") != "pass" or runtime_package.get("workload") != case_name:
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
                "cgra_sim_cycles": str(expected["cgra_cycles"]),
                "selection_status": "selected",
            },
            label=f"{case_name} DSE",
        )

        workload_bundle = read_json_object(out_dir / "workload-report-bundle.json")
        if workload_bundle.get("report_status") != "pass" or workload_bundle.get("workload") != case_name:
            raise AssertionError(f"unexpected {case_name} workload report bundle: {workload_bundle}")
        metric_ids = {
            metric.get("metric_id")
            for metric in workload_bundle.get("metric_records", [])
            if isinstance(metric, dict)
        }
        for metric_id in (
            f"metric::{case_name}::cgra_sim_cycles",
            f"metric::{case_name}::estimated_runtime_us",
            f"metric::{case_name}::energy_nj",
        ):
            if metric_id not in metric_ids:
                raise AssertionError(f"workload report bundle missed {metric_id}: {workload_bundle}")

        hardware_bundle = read_json_object(out_dir / "hardware-report-bundle.json")
        if hardware_bundle.get("supported_workload_classes") != [case_name]:
            raise AssertionError(f"hardware report should cite {case_name} FPA support: {hardware_bundle}")

        audit = read_json_object(out_dir / "artifact-audit-summary.json")
        if audit.get("verdict") != "pass" or audit.get("cross_artifact_findings"):
            raise AssertionError(f"expected {case_name} chain audit pass, got {audit}")

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
