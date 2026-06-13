#!/usr/bin/env python3
"""Regression test for the byte_swap full-stack artifact chain."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import artifact_test_common


EXPECTED_FILES = [
    "source-compat-summary.csv",
    "compiler-pipeline-summary.csv",
    "dataflow-primitive-coverage.csv",
    "adg-hardware-summary.csv",
    "pnr-mapping-summary.csv",
    "pnr-mapping.json",
    "byte_swap-dfg-sim-report.json",
    "byte_swap-dfg-sim-cycle-summary.csv",
    "byte_swap-cgra-sim-report.json",
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
    "unsupported-scope-ledger.csv",
    "artifact-audit-summary.json",
]

WORKLOAD = "byte_swap"
GRAPH = "g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0"
HARDWARE = "shared_vector_alu_adg"
MAPPING_ID = f"{WORKLOAD}__{GRAPH}__{HARDWARE}"


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
    matches = [row for row in rows if row[key] == value]
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


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-byte-swap-chain-") as tmp:
        out_dir = Path(tmp)
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--case",
                WORKLOAD,
            ],
            "byte_swap intermediate artifact chain",
        )

        missing = [name for name in EXPECTED_FILES if not (out_dir / name).is_file()]
        if missing:
            raise AssertionError(f"missing byte_swap chain artifacts: {missing}")

        mapping = single_row(
            read_csv_rows(out_dir / "pnr-mapping-summary.csv"),
            key="workload",
            value=WORKLOAD,
            label="byte_swap mapping",
        )
        assert_fields(
            mapping,
            {
                "hardware": HARDWARE,
                "mapping_id": MAPPING_ID,
                "placed_records": "4",
                "routed_edges": "4",
                "unrouted_edges": "0",
                "unplaced_records": "0",
                "status": "pass",
            },
            label="byte_swap mapping",
        )

        mapping_artifact = read_json_object(out_dir / "pnr-mapping.json")
        assert_fields(
            mapping_artifact,
            {
                "workload": WORKLOAD,
                "graph": GRAPH,
                "mapping_id": MAPPING_ID,
                "placed_records": 4,
                "routed_edges": 4,
                "config_records": 85,
                "status": "pass",
            },
            label="byte_swap mapping artifact",
        )

        dfg_report = read_json_object(out_dir / "byte_swap-dfg-sim-report.json")
        assert_fields(
            dfg_report,
            {
                "status": "pass",
                "workload": WORKLOAD,
                "graph": GRAPH,
                "optimistic_cycles": 320,
                "dynamic_work_items": 32,
            },
            label="byte_swap DFG-sim report",
        )
        fire_counts = dfg_report.get("operation_fire_counts", {})
        if not isinstance(fire_counts, dict):
            raise AssertionError(f"byte_swap DFG-sim report lacks operation fire counts: {dfg_report}")
        assert_fields(
            fire_counts,
            {
                "llvm.intr.bswap": 32,
                "dataflow.load": 32,
                "dataflow.store": 32,
                "dataflow.sync": 32,
            },
            label="byte_swap fire count",
        )

        cgra_report = read_json_object(out_dir / "byte_swap-cgra-sim-report.json")
        assert_fields(
            cgra_report,
            {
                "status": "pass",
                "workload": WORKLOAD,
                "mapping_id": MAPPING_ID,
                "dfg_cycles": 320,
                "hardware": HARDWARE,
                "hardware_aware_cycles": 342,
                "difference_classification": "expected_hardware_constraint",
                "route_segments": 14,
            },
            label="byte_swap CGRA-sim report",
        )
        if cgra_report["hardware_aware_cycles"] < dfg_report["optimistic_cycles"]:
            raise AssertionError(f"CGRA-sim must not be more optimistic than DFG-sim: {cgra_report}")

        sim_row = single_row(
            read_csv_rows(out_dir / "sim-cycle-summary.csv"),
            key="kernel",
            value=WORKLOAD,
            label="byte_swap sim",
        )
        assert_fields(
            sim_row,
            {"dfg_sim_cycles": "320", "cgra_sim_cycles": "342", "status": "pass"},
            label="byte_swap sim row",
        )
        if int(sim_row["dfg_sim_cycles"]) in {448, 579, 1027}:
            raise AssertionError(f"byte_swap cycles should add distinct workload evidence: {sim_row}")

        comparison = read_json_object(out_dir / "sim-comparison-report.json")
        assert_fields(
            comparison,
            {
                "status": "pass",
                "workload": WORKLOAD,
                "dfg_sim_cycles": 320,
                "cgra_sim_cycles": 342,
                "difference_classification": "expected_hardware_constraint",
            },
            label="byte_swap simulation comparison",
        )

        runtime_package = read_json_object(out_dir / "runtime-package.json")
        if runtime_package.get("status") != "pass" or runtime_package.get("workload") != WORKLOAD:
            raise AssertionError(f"unexpected byte_swap runtime package: {runtime_package}")
        if runtime_package.get("work_package_identity") != f"work-package::{WORKLOAD}::{MAPPING_ID}":
            raise AssertionError(f"unexpected byte_swap work package identity: {runtime_package}")
        memory_descriptors = runtime_package.get("memory_descriptors")
        if not isinstance(memory_descriptors, list) or len(memory_descriptors) != 1:
            raise AssertionError(f"byte_swap runtime package needs one memory descriptor: {runtime_package}")
        assert_fields(
            memory_descriptors[0],
            {
                "logical_argument": "byte_swap.default_input",
                "host_buffer_identity": "runtime-buffer::byte_swap::default_input",
                "policy": "simulated",
                "runtime_input_identity": "test-app-fixture::byte_swap::default",
                "layout_source_kind": "static_workload_fixture",
                "layout_source_identity": "test-app-fixture::byte_swap::default",
                "byte_size": 256,
                "element_layout": "u32[32];u32[32]",
                "alignment_bytes": 4,
                "address_space": "simulator::memory_model",
                "coherence_requirement": "simulator_consistent",
                "transfer_policy": "simulated",
            },
            label="byte_swap memory descriptor",
        )

        dse_row = single_row(
            read_csv_rows(out_dir / "dse-candidate-summary.csv"),
            key="workload",
            value=WORKLOAD,
            label="byte_swap DSE",
        )
        assert_fields(
            dse_row,
            {
                "mapping_id": MAPPING_ID,
                "cgra_sim_cycles": "342",
                "frequency_mhz": "420.000",
                "area_um2": "3000.000",
                "dynamic_power_mw": "2.600",
                "energy_nj": "2.443",
                "selection_status": "selected",
            },
            label="byte_swap DSE",
        )
        if "rtl-fpa-summary" not in dse_row.get("input_artifacts", ""):
            raise AssertionError(f"byte_swap DSE should consume FPA evidence: {dse_row}")

        workload_bundle = read_json_object(out_dir / "workload-report-bundle.json")
        if workload_bundle.get("report_status") != "pass" or workload_bundle.get("workload") != WORKLOAD:
            raise AssertionError(f"unexpected byte_swap workload report bundle: {workload_bundle}")
        metric_ids = {
            metric.get("metric_id")
            for metric in workload_bundle.get("metric_records", [])
            if isinstance(metric, dict)
        }
        for metric_id in (
            "metric::byte_swap::dfg_sim_cycles",
            "metric::byte_swap::workload_size_items",
            "metric::byte_swap::cgra_sim_cycles",
            "metric::shared_vector_alu_adg::frequency_mhz",
        ):
            if metric_id not in metric_ids:
                raise AssertionError(f"workload report bundle missed {metric_id}: {workload_bundle}")

        hardware_bundle = read_json_object(out_dir / "hardware-report-bundle.json")
        if hardware_bundle.get("supported_workload_classes") != [WORKLOAD]:
            raise AssertionError(f"hardware report should cite byte_swap FPA support: {hardware_bundle}")
        fpa_row = single_row(
            read_csv_rows(out_dir / "rtl-fpa-summary.csv"),
            key="workload",
            value=WORKLOAD,
            label="byte_swap FPA",
        )
        if fpa_row.get("fidelity_level") != "analytic" or fpa_row.get("status") != "pass":
            raise AssertionError(f"byte_swap FPA evidence should stay analytic and passing: {fpa_row}")
        for source_key in ("frequency_source", "area_source", "power_source"):
            if fpa_row.get(source_key) != "analytic_fpa_model":
                raise AssertionError(f"byte_swap FPA source drifted: {fpa_row}")

        manifest = read_json_object(out_dir / "full-stack-artifact-manifest.json")
        manifest_artifacts = {
            artifact.get("logical_path")
            for artifact in manifest.get("artifacts", [])
            if isinstance(artifact, dict)
        }
        for logical_path in ("byte_swap-dfg-sim-report.json", "byte_swap-cgra-sim-report.json"):
            if logical_path not in manifest_artifacts:
                raise AssertionError(f"manifest missed {logical_path}: {manifest}")
        edges = {(edge["from"], edge["to"]) for edge in manifest.get("edges", [])}
        required_edges = {
            ("byte_swap-dfg-sim-report", "sim-cycle-summary"),
            ("byte_swap-cgra-sim-report", "sim-cycle-summary"),
            ("pnr-mapping", "rtl-manifest"),
            ("rtl-manifest", "rtl-sim-eda-report"),
            ("rtl-sim-eda-report", "rtl-fpa-summary"),
            ("rtl-sim-eda-report", "rtl-fpa-report"),
            ("sim-comparison-report", "runtime-package"),
            ("runtime-package", "workload-report-bundle"),
            ("rtl-fpa-report", "hardware-report-bundle"),
            ("dse-candidate-summary", "workload-report-bundle"),
        }
        missing_edges = required_edges - edges
        if missing_edges:
            raise AssertionError(f"manifest missed byte_swap dependency edges {missing_edges}: {edges}")

        audit = read_json_object(out_dir / "artifact-audit-summary.json")
        if audit.get("verdict") != "pass" or audit.get("cross_artifact_findings"):
            raise AssertionError(f"expected byte_swap chain audit pass, got {audit}")
        cross_checks = {
            check.get("rule")
            for check in audit.get("cross_artifact_checks", [])
            if isinstance(check, dict)
        }
        expected_cross_checks = {
            "sim_cycle_dfg_report_evidence",
            "sim_cycle_report_mapping_evidence",
        }
        if not expected_cross_checks <= cross_checks:
            raise AssertionError(
                f"audit missed byte_swap cross-artifact checks {expected_cross_checks - cross_checks}: {audit}"
            )

        missing_layout_source = out_dir / "missing-layout-source-runtime-package.json"
        missing_layout_source_data = read_json_object(out_dir / "runtime-package.json")
        missing_layout_source_data["memory_descriptors"][0].pop("layout_source_identity", None)
        missing_layout_source.write_text(
            json.dumps(missing_layout_source_data, indent=2, sort_keys=True) + "\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "missing-layout-source-audit.json"),
                str(missing_layout_source),
            ],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0:
            raise AssertionError("runtime package without memory layout source unexpectedly passed audit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
