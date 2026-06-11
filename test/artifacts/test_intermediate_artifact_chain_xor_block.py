#!/usr/bin/env python3
"""Regression test for the xor_block full-stack artifact chain."""

from __future__ import annotations

import csv
import json
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
    "xor_block-dfg-sim-report.json",
    "xor_block-dfg-sim-cycle-summary.csv",
    "xor_block-cgra-sim-report.json",
    "sim-comparison-report.json",
    "runtime-package.json",
    "sim-cycle-summary.csv",
    "rtl-manifest.json",
    "rtl-eda-report.json",
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
    with artifact_test_common.repo_temp_dir(repo, "loom-xor-block-chain-") as tmp:
        out_dir = Path(tmp)
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--case",
                "xor_block",
            ],
            "xor_block intermediate artifact chain",
        )

        missing = [name for name in EXPECTED_FILES if not (out_dir / name).is_file()]
        if missing:
            raise AssertionError(f"missing xor_block chain artifacts: {missing}")

        mapping = single_row(
            read_csv_rows(out_dir / "pnr-mapping-summary.csv"),
            key="workload",
            value="xor_block",
            label="xor_block mapping",
        )
        assert_fields(
            mapping,
            {
                "hardware": "shared_reduction_adg",
                "mapping_id": "xor_block__g_t_xor_block_0_0__shared_reduction_adg",
                "placed_records": "5",
                "routed_edges": "6",
                "unrouted_edges": "0",
                "unplaced_records": "0",
                "status": "pass",
            },
            label="xor_block mapping",
        )

        mapping_artifact = read_json_object(out_dir / "pnr-mapping.json")
        assert_fields(
            mapping_artifact,
            {
                "workload": "xor_block",
                "graph": "g_t_xor_block_0_0",
                "mapping_id": "xor_block__g_t_xor_block_0_0__shared_reduction_adg",
            },
            label="xor_block mapping artifact",
        )
        if int(mapping_artifact.get("config_records", 0)) <= 0:
            raise AssertionError(f"mapping artifact should carry real config records: {mapping_artifact}")

        dfg_report = read_json_object(out_dir / "xor_block-dfg-sim-report.json")
        if dfg_report.get("status") != "pass" or dfg_report.get("workload") != "xor_block":
            raise AssertionError(f"unexpected xor_block DFG-sim report: {dfg_report}")
        if dfg_report.get("optimistic_cycles") != 448 or dfg_report.get("dynamic_work_items") != 32:
            raise AssertionError(f"unexpected xor_block DFG-sim cycles: {dfg_report}")
        fire_counts = dfg_report.get("operation_fire_counts", {})
        if not isinstance(fire_counts, dict):
            raise AssertionError(f"xor_block DFG-sim report lacks operation fire counts: {dfg_report}")
        expected_fire_counts = {
            "arith.xori": 32,
            "dataflow.load": 64,
            "dataflow.store": 32,
            "dataflow.sync": 32,
        }
        assert_fields(fire_counts, expected_fire_counts, label="xor_block fire count")

        cgra_report = read_json_object(out_dir / "xor_block-cgra-sim-report.json")
        if cgra_report.get("status") != "pass" or cgra_report.get("workload") != "xor_block":
            raise AssertionError(f"unexpected xor_block CGRA-sim report: {cgra_report}")
        if cgra_report.get("mapping_id") != "xor_block__g_t_xor_block_0_0__shared_reduction_adg":
            raise AssertionError(f"unexpected xor_block CGRA mapping identity: {cgra_report}")
        if cgra_report.get("hardware_aware_cycles") != 466:
            raise AssertionError(f"unexpected xor_block CGRA-sim cycles: {cgra_report}")
        if cgra_report.get("hardware_aware_cycles") < dfg_report.get("optimistic_cycles"):
            raise AssertionError(f"CGRA-sim must not be more optimistic than DFG-sim: {cgra_report}")

        sim_row = single_row(
            read_csv_rows(out_dir / "sim-cycle-summary.csv"),
            key="kernel",
            value="xor_block",
            label="xor_block sim",
        )
        assert_fields(
            sim_row,
            {"dfg_sim_cycles": "448", "cgra_sim_cycles": "466", "status": "pass"},
            label="xor_block sim row",
        )
        if int(sim_row["dfg_sim_cycles"]) in {579, 1027}:
            raise AssertionError(f"xor_block cycles should differ from existing vecsum/dotproduct evidence: {sim_row}")

        comparison = read_json_object(out_dir / "sim-comparison-report.json")
        if comparison.get("status") != "pass" or comparison.get("workload") != "xor_block":
            raise AssertionError(f"unexpected xor_block comparison report: {comparison}")
        if comparison.get("dfg_sim_cycles") != 448 or comparison.get("cgra_sim_cycles") != 466:
            raise AssertionError(f"comparison should preserve xor_block cycles: {comparison}")
        if comparison.get("difference_classification") != "expected_hardware_constraint":
            raise AssertionError(f"comparison should classify hardware constraint difference: {comparison}")

        runtime_package = read_json_object(out_dir / "runtime-package.json")
        if runtime_package.get("status") != "pass" or runtime_package.get("workload") != "xor_block":
            raise AssertionError(f"unexpected xor_block runtime package: {runtime_package}")
        if runtime_package.get("work_package_identity") != (
            "work-package::xor_block::xor_block__g_t_xor_block_0_0__shared_reduction_adg"
        ):
            raise AssertionError(f"unexpected xor_block work package identity: {runtime_package}")
        memory_descriptors = runtime_package.get("memory_descriptors")
        if not isinstance(memory_descriptors, list) or len(memory_descriptors) != 1:
            raise AssertionError(f"xor_block runtime package needs one memory descriptor: {runtime_package}")
        memory_descriptor = memory_descriptors[0]
        expected_descriptor_fields = {
            "logical_argument": "xor_block.default_input",
            "host_buffer_identity": "runtime-buffer::xor_block::default_input",
            "policy": "simulated",
            "runtime_input_identity": "test-app-fixture::xor_block::default",
            "byte_size": 384,
            "element_layout": "u32[32];u32[32];u32[32]",
            "alignment_bytes": 4,
            "address_space": "simulator::memory_model",
            "coherence_requirement": "simulator_consistent",
            "transfer_policy": "simulated",
        }
        assert_fields(
            memory_descriptor,
            expected_descriptor_fields,
            label="xor_block memory descriptor",
        )

        xor_dse = single_row(
            read_csv_rows(out_dir / "dse-candidate-summary.csv"),
            key="workload",
            value="xor_block",
            label="xor_block DSE",
        )
        assert_fields(
            xor_dse,
            {
                "mapping_id": "xor_block__g_t_xor_block_0_0__shared_reduction_adg",
                "cgra_sim_cycles": "466",
                "frequency_mhz": "250.000",
                "area_um2": "7250.000",
                "dynamic_power_mw": "6.000",
                "energy_nj": "12.722",
                "selection_status": "selected",
            },
            label="xor_block DSE",
        )
        if "rtl-fpa-summary" not in xor_dse.get("input_artifacts", ""):
            raise AssertionError(f"xor_block DSE should consume FPA evidence: {xor_dse}")

        workload_bundle = read_json_object(out_dir / "workload-report-bundle.json")
        if workload_bundle.get("report_status") != "pass" or workload_bundle.get("workload") != "xor_block":
            raise AssertionError(f"unexpected xor_block workload report bundle: {workload_bundle}")
        metric_ids = {
            metric.get("metric_id")
            for metric in workload_bundle.get("metric_records", [])
            if isinstance(metric, dict)
        }
        for metric_id in (
            "metric::xor_block::cgra_sim_cycles",
            "metric::xor_block::estimated_runtime_us",
            "metric::xor_block::energy_nj",
        ):
            if metric_id not in metric_ids:
                raise AssertionError(f"workload report bundle missed {metric_id}: {workload_bundle}")

        hardware_bundle = read_json_object(out_dir / "hardware-report-bundle.json")
        if hardware_bundle.get("supported_workload_classes") != ["xor_block"]:
            raise AssertionError(f"hardware report should cite xor_block FPA support: {hardware_bundle}")
        xor_fpa = single_row(
            read_csv_rows(out_dir / "rtl-fpa-summary.csv"),
            key="workload",
            value="xor_block",
            label="xor_block FPA",
        )
        if xor_fpa.get("fidelity_level") != "analytic" or xor_fpa.get("status") != "pass":
            raise AssertionError(f"xor_block FPA evidence should stay analytic and passing: {xor_fpa}")
        for source_key in ("frequency_source", "area_source", "power_source"):
            if xor_fpa.get(source_key) != "analytic_fpa_model":
                raise AssertionError(f"xor_block FPA source drifted: {xor_fpa}")

        audit = read_json_object(out_dir / "artifact-audit-summary.json")
        if audit.get("verdict") != "pass":
            raise AssertionError(f"expected xor_block chain audit pass, got {audit}")
        if audit.get("cross_artifact_findings"):
            raise AssertionError(f"xor_block chain should not have cross-artifact findings: {audit}")

        manifest = read_json_object(out_dir / "full-stack-artifact-manifest.json")
        manifest_artifacts = {
            artifact.get("logical_path")
            for artifact in manifest.get("artifacts", [])
            if isinstance(artifact, dict)
        }
        for logical_path in ("xor_block-dfg-sim-report.json", "xor_block-cgra-sim-report.json"):
            if logical_path not in manifest_artifacts:
                raise AssertionError(f"manifest missed {logical_path}: {manifest}")
        edges = {(edge["from"], edge["to"]) for edge in manifest.get("edges", [])}
        required_edges = {
            ("xor_block-dfg-sim-report", "sim-cycle-summary"),
            ("xor_block-cgra-sim-report", "sim-cycle-summary"),
            ("pnr-mapping", "rtl-manifest"),
            ("runtime-package", "workload-report-bundle"),
            ("rtl-fpa-report", "hardware-report-bundle"),
        }
        missing_edges = required_edges - edges
        if missing_edges:
            raise AssertionError(f"manifest missed xor_block dependency edges {missing_edges}: {edges}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
