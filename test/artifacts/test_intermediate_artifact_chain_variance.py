#!/usr/bin/env python3
"""Regression test for the variance full-stack artifact chain."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import artifact_test_common


EXPECTED_FILES = [
    "source-compat-summary.csv",
    "compiler-pipeline-summary.csv",
    "dataflow-primitive-coverage.csv",
    "adg-hardware-summary.csv",
    "pnr-mapping-summary.csv",
    "pnr-mapping-mean.json",
    "pnr-mapping-var.json",
    "pnr-mapping.json",
    "variance-dfg-sim-mean.report.json",
    "variance-dfg-sim-var.report.json",
    "variance-dfg-sim-report.json",
    "variance-dfg-sim-cycle-summary.csv",
    "variance-cgra-sim-mean-report.json",
    "variance-cgra-sim-var-report.json",
    "variance-cgra-sim-report.json",
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
    "artifact-audit-summary.json",
]

MEAN_MAPPING_ID = "variance__g_t_variance_red_0_0__shared_reduction_adg"
VAR_MAPPING_ID = "variance__g_t_variance_red_1_0__shared_reduction_adg"
AGGREGATE_MAPPING_ID = "variance__workload_graph_set__shared_reduction_adg"
EXPECTED_MAPPING_IDS = {MEAN_MAPPING_ID, VAR_MAPPING_ID}
EXPECTED_GRAPHS = {"g_t_variance_red_0_0", "g_t_variance_red_1_0"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise AssertionError(f"{label} should be a positive integer, got {value!r}")
    return value


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-variance-chain-") as tmp:
        out_dir = Path(tmp)
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--case",
                "variance",
            ],
            "variance intermediate artifact chain",
        )

        missing = [name for name in EXPECTED_FILES if not (out_dir / name).is_file()]
        if missing:
            raise AssertionError(f"missing variance chain artifacts: {missing}")

        mapping_rows = read_csv_rows(out_dir / "pnr-mapping-summary.csv")
        variance_rows = [row for row in mapping_rows if row["workload"] == "variance"]
        if len(variance_rows) != 1:
            raise AssertionError(f"expected one aggregate variance mapping row, got {mapping_rows}")
        mapping_row = variance_rows[0]
        if mapping_row["mapping_id"] != AGGREGATE_MAPPING_ID or mapping_row["status"] != "pass":
            raise AssertionError(f"unexpected variance aggregate mapping row: {mapping_row}")
        if mapping_row["routed_edges"] != "22" or mapping_row["unrouted_edges"] != "0" or mapping_row["unplaced_records"] != "0":
            raise AssertionError(f"variance aggregate mapping should expose fully routed evidence: {mapping_row}")

        mapping_artifact = json.loads((out_dir / "pnr-mapping.json").read_text())
        if mapping_artifact.get("graph") != "workload_graph_set":
            raise AssertionError(f"variance aggregate mapping should be graph-set scoped: {mapping_artifact}")
        if mapping_artifact.get("mapping_id") != AGGREGATE_MAPPING_ID:
            raise AssertionError(f"unexpected variance aggregate mapping id: {mapping_artifact}")
        if set(mapping_artifact.get("component_mapping_ids", [])) != EXPECTED_MAPPING_IDS:
            raise AssertionError(f"variance aggregate mapping missed component mappings: {mapping_artifact}")
        if set(mapping_artifact.get("component_graphs", [])) != EXPECTED_GRAPHS:
            raise AssertionError(f"variance aggregate mapping missed component graphs: {mapping_artifact}")
        if (
            mapping_artifact.get("status") != "pass"
            or mapping_artifact.get("routed_edges") != 22
            or mapping_artifact.get("unrouted_edges") != 0
            or mapping_artifact.get("config_records") != 472
        ):
            raise AssertionError(f"variance aggregate mapping should expose fully routed evidence: {mapping_artifact}")

        component_expectations = {
            "pnr-mapping-mean.json": ("g_t_variance_red_0_0", MEAN_MAPPING_ID, "pass", 9, 0, 190),
            "pnr-mapping-var.json": ("g_t_variance_red_1_0", VAR_MAPPING_ID, "pass", 13, 0, 282),
        }
        for name, (graph, mapping_id, status, routed_edges, unrouted_edges, config_records) in component_expectations.items():
            component = json.loads((out_dir / name).read_text())
            if component.get("graph") != graph or component.get("mapping_id") != mapping_id:
                raise AssertionError(f"unexpected variance component mapping {name}: {component}")
            if (
                component.get("status") != status
                or component.get("routed_edges") != routed_edges
                or component.get("unrouted_edges") != unrouted_edges
                or component.get("config_records") != config_records
            ):
                raise AssertionError(f"unexpected variance component mapping status {name}: {component}")

        dfg_report = json.loads((out_dir / "variance-dfg-sim-report.json").read_text())
        dfg_cycles = positive_int(dfg_report.get("optimistic_cycles"), "variance DFG-sim cycles")
        if dfg_cycles != 659:
            raise AssertionError(f"variance aggregate DFG cycles should include both passes: {dfg_report}")
        if set(dfg_report.get("component_graphs", [])) != EXPECTED_GRAPHS:
            raise AssertionError(f"variance aggregate DFG report missed component graphs: {dfg_report}")

        cgra_report = json.loads((out_dir / "variance-cgra-sim-report.json").read_text())
        cgra_cycles = positive_int(cgra_report.get("hardware_aware_cycles"), "variance CGRA-sim cycles")
        if cgra_report.get("mapping_id") != AGGREGATE_MAPPING_ID:
            raise AssertionError(f"unexpected variance CGRA mapping identity: {cgra_report}")
        if cgra_report.get("status") != "pass" or cgra_cycles != 772:
            raise AssertionError(f"variance aggregate CGRA report should preserve routed component cost: {cgra_report}")
        if cgra_report.get("routed_edges") != 22 or cgra_report.get("config_records") != 472:
            raise AssertionError(f"variance aggregate CGRA report should expose fully routed evidence: {cgra_report}")
        if cgra_cycles < dfg_cycles:
            raise AssertionError(f"variance CGRA-sim must not be more optimistic than DFG-sim: {cgra_report}")
        if set(cgra_report.get("component_mapping_ids", [])) != EXPECTED_MAPPING_IDS:
            raise AssertionError(f"variance aggregate CGRA report missed component mappings: {cgra_report}")

        runtime_package = json.loads((out_dir / "runtime-package.json").read_text())
        if runtime_package.get("status") != "pass" or runtime_package.get("workload") != "variance":
            raise AssertionError(f"unexpected variance runtime package: {runtime_package}")
        if runtime_package.get("work_package_identity") != f"work-package::variance::{AGGREGATE_MAPPING_ID}":
            raise AssertionError(f"unexpected variance work package identity: {runtime_package}")
        memory_descriptors = runtime_package.get("memory_descriptors")
        if not isinstance(memory_descriptors, list) or len(memory_descriptors) != 1:
            raise AssertionError(f"variance runtime package needs one memory descriptor: {runtime_package}")
        descriptor = memory_descriptors[0]
        expected_descriptor = {
            "logical_argument": "variance.default_input",
            "host_buffer_identity": "runtime-buffer::variance::default_input",
            "runtime_input_identity": "test-app-fixture::variance::default",
            "byte_size": 64,
            "element_layout": "f32[16]",
            "alignment_bytes": 4,
            "coherence_requirement": "simulator_consistent",
            "transfer_policy": "simulated",
        }
        for key, value in expected_descriptor.items():
            if descriptor.get(key) != value:
                raise AssertionError(f"unexpected variance memory descriptor {key}: {runtime_package}")

        sim_rows = read_csv_rows(out_dir / "sim-cycle-summary.csv")
        variance_sim = [row for row in sim_rows if row["kernel"] == "variance"]
        if len(variance_sim) != 1:
            raise AssertionError(f"expected one variance sim row, got {sim_rows}")
        if variance_sim[0]["dfg_sim_cycles"] != str(dfg_cycles):
            raise AssertionError(f"variance sim summary missed DFG cycles: {variance_sim[0]}")
        if variance_sim[0]["cgra_sim_cycles"] != str(cgra_cycles) or variance_sim[0]["status"] != "pass":
            raise AssertionError(f"variance sim summary should preserve passing CGRA status: {variance_sim[0]}")

        dse_rows = read_csv_rows(out_dir / "dse-candidate-summary.csv")
        variance_dse = [row for row in dse_rows if row["workload"] == "variance"]
        if len(variance_dse) != 1:
            raise AssertionError(f"expected one variance DSE row, got {dse_rows}")
        dse_row = variance_dse[0]
        if dse_row["mapping_id"] != AGGREGATE_MAPPING_ID or dse_row["selection_status"] != "selected":
            raise AssertionError(f"unexpected variance DSE row: {dse_row}")
        expected_dse = {
            "cgra_sim_cycles": "772",
            "frequency_mhz": "50.000",
            "area_um2": "59250.000",
            "dynamic_power_mw": "47.600",
            "leakage_power_mw": "6.025",
            "energy_nj": "827.970",
            "hardware_evidence_kind": "analytic_model_only",
        }
        for key, value in expected_dse.items():
            if dse_row[key] != value:
                raise AssertionError(f"unexpected variance DSE {key}: {dse_row}")
        metric_records = {entry for entry in dse_row.get("metric_records", "").split(";") if entry}
        required_dse_metrics = {
            "cgra_sim_cycles=772",
            "frequency_mhz=50.000",
            "area_um2=59250.000",
            "dynamic_power_mw=47.600",
            "leakage_power_mw=6.025",
            "energy_nj=827.970",
        }
        if not required_dse_metrics.issubset(metric_records):
            raise AssertionError(f"selected variance DSE row missed objective metrics: {dse_row}")

        workload_bundle = json.loads((out_dir / "workload-report-bundle.json").read_text())
        if workload_bundle.get("report_status") != "pass" or workload_bundle.get("workload") != "variance":
            raise AssertionError(f"unexpected variance workload report bundle: {workload_bundle}")
        metric_ids = {
            metric.get("metric_id")
            for metric in workload_bundle.get("metric_records", [])
            if isinstance(metric, dict)
        }
        for metric_id in (
            "metric::variance::dfg_sim_cycles",
            "metric::variance::cgra_sim_cycles",
            "metric::variance::energy_nj",
            "metric::variance::workload_size_items",
            "metric::shared_reduction_adg::frequency_mhz",
        ):
            if metric_id not in metric_ids:
                raise AssertionError(f"variance workload report missed {metric_id}: {workload_bundle}")

        hardware_bundle = json.loads((out_dir / "hardware-report-bundle.json").read_text())
        if hardware_bundle.get("supported_workload_classes") != ["variance"]:
            raise AssertionError(f"hardware report should cite variance FPA support: {hardware_bundle}")
        fpa_rows = read_csv_rows(out_dir / "rtl-fpa-summary.csv")
        variance_fpa = [row for row in fpa_rows if row["workload"] == "variance"]
        if len(variance_fpa) != 1:
            raise AssertionError(f"expected one variance FPA row, got {fpa_rows}")
        if variance_fpa[0]["status"] != "pass" or variance_fpa[0]["fidelity_level"] != "analytic":
            raise AssertionError(f"variance FPA evidence should stay analytic and passing: {variance_fpa[0]}")

        manifest = json.loads((out_dir / "full-stack-artifact-manifest.json").read_text())
        edges = {
            (edge.get("from"), edge.get("to"))
            for edge in manifest.get("edges", [])
            if isinstance(edge, dict)
        }
        required_edges = {
            ("pnr-mapping-mean", "pnr-mapping"),
            ("pnr-mapping-var", "pnr-mapping"),
            ("variance-dfg-sim-mean.report", "variance-dfg-sim-report"),
            ("variance-dfg-sim-var.report", "variance-dfg-sim-report"),
            ("variance-cgra-sim-mean-report", "variance-cgra-sim-report"),
            ("variance-cgra-sim-var-report", "variance-cgra-sim-report"),
            ("pnr-mapping", "variance-cgra-sim-report"),
            ("pnr-mapping", "dse-candidate-summary"),
            ("variance-cgra-sim-report", "dse-candidate-summary"),
            ("pnr-mapping", "rtl-manifest"),
            ("rtl-manifest", "rtl-sim-eda-report"),
            ("rtl-sim-eda-report", "rtl-fpa-summary"),
            ("rtl-sim-eda-report", "rtl-fpa-report"),
        }
        missing_edges = sorted(required_edges - edges)
        if missing_edges:
            raise AssertionError(f"variance manifest missed aggregate edges: {missing_edges}")
        forbidden_dse_edges = {
            ("pnr-mapping-mean", "dse-candidate-summary"),
            ("pnr-mapping-var", "dse-candidate-summary"),
            ("variance-cgra-sim-mean-report", "dse-candidate-summary"),
            ("variance-cgra-sim-var-report", "dse-candidate-summary"),
        }
        unexpected_dse_edges = sorted(forbidden_dse_edges & edges)
        if unexpected_dse_edges:
            raise AssertionError(f"variance manifest overstated direct DSE inputs: {unexpected_dse_edges}")

        audit = json.loads((out_dir / "artifact-audit-summary.json").read_text())
        if audit.get("verdict") != "pass":
            raise AssertionError(f"expected variance chain audit pass, got {audit}")
        if audit.get("cross_artifact_findings"):
            raise AssertionError(f"variance chain should not have cross-artifact findings: {audit}")
        cgra_evidence = [
            check
            for check in audit.get("cross_artifact_checks", [])
            if check.get("rule") == "sim_cycle_report_mapping_evidence"
            and check.get("workload") == "variance"
        ]
        if not cgra_evidence:
            raise AssertionError(f"variance chain should carry CGRA mapping cross evidence: {audit}")
        if set(cgra_evidence[0].get("component_mapping_ids", [])) != EXPECTED_MAPPING_IDS:
            raise AssertionError(f"variance CGRA evidence should cite both component mappings: {cgra_evidence}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
