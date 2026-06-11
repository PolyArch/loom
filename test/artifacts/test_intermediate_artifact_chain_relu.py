#!/usr/bin/env python3
"""Regression test for the relu full-stack artifact chain."""

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
    "pnr-mapping-main.json",
    "pnr-mapping-checksum.json",
    "pnr-mapping.json",
    "relu-dfg-sim-main.report.json",
    "relu-dfg-sim-checksum.report.json",
    "relu-dfg-sim-report.json",
    "relu-dfg-sim-cycle-summary.csv",
    "relu-cgra-sim-main-report.json",
    "relu-cgra-sim-checksum-report.json",
    "relu-cgra-sim-report.json",
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
    "artifact-audit-summary.json",
]

MAIN_MAPPING_ID = "relu__g_t_relu_0_0__shared_reduction_adg"
CHECKSUM_MAPPING_ID = "relu__g_t_main_red_0_0__shared_reduction_adg"
AGGREGATE_MAPPING_ID = "relu__workload_graph_set__shared_reduction_adg"
EXPECTED_MAPPING_IDS = {MAIN_MAPPING_ID, CHECKSUM_MAPPING_ID}
EXPECTED_GRAPHS = {"g_t_relu_0_0", "g_t_main_red_0_0"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise AssertionError(f"{label} should be a positive integer, got {value!r}")
    return value


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-relu-chain-") as tmp:
        out_dir = Path(tmp)
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--case",
                "relu",
            ],
            "relu intermediate artifact chain",
        )

        missing = [name for name in EXPECTED_FILES if not (out_dir / name).is_file()]
        if missing:
            raise AssertionError(f"missing relu chain artifacts: {missing}")

        mapping_rows = read_csv_rows(out_dir / "pnr-mapping-summary.csv")
        relu_rows = [row for row in mapping_rows if row["workload"] == "relu"]
        if len(relu_rows) != 1:
            raise AssertionError(f"expected one aggregate relu mapping row, got {mapping_rows}")
        mapping_row = relu_rows[0]
        if mapping_row["mapping_id"] != AGGREGATE_MAPPING_ID or mapping_row["status"] != "blocked":
            raise AssertionError(f"unexpected relu aggregate mapping row: {mapping_row}")
        if mapping_row["placed_records"] != "10" or mapping_row["routed_edges"] != "0":
            raise AssertionError(f"relu aggregate mapping missed component placement evidence: {mapping_row}")
        if mapping_row["unrouted_edges"] != "12" or mapping_row["unplaced_records"] != "0":
            raise AssertionError(f"relu aggregate mapping should preserve unrouted edges: {mapping_row}")

        mapping_artifact = json.loads((out_dir / "pnr-mapping.json").read_text())
        if mapping_artifact.get("graph") != "workload_graph_set":
            raise AssertionError(f"relu aggregate mapping should be graph-set scoped: {mapping_artifact}")
        if mapping_artifact.get("mapping_id") != AGGREGATE_MAPPING_ID:
            raise AssertionError(f"unexpected relu aggregate mapping id: {mapping_artifact}")
        if set(mapping_artifact.get("component_mapping_ids", [])) != EXPECTED_MAPPING_IDS:
            raise AssertionError(f"relu aggregate mapping missed component mappings: {mapping_artifact}")
        if set(mapping_artifact.get("component_graphs", [])) != EXPECTED_GRAPHS:
            raise AssertionError(f"relu aggregate mapping missed component graphs: {mapping_artifact}")
        if mapping_artifact.get("config_records") != 0 or mapping_artifact.get("status") != "blocked":
            raise AssertionError(f"relu aggregate mapping should preserve blocked route evidence: {mapping_artifact}")

        component_expectations = {
            "pnr-mapping-main.json": ("g_t_relu_0_0", MAIN_MAPPING_ID),
            "pnr-mapping-checksum.json": ("g_t_main_red_0_0", CHECKSUM_MAPPING_ID),
        }
        for name, (graph, mapping_id) in component_expectations.items():
            component = json.loads((out_dir / name).read_text())
            if component.get("graph") != graph or component.get("mapping_id") != mapping_id:
                raise AssertionError(f"unexpected relu component mapping {name}: {component}")
            if component.get("status") != "fail":
                raise AssertionError(f"relu component mapping {name} should expose unrouted failure: {component}")

        dfg_report = json.loads((out_dir / "relu-dfg-sim-report.json").read_text())
        dfg_cycles = positive_int(dfg_report.get("optimistic_cycles"), "relu DFG-sim cycles")
        if dfg_cycles != 707:
            raise AssertionError(f"relu aggregate DFG cycles should include checksum slice: {dfg_report}")
        if set(dfg_report.get("component_graphs", [])) != EXPECTED_GRAPHS:
            raise AssertionError(f"relu aggregate DFG report missed component graphs: {dfg_report}")

        cgra_report = json.loads((out_dir / "relu-cgra-sim-report.json").read_text())
        cgra_cycles = positive_int(cgra_report.get("hardware_aware_cycles"), "relu CGRA-sim cycles")
        if cgra_report.get("mapping_id") != AGGREGATE_MAPPING_ID:
            raise AssertionError(f"unexpected relu CGRA mapping identity: {cgra_report}")
        if cgra_report.get("status") != "blocked" or cgra_cycles != 707:
            raise AssertionError(f"relu aggregate CGRA report should be blocked at DFG cycles: {cgra_report}")
        if cgra_cycles < dfg_cycles:
            raise AssertionError(f"relu CGRA-sim must not be more optimistic than DFG-sim: {cgra_report}")
        if dfg_cycles in {448, 546, 579, 1027, 1603} or cgra_cycles in {466, 576, 589, 1044, 1631}:
            raise AssertionError("relu cycles should remain distinct from existing full-chain workload evidence")
        if set(cgra_report.get("component_mapping_ids", [])) != EXPECTED_MAPPING_IDS:
            raise AssertionError(f"relu aggregate CGRA report missed component mappings: {cgra_report}")

        runtime_package = json.loads((out_dir / "runtime-package.json").read_text())
        if runtime_package.get("status") != "blocked" or runtime_package.get("workload") != "relu":
            raise AssertionError(f"unexpected relu runtime package: {runtime_package}")
        if runtime_package.get("work_package_identity") != f"work-package::relu::{AGGREGATE_MAPPING_ID}":
            raise AssertionError(f"unexpected relu work package identity: {runtime_package}")
        memory_descriptors = runtime_package.get("memory_descriptors")
        if not isinstance(memory_descriptors, list) or len(memory_descriptors) != 1:
            raise AssertionError(f"relu runtime package needs one memory descriptor: {runtime_package}")
        descriptor = memory_descriptors[0]
        expected_descriptor = {
            "logical_argument": "relu.default_input",
            "host_buffer_identity": "runtime-buffer::relu::default_input",
            "runtime_input_identity": "test-app-fixture::relu::default",
            "byte_size": 256,
            "element_layout": "f32[32];f32[32]",
            "alignment_bytes": 4,
            "coherence_requirement": "simulator_consistent",
            "transfer_policy": "simulated",
        }
        for key, value in expected_descriptor.items():
            if descriptor.get(key) != value:
                raise AssertionError(f"unexpected relu memory descriptor {key}: {runtime_package}")

        sim_rows = read_csv_rows(out_dir / "sim-cycle-summary.csv")
        relu_sim = [row for row in sim_rows if row["kernel"] == "relu"]
        if len(relu_sim) != 1:
            raise AssertionError(f"expected one relu sim row, got {sim_rows}")
        if relu_sim[0]["dfg_sim_cycles"] != str(dfg_cycles):
            raise AssertionError(f"relu sim summary missed DFG cycles: {relu_sim[0]}")
        if relu_sim[0]["cgra_sim_cycles"] != "" or relu_sim[0]["status"] != "blocked":
            raise AssertionError(f"relu sim summary should preserve blocked CGRA status: {relu_sim[0]}")

        dse_rows = read_csv_rows(out_dir / "dse-candidate-summary.csv")
        relu_dse = [row for row in dse_rows if row["workload"] == "relu"]
        if len(relu_dse) != 1:
            raise AssertionError(f"expected one relu DSE row, got {dse_rows}")
        dse_row = relu_dse[0]
        if dse_row["mapping_id"] != AGGREGATE_MAPPING_ID or dse_row["selection_status"] != "blocked":
            raise AssertionError(f"unexpected relu DSE row: {dse_row}")
        if dse_row["cgra_sim_cycles"] != "" or dse_row["energy_nj"] != "":
            raise AssertionError(f"blocked relu DSE row must not expose objective metrics: {dse_row}")

        workload_bundle = json.loads((out_dir / "workload-report-bundle.json").read_text())
        if workload_bundle.get("report_status") != "blocked" or workload_bundle.get("workload") != "relu":
            raise AssertionError(f"unexpected relu workload report bundle: {workload_bundle}")
        metric_ids = {
            metric.get("metric_id")
            for metric in workload_bundle.get("metric_records", [])
            if isinstance(metric, dict)
        }
        for metric_id in (
            "metric::relu::dfg_sim_cycles",
            "metric::relu::workload_size_items",
            "metric::shared_reduction_adg::frequency_mhz",
        ):
            if metric_id not in metric_ids:
                raise AssertionError(f"relu workload report missed {metric_id}: {workload_bundle}")

        hardware_bundle = json.loads((out_dir / "hardware-report-bundle.json").read_text())
        if hardware_bundle.get("supported_workload_classes") != ["relu"]:
            raise AssertionError(f"hardware report should cite relu FPA support: {hardware_bundle}")
        fpa_rows = read_csv_rows(out_dir / "rtl-fpa-summary.csv")
        relu_fpa = [row for row in fpa_rows if row["workload"] == "relu"]
        if len(relu_fpa) != 1:
            raise AssertionError(f"expected one relu FPA row, got {fpa_rows}")
        if relu_fpa[0]["status"] != "pass" or relu_fpa[0]["fidelity_level"] != "analytic":
            raise AssertionError(f"relu FPA evidence should stay analytic and passing: {relu_fpa[0]}")

        manifest = json.loads((out_dir / "full-stack-artifact-manifest.json").read_text())
        edges = {
            (edge.get("from"), edge.get("to"))
            for edge in manifest.get("edges", [])
            if isinstance(edge, dict)
        }
        required_edges = {
            ("pnr-mapping-main", "pnr-mapping"),
            ("pnr-mapping-checksum", "pnr-mapping"),
            ("relu-dfg-sim-main.report", "relu-dfg-sim-report"),
            ("relu-dfg-sim-checksum.report", "relu-dfg-sim-report"),
            ("relu-cgra-sim-main-report", "relu-cgra-sim-report"),
            ("relu-cgra-sim-checksum-report", "relu-cgra-sim-report"),
            ("pnr-mapping", "relu-cgra-sim-report"),
            ("pnr-mapping", "rtl-manifest"),
        }
        missing_edges = sorted(required_edges - edges)
        if missing_edges:
            raise AssertionError(f"relu manifest missed aggregate edges: {missing_edges}")
        forbidden_dse_edges = {
            ("pnr-mapping-main", "dse-candidate-summary"),
            ("pnr-mapping-checksum", "dse-candidate-summary"),
            ("relu-cgra-sim-main-report", "dse-candidate-summary"),
            ("relu-cgra-sim-checksum-report", "dse-candidate-summary"),
        }
        unexpected_dse_edges = sorted(forbidden_dse_edges & edges)
        if unexpected_dse_edges:
            raise AssertionError(f"relu manifest overstated direct DSE inputs: {unexpected_dse_edges}")

        audit = json.loads((out_dir / "artifact-audit-summary.json").read_text())
        if audit.get("verdict") != "pass":
            raise AssertionError(f"expected relu chain audit pass, got {audit}")
        if audit.get("cross_artifact_findings"):
            raise AssertionError(f"relu chain should not have cross-artifact findings: {audit}")
        cgra_evidence = [
            check
            for check in audit.get("cross_artifact_checks", [])
            if check.get("rule") == "sim_cycle_blocked_mapping_evidence"
            and check.get("workload") == "relu"
        ]
        if not cgra_evidence:
            raise AssertionError(f"relu chain should carry CGRA mapping cross evidence: {audit}")
        if set(cgra_evidence[0].get("component_mapping_ids", [])) != EXPECTED_MAPPING_IDS:
            raise AssertionError(f"relu CGRA evidence should cite both component mappings: {cgra_evidence}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
