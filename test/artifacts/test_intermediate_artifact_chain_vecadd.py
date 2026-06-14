#!/usr/bin/env python3
"""Regression test for the vecadd full-stack artifact chain."""

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
    "pnr-mapping-reduction.json",
    "pnr-mapping.json",
    "vecadd-dfg-sim-main.report.json",
    "vecadd-dfg-sim-main.reduction.report.json",
    "vecadd-dfg-sim-report.json",
    "vecadd-dfg-sim-cycle-summary.csv",
    "vecadd-cgra-sim-main-report.json",
    "vecadd-cgra-sim-reduction-report.json",
    "vecadd-cgra-sim-report.json",
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

AUDIT_INPUT_FILES = [
    "old-app-corpus-inventory.csv",
    "app-corpus-import-status.csv",
    "source-compat-summary.csv",
    "compiler-pipeline-summary.csv",
    "cmsis-compiler-pipeline-summary.csv",
    "dataflow-primitive-coverage.csv",
    "adg-hardware-summary.csv",
    "pnr-mapping-summary.csv",
    "vecadd-dfg-sim-main.report.json",
    "vecadd-dfg-sim-main.reduction.report.json",
    "pnr-mapping-main.json",
    "pnr-mapping-reduction.json",
    "vecadd-cgra-sim-main-report.json",
    "vecadd-cgra-sim-reduction-report.json",
    "pnr-mapping.json",
    "vecadd-dfg-sim-report.json",
    "vecadd-dfg-sim-cycle-summary.csv",
    "vecadd-cgra-sim-report.json",
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
    "dse-report-bundle.json",
    "full-stack-artifact-manifest.json",
    "e2e-demonstrator-summary.csv",
    "dse-candidate-summary.csv",
    "unsupported-scope-ledger.csv",
]

MAIN_MAPPING_ID = "vecadd__g_t_vecadd_0_0__shared_reduction_adg"
REDUCTION_MAPPING_ID = "vecadd__g_t_main_red_0_0__shared_reduction_adg"
AGGREGATE_MAPPING_ID = "vecadd__workload_graph_set__shared_reduction_adg"
EXPECTED_MAPPING_IDS = {MAIN_MAPPING_ID, REDUCTION_MAPPING_ID}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise AssertionError(f"{label} should be a positive integer, got {value!r}")
    return value


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-vecadd-chain-") as tmp:
        out_dir = Path(tmp)
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--case",
                "vecadd",
            ],
            "vecadd intermediate artifact chain",
        )

        missing = [name for name in EXPECTED_FILES if not (out_dir / name).is_file()]
        if missing:
            raise AssertionError(f"missing vecadd chain artifacts: {missing}")

        mapping_rows = read_csv_rows(out_dir / "pnr-mapping-summary.csv")
        vecadd_mapping_rows = [row for row in mapping_rows if row["workload"] == "vecadd"]
        if len(vecadd_mapping_rows) != 1:
            raise AssertionError(f"expected one aggregate vecadd mapping row, got {mapping_rows}")
        mapping = vecadd_mapping_rows[0]
        expected_mapping = {
            "hardware": "shared_reduction_adg",
            "mapping_id": AGGREGATE_MAPPING_ID,
            "routed_edges": "12",
            "unrouted_edges": "0",
            "unplaced_records": "0",
            "status": "pass",
        }
        for key, value in expected_mapping.items():
            if mapping[key] != value:
                raise AssertionError(f"unexpected vecadd mapping {key}: {mapping}")
        if int(mapping["placed_records"]) <= 0:
            raise AssertionError(f"vecadd mapping should carry placement evidence: {mapping}")

        mapping_artifact = json.loads((out_dir / "pnr-mapping.json").read_text())
        expected_mapping_artifact = {
            "workload": "vecadd",
            "graph": "workload_graph_set",
            "hardware": "shared_reduction_adg",
            "mapping_id": AGGREGATE_MAPPING_ID,
            "status": "pass",
        }
        for key, value in expected_mapping_artifact.items():
            if mapping_artifact.get(key) != value:
                raise AssertionError(f"unexpected vecadd mapping artifact {key}: {mapping_artifact}")
        if mapping_artifact.get("config_records") != 232:
            raise AssertionError(f"vecadd mapping should aggregate component config: {mapping_artifact}")
        component_mapping_ids = set(mapping_artifact.get("component_mapping_ids", []))
        if component_mapping_ids != EXPECTED_MAPPING_IDS:
            raise AssertionError(f"aggregate mapping must cite both component mappings: {mapping_artifact}")

        component_expectations = {
            "pnr-mapping-main.json": ("g_t_vecadd_0_0", MAIN_MAPPING_ID, "pass", 6, 0, 111),
            "pnr-mapping-reduction.json": ("g_t_main_red_0_0", REDUCTION_MAPPING_ID, "pass", 6, 0, 121),
        }
        for name, (graph, mapping_id, status, routed_edges, unrouted_edges, config_records) in component_expectations.items():
            component = json.loads((out_dir / name).read_text())
            if component.get("graph") != graph or component.get("mapping_id") != mapping_id:
                raise AssertionError(f"unexpected component mapping {name}: {component}")
            if (
                component.get("status") != status
                or component.get("routed_edges") != routed_edges
                or component.get("unrouted_edges") != unrouted_edges
                or component.get("config_records") != config_records
            ):
                raise AssertionError(f"unexpected component routing state for {name}: {component}")
            component_unrouted_details = component.get("unrouted_edge_details", [])
            if status == "fail" and not component_unrouted_details:
                raise AssertionError(f"failing component should expose unrouted details: {component}")
            if status == "pass" and component_unrouted_details:
                raise AssertionError(f"passing component should not expose unrouted details: {component}")

        dfg_report = json.loads((out_dir / "vecadd-dfg-sim-report.json").read_text())
        if dfg_report.get("status") != "pass" or dfg_report.get("workload") != "vecadd":
            raise AssertionError(f"unexpected vecadd DFG-sim report: {dfg_report}")
        dfg_cycles = positive_int(dfg_report.get("optimistic_cycles"), "vecadd DFG-sim cycles")
        if dfg_cycles != 1603:
            raise AssertionError(f"vecadd aggregate DFG cycles should include checksum reduction tail: {dfg_report}")
        if set(dfg_report.get("component_graphs", [])) != {"g_t_vecadd_0_0", "g_t_main_red_0_0"}:
            raise AssertionError(f"aggregate DFG report must cite both component graphs: {dfg_report}")
        dfg_final_outputs = dfg_report.get("final_outputs")
        if not isinstance(dfg_final_outputs, list) or not dfg_final_outputs:
            raise AssertionError(f"aggregate DFG report must expose final outputs: {dfg_report}")
        dfg_final_memory = dfg_report.get("final_memory_state")
        if not isinstance(dfg_final_memory, dict):
            raise AssertionError(f"aggregate DFG report must expose final memory state: {dfg_report}")

        cgra_report = json.loads((out_dir / "vecadd-cgra-sim-report.json").read_text())
        if cgra_report.get("status") != "pass" or cgra_report.get("workload") != "vecadd":
            raise AssertionError(f"unexpected vecadd CGRA-sim report: {cgra_report}")
        if cgra_report.get("mapping_id") != AGGREGATE_MAPPING_ID:
            raise AssertionError(f"unexpected vecadd CGRA mapping identity: {cgra_report}")
        cgra_cycles = positive_int(cgra_report.get("hardware_aware_cycles"), "vecadd CGRA-sim cycles")
        if cgra_cycles != 1657:
            raise AssertionError(f"vecadd aggregate CGRA report should include both component latencies: {cgra_report}")
        if cgra_report.get("performance_delta_cycles") != 54 or cgra_report.get("route_segments") != 38:
            raise AssertionError(f"vecadd aggregate CGRA report should expose routed component cost: {cgra_report}")
        if cgra_cycles < dfg_cycles:
            raise AssertionError(f"CGRA-sim must not be more optimistic than DFG-sim: {cgra_report}")
        if dfg_cycles in {579, 1027, 448} or cgra_cycles in {589, 1044, 466}:
            raise AssertionError(f"vecadd cycles should differ from vecsum/dotproduct/xor_block evidence")
        if set(cgra_report.get("component_mapping_ids", [])) != EXPECTED_MAPPING_IDS:
            raise AssertionError(f"aggregate CGRA report must cite both component mappings: {cgra_report}")
        if (
            cgra_report.get("functional_state_source")
            != "component_cgra_sim_reports_carried_from_dfg_sim_reports"
        ):
            raise AssertionError(f"aggregate CGRA report must label carried functional state: {cgra_report}")
        if cgra_report.get("final_outputs") != dfg_final_outputs:
            raise AssertionError(f"aggregate CGRA report must expose matching final outputs: {cgra_report}")
        if cgra_report.get("final_memory_state") != dfg_final_memory:
            raise AssertionError(f"aggregate CGRA report must expose matching final memory state: {cgra_report}")

        runtime_package = json.loads((out_dir / "runtime-package.json").read_text())
        if runtime_package.get("status") != "pass" or runtime_package.get("workload") != "vecadd":
            raise AssertionError(f"unexpected vecadd runtime package: {runtime_package}")
        if runtime_package.get("work_package_identity") != f"work-package::vecadd::{AGGREGATE_MAPPING_ID}":
            raise AssertionError(f"unexpected vecadd work package identity: {runtime_package}")
        memory_descriptors = runtime_package.get("memory_descriptors")
        if not isinstance(memory_descriptors, list) or len(memory_descriptors) != 1:
            raise AssertionError(f"vecadd runtime package needs one memory descriptor: {runtime_package}")
        memory_descriptor = memory_descriptors[0]
        expected_descriptor = {
            "logical_argument": "vecadd.default_input",
            "host_buffer_identity": "runtime-buffer::vecadd::default_input",
            "policy": "simulated",
            "runtime_input_identity": "test-app-fixture::vecadd::default",
            "byte_size": 768,
            "element_layout": "f32[64];f32[64];f32[64]",
            "alignment_bytes": 4,
            "address_space": "simulator::memory_model",
            "coherence_requirement": "simulator_consistent",
            "transfer_policy": "simulated",
        }
        for key, value in expected_descriptor.items():
            if memory_descriptor.get(key) != value:
                raise AssertionError(f"unexpected vecadd memory descriptor {key}: {runtime_package}")

        sim_rows = read_csv_rows(out_dir / "sim-cycle-summary.csv")
        vecadd_sim_rows = [row for row in sim_rows if row["kernel"] == "vecadd"]
        if len(vecadd_sim_rows) != 1:
            raise AssertionError(f"expected one vecadd sim row, got {sim_rows}")
        sim_row = vecadd_sim_rows[0]
        if sim_row["dfg_sim_cycles"] != str(dfg_cycles) or sim_row["cgra_sim_cycles"] != str(cgra_cycles):
            raise AssertionError(f"vecadd sim summary should expose CGRA status: {sim_row}")
        if sim_row["status"] != "pass":
            raise AssertionError(f"unexpected vecadd sim summary status: {sim_row}")

        dse_rows = read_csv_rows(out_dir / "dse-candidate-summary.csv")
        vecadd_dse_rows = [row for row in dse_rows if row["workload"] == "vecadd"]
        if len(vecadd_dse_rows) != 1:
            raise AssertionError(f"expected one vecadd DSE row, got {dse_rows}")
        vecadd_dse = vecadd_dse_rows[0]
        if vecadd_dse["mapping_id"] != AGGREGATE_MAPPING_ID or vecadd_dse["selection_status"] != "selected":
            raise AssertionError(f"unexpected vecadd DSE row: {vecadd_dse}")
        if vecadd_dse["cgra_sim_cycles"] != str(cgra_cycles) or vecadd_dse["energy_nj"] == "":
            raise AssertionError(f"selected vecadd DSE row must expose objective metrics: {vecadd_dse}")

        workload_bundle = json.loads((out_dir / "workload-report-bundle.json").read_text())
        if workload_bundle.get("report_status") != "pass" or workload_bundle.get("workload") != "vecadd":
            raise AssertionError(f"unexpected vecadd workload report bundle: {workload_bundle}")
        metric_ids = {
            metric.get("metric_id")
            for metric in workload_bundle.get("metric_records", [])
            if isinstance(metric, dict)
        }
        for metric_id in (
            "metric::vecadd::dfg_sim_cycles",
            "metric::vecadd::workload_size_items",
            "metric::shared_reduction_adg::frequency_mhz",
        ):
            if metric_id not in metric_ids:
                raise AssertionError(f"workload report bundle missed {metric_id}: {workload_bundle}")

        hardware_bundle = json.loads((out_dir / "hardware-report-bundle.json").read_text())
        if hardware_bundle.get("supported_workload_classes") != ["vecadd"]:
            raise AssertionError(f"hardware report should cite vecadd FPA support: {hardware_bundle}")

        fpa_rows = read_csv_rows(out_dir / "rtl-fpa-summary.csv")
        vecadd_fpa_rows = [row for row in fpa_rows if row["workload"] == "vecadd"]
        if len(vecadd_fpa_rows) != 1:
            raise AssertionError(f"expected one vecadd FPA row, got {fpa_rows}")
        vecadd_fpa = vecadd_fpa_rows[0]
        if vecadd_fpa["status"] != "pass" or vecadd_fpa["fidelity_level"] != "analytic":
            raise AssertionError(f"vecadd FPA evidence should stay analytic and passing: {vecadd_fpa}")

        audit = json.loads((out_dir / "artifact-audit-summary.json").read_text())
        if audit.get("verdict") != "pass":
            raise AssertionError(f"expected vecadd chain audit pass, got {audit}")
        if audit.get("cross_artifact_findings"):
            raise AssertionError(f"vecadd chain should not have cross-artifact findings: {audit}")
        cgra_evidence = [
            check
            for check in audit.get("cross_artifact_checks", [])
            if check.get("rule") == "sim_cycle_report_mapping_evidence"
            and check.get("workload") == "vecadd"
        ]
        if not cgra_evidence:
            raise AssertionError(f"vecadd chain should carry CGRA mapping cross evidence: {audit}")
        mapping_ids = set(cgra_evidence[0].get("component_mapping_ids", []))
        if mapping_ids != EXPECTED_MAPPING_IDS:
            raise AssertionError(f"vecadd CGRA evidence should cite both component mappings: {cgra_evidence}")

        manifest = json.loads((out_dir / "full-stack-artifact-manifest.json").read_text())
        edges = {
            (edge.get("from"), edge.get("to"))
            for edge in manifest.get("edges", [])
            if isinstance(edge, dict)
        }
        required_edges = {
            ("pnr-mapping-main", "pnr-mapping"),
            ("pnr-mapping-reduction", "pnr-mapping"),
            ("vecadd-dfg-sim-main.report", "vecadd-dfg-sim-report"),
            ("vecadd-dfg-sim-main.reduction.report", "vecadd-dfg-sim-report"),
            ("vecadd-cgra-sim-main-report", "vecadd-cgra-sim-report"),
            ("vecadd-cgra-sim-reduction-report", "vecadd-cgra-sim-report"),
            ("pnr-mapping", "vecadd-cgra-sim-report"),
            ("pnr-mapping", "rtl-manifest"),
            ("rtl-manifest", "rtl-sim-eda-report"),
            ("rtl-sim-eda-report", "rtl-fpa-summary"),
            ("rtl-sim-eda-report", "rtl-fpa-report"),
        }
        missing_edges = sorted(required_edges - edges)
        if missing_edges:
            raise AssertionError(f"vecadd manifest missed aggregate producer-consumer edges: {missing_edges}")
        impossible_edges = {
            ("pnr-mapping-main", "vecadd-cgra-sim-reduction-report"),
            ("pnr-mapping-reduction", "vecadd-cgra-sim-main-report"),
            ("pnr-mapping-main", "vecadd-cgra-sim-report"),
            ("pnr-mapping-reduction", "vecadd-cgra-sim-report"),
        }
        present_impossible_edges = sorted(impossible_edges & edges)
        if present_impossible_edges:
            raise AssertionError(f"vecadd manifest carried non-consuming edges: {present_impossible_edges}")

        tampered_cgra = json.loads((out_dir / "vecadd-cgra-sim-report.json").read_text())
        fingerprints = tampered_cgra.get("input_artifact_fingerprints")
        if not isinstance(fingerprints, dict):
            raise AssertionError(f"aggregate CGRA report needs input fingerprints: {tampered_cgra}")
        fingerprints["vecadd-dfg-sim-main.report"] = "0" * 64
        (out_dir / "vecadd-cgra-sim-report.json").write_text(
            json.dumps(tampered_cgra, indent=2, sort_keys=True) + "\n"
        )
        tampered_audit = out_dir / "tampered-artifact-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(tampered_audit),
                str(out_dir / "vecadd-dfg-sim-main.report.json"),
                str(out_dir / "vecadd-dfg-sim-main.reduction.report.json"),
                str(out_dir / "vecadd-cgra-sim-report.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("tampered aggregate CGRA DFG fingerprint should fail artifact audit")
        tampered_summary = json.loads(tampered_audit.read_text())
        diagnostics = "\n".join(str(item) for item in tampered_summary.get("diagnostics", []))
        if "CGRA simulator aggregate report input_artifact_fingerprints stale" not in diagnostics:
            raise AssertionError(f"tampered audit failed for the wrong reason: {tampered_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
