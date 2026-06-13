#!/usr/bin/env python3
"""Regression test for a non-vecsum full-stack artifact chain."""

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
    "pnr-mapping.json",
    "dotproduct-dfg-sim-report.json",
    "dotproduct-dfg-sim-cycle-summary.csv",
    "dotproduct-cgra-sim-report.json",
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-dotproduct-chain-") as tmp:
        out_dir = Path(tmp)
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--case",
                "dotproduct",
            ],
            "dotproduct intermediate artifact chain",
        )

        missing = [name for name in EXPECTED_FILES if not (out_dir / name).is_file()]
        if missing:
            raise AssertionError(f"missing dotproduct chain artifacts: {missing}")

        mapping_rows = read_csv_rows(out_dir / "pnr-mapping-summary.csv")
        dotproduct_mapping_rows = [row for row in mapping_rows if row["workload"] == "dotproduct"]
        if len(dotproduct_mapping_rows) != 1:
            raise AssertionError(f"expected one dotproduct mapping row, got {mapping_rows}")
        mapping = dotproduct_mapping_rows[0]
        expected_mapping = {
            "hardware": "shared_reduction_adg",
            "mapping_id": "dotproduct__g_t_dotproduct_red_0_0__shared_reduction_adg",
            "placed_records": "6",
            "routed_edges": "9",
            "unrouted_edges": "0",
            "unplaced_records": "0",
            "status": "pass",
        }
        for key, value in expected_mapping.items():
            if mapping[key] != value:
                raise AssertionError(f"unexpected dotproduct mapping {key}: {mapping}")

        mapping_artifact = json.loads((out_dir / "pnr-mapping.json").read_text())
        if mapping_artifact.get("workload") != "dotproduct":
            raise AssertionError(f"mapping artifact should carry dotproduct workload: {mapping_artifact}")
        if mapping_artifact.get("graph") != "g_t_dotproduct_red_0_0":
            raise AssertionError(f"mapping artifact should carry dotproduct graph: {mapping_artifact}")
        if mapping_artifact.get("config_records") != 170 or mapping_artifact.get("status") != "pass":
            raise AssertionError(f"mapping artifact should expose routed evidence: {mapping_artifact}")

        dfg_report = json.loads((out_dir / "dotproduct-dfg-sim-report.json").read_text())
        if dfg_report.get("status") != "pass" or dfg_report.get("workload") != "dotproduct":
            raise AssertionError(f"unexpected dotproduct DFG-sim report: {dfg_report}")
        if dfg_report.get("optimistic_cycles") != 1219 or dfg_report.get("dynamic_work_items") != 64:
            raise AssertionError(f"unexpected dotproduct DFG-sim cycles: {dfg_report}")

        cgra_report = json.loads((out_dir / "dotproduct-cgra-sim-report.json").read_text())
        if cgra_report.get("status") != "pass" or cgra_report.get("workload") != "dotproduct":
            raise AssertionError(f"unexpected dotproduct CGRA-sim report: {cgra_report}")
        if cgra_report.get("mapping_id") != "dotproduct__g_t_dotproduct_red_0_0__shared_reduction_adg":
            raise AssertionError(f"unexpected dotproduct CGRA mapping identity: {cgra_report}")
        if cgra_report.get("hardware_aware_cycles") != 1256:
            raise AssertionError(f"unexpected dotproduct CGRA-sim cycles: {cgra_report}")
        if cgra_report.get("difference_classification") != "expected_hardware_constraint":
            raise AssertionError(f"dotproduct CGRA report should classify expected hardware cost: {cgra_report}")
        if cgra_report.get("hardware_aware_cycles") < dfg_report.get("optimistic_cycles"):
            raise AssertionError(f"CGRA-sim must not be more optimistic than DFG-sim: {cgra_report}")

        sim_rows = read_csv_rows(out_dir / "sim-cycle-summary.csv")
        dotproduct_sim_rows = [row for row in sim_rows if row["kernel"] == "dotproduct"]
        if len(dotproduct_sim_rows) != 1:
            raise AssertionError(f"expected one dotproduct sim row, got {sim_rows}")
        sim_row = dotproduct_sim_rows[0]
        expected_cycles = {"dfg_sim_cycles": "1219", "cgra_sim_cycles": "1256", "status": "pass"}
        for key, value in expected_cycles.items():
            if sim_row[key] != value:
                raise AssertionError(f"unexpected dotproduct sim row {key}: {sim_row}")

        comparison = json.loads((out_dir / "sim-comparison-report.json").read_text())
        if comparison.get("status") != "pass" or comparison.get("workload") != "dotproduct":
            raise AssertionError(f"unexpected dotproduct comparison report: {comparison}")
        if comparison.get("dfg_sim_cycles") != 1219 or comparison.get("cgra_sim_cycles") != 1256:
            raise AssertionError(f"comparison should preserve dotproduct cycles: {comparison}")

        runtime_package = json.loads((out_dir / "runtime-package.json").read_text())
        if runtime_package.get("status") != "pass" or runtime_package.get("workload") != "dotproduct":
            raise AssertionError(f"unexpected dotproduct runtime package: {runtime_package}")
        if runtime_package.get("work_package_identity") != (
            "work-package::dotproduct::dotproduct__g_t_dotproduct_red_0_0__shared_reduction_adg"
        ):
            raise AssertionError(f"unexpected dotproduct work package identity: {runtime_package}")
        memory_descriptors = runtime_package.get("memory_descriptors")
        if not isinstance(memory_descriptors, list) or len(memory_descriptors) != 1:
            raise AssertionError(f"dotproduct runtime package needs one memory descriptor: {runtime_package}")
        memory_descriptor = memory_descriptors[0]
        expected_descriptor_fields = {
            "logical_argument": "dotproduct.default_input",
            "host_buffer_identity": "runtime-buffer::dotproduct::default_input",
            "policy": "simulated",
            "runtime_input_identity": "test-app-fixture::dotproduct::default",
            "byte_size": 512,
            "element_layout": "f32[64];f32[64]",
            "alignment_bytes": 4,
            "address_space": "simulator::memory_model",
            "coherence_requirement": "simulator_consistent",
            "transfer_policy": "simulated",
        }
        for key, value in expected_descriptor_fields.items():
            if memory_descriptor.get(key) != value:
                raise AssertionError(f"unexpected dotproduct memory descriptor {key}: {runtime_package}")

        dse_rows = read_csv_rows(out_dir / "dse-candidate-summary.csv")
        dotproduct_dse_rows = [row for row in dse_rows if row["workload"] == "dotproduct"]
        if len(dotproduct_dse_rows) != 1:
            raise AssertionError(f"expected one dotproduct DSE row, got {dse_rows}")
        dotproduct_dse = dotproduct_dse_rows[0]
        expected_dse = {
            "mapping_id": "dotproduct__g_t_dotproduct_red_0_0__shared_reduction_adg",
            "cgra_sim_cycles": "1256",
            "selection_status": "selected",
        }
        for key, value in expected_dse.items():
            if dotproduct_dse[key] != value:
                raise AssertionError(f"unexpected dotproduct DSE {key}: {dotproduct_dse}")

        workload_bundle = json.loads((out_dir / "workload-report-bundle.json").read_text())
        if workload_bundle.get("report_status") != "pass" or workload_bundle.get("workload") != "dotproduct":
            raise AssertionError(f"unexpected dotproduct workload report bundle: {workload_bundle}")
        metric_ids = {
            metric.get("metric_id")
            for metric in workload_bundle.get("metric_records", [])
            if isinstance(metric, dict)
        }
        for metric_id in (
            "metric::dotproduct::dfg_sim_cycles",
            "metric::dotproduct::workload_size_items",
            "metric::shared_reduction_adg::frequency_mhz",
        ):
            if metric_id not in metric_ids:
                raise AssertionError(f"workload report bundle missed {metric_id}: {workload_bundle}")

        hardware_bundle = json.loads((out_dir / "hardware-report-bundle.json").read_text())
        if hardware_bundle.get("supported_workload_classes") != ["dotproduct"]:
            raise AssertionError(f"hardware report should cite dotproduct FPA support: {hardware_bundle}")
        eda_report = json.loads((out_dir / "rtl-eda-report.json").read_text())
        sim_eda_report = json.loads((out_dir / "rtl-sim-eda-report.json").read_text())
        if eda_report.get("kind") != "eda_report" or eda_report.get("capability_class") != "rtl_lint":
            raise AssertionError(f"unexpected dotproduct RTL EDA report: {eda_report}")
        if sim_eda_report.get("kind") != "eda_report" or sim_eda_report.get("capability_class") != "rtl_sim":
            raise AssertionError(f"unexpected dotproduct RTL sim EDA report: {sim_eda_report}")
        expected_eda_identities = []
        if eda_report.get("status") == "pass":
            expected_eda_identities.append("rtl-eda-report")
        elif eda_report.get("status") != "blocked":
            raise AssertionError(f"unexpected dotproduct RTL EDA report status: {eda_report}")
        if sim_eda_report.get("status") == "pass":
            expected_eda_identities.append("rtl-sim-eda-report")
        elif sim_eda_report.get("status") != "blocked":
            raise AssertionError(f"unexpected dotproduct RTL sim EDA report status: {sim_eda_report}")
        if hardware_bundle.get("eda_report_identities") != expected_eda_identities:
            raise AssertionError(f"hardware report EDA evidence mismatch: {hardware_bundle}")

        audit = json.loads((out_dir / "artifact-audit-summary.json").read_text())
        if audit.get("verdict") != "pass":
            raise AssertionError(f"expected dotproduct chain audit pass, got {audit}")
        if audit.get("cross_artifact_findings"):
            raise AssertionError(f"dotproduct chain should not have cross-artifact findings: {audit}")

        manifest = json.loads((out_dir / "full-stack-artifact-manifest.json").read_text())
        manifest_artifacts = {
            artifact.get("logical_path")
            for artifact in manifest.get("artifacts", [])
            if isinstance(artifact, dict)
        }
        if "dotproduct-dfg-sim-report.json" not in manifest_artifacts:
            raise AssertionError(f"manifest missed dotproduct DFG-sim report: {manifest}")
        if "dotproduct-cgra-sim-report.json" not in manifest_artifacts:
            raise AssertionError(f"manifest missed dotproduct CGRA-sim report: {manifest}")
        edges = {(edge["from"], edge["to"]) for edge in manifest.get("edges", [])}
        required_edges = {
            ("pnr-mapping", "rtl-manifest"),
            ("rtl-manifest", "rtl-sim-eda-report"),
            ("rtl-sim-eda-report", "rtl-fpa-summary"),
            ("rtl-sim-eda-report", "rtl-fpa-report"),
        }
        if not required_edges.issubset(edges):
            raise AssertionError(f"manifest missed RTL sim dependency edges {required_edges - edges}: {edges}")
        eda_to_hardware_edge = ("rtl-eda-report", "hardware-report-bundle")
        if "rtl-eda-report" in hardware_bundle.get("eda_report_identities", []):
            if eda_to_hardware_edge not in edges:
                raise AssertionError(f"manifest missed consumed EDA report edge: {edges}")
        elif eda_to_hardware_edge in edges:
            raise AssertionError(f"manifest must not feed blocked EDA report to hardware bundle: {edges}")
        sim_eda_to_hardware_edge = ("rtl-sim-eda-report", "hardware-report-bundle")
        if "rtl-sim-eda-report" in hardware_bundle.get("eda_report_identities", []):
            if sim_eda_to_hardware_edge not in edges:
                raise AssertionError(f"manifest missed consumed sim EDA report edge: {edges}")
        elif sim_eda_to_hardware_edge in edges:
            raise AssertionError(f"manifest must not feed blocked sim EDA report to hardware bundle: {edges}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
