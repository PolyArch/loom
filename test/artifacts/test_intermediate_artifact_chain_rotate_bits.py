#!/usr/bin/env python3
"""Regression test for the rotate_bits full-stack artifact chain."""

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
    "rotate_bits-dfg-sim-report.json",
    "rotate_bits-dfg-sim-cycle-summary.csv",
    "rotate_bits-cgra-sim-report.json",
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

MAPPING_ID = "rotate_bits__g_t_rotate_bits_0_0__shared_reduction_adg"
GRAPH = "g_t_rotate_bits_0_0"
ROUTED_EDGE_REFS = {
    "arith.andi#0.result0->arith.cmpi#0.operand0",
    "arith.cmpi#0.result0->arith.select#0.operand0",
    "arith.select#0.result0->dataflow.store#0.operand2",
    "dataflow.load#0.result0->arith.andi#0.operand0",
    "dataflow.load#0.result0->llvm.intr.fshl#0.operand2",
    "dataflow.load#0.result1->dataflow.sync#0.operand0",
    "dataflow.load#1.result0->arith.select#0.operand1",
    "dataflow.load#1.result0->llvm.intr.fshl#0.operand0",
    "dataflow.load#1.result0->llvm.intr.fshl#0.operand1",
    "dataflow.load#1.result1->dataflow.sync#0.operand1",
    "dataflow.store#0.result0->dataflow.sync#0.operand2",
    "llvm.intr.fshl#0.result0->arith.select#0.operand2",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-rotate-bits-chain-") as tmp:
        out_dir = Path(tmp)
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--case",
                "rotate_bits",
            ],
            "rotate_bits intermediate artifact chain",
        )

        missing = [name for name in EXPECTED_FILES if not (out_dir / name).is_file()]
        if missing:
            raise AssertionError(f"missing rotate_bits chain artifacts: {missing}")

        mapping_rows = read_csv_rows(out_dir / "pnr-mapping-summary.csv")
        rotate_rows = [row for row in mapping_rows if row["workload"] == "rotate_bits"]
        if len(rotate_rows) != 1:
            raise AssertionError(f"expected one rotate_bits mapping row, got {mapping_rows}")
        mapping_row = rotate_rows[0]
        expected_mapping_row = {
            "hardware": "shared_reduction_adg",
            "mapping_id": MAPPING_ID,
            "placed_records": "8",
            "routed_edges": "12",
            "unrouted_edges": "0",
            "unplaced_records": "0",
            "status": "pass",
        }
        for key, value in expected_mapping_row.items():
            if mapping_row[key] != value:
                raise AssertionError(f"unexpected rotate_bits mapping {key}: {mapping_row}")

        mapping_artifact = json.loads((out_dir / "pnr-mapping.json").read_text())
        if mapping_artifact.get("workload") != "rotate_bits" or mapping_artifact.get("graph") != GRAPH:
            raise AssertionError(f"unexpected rotate_bits mapping identity: {mapping_artifact}")
        if mapping_artifact.get("mapping_id") != MAPPING_ID:
            raise AssertionError(f"unexpected rotate_bits mapping id: {mapping_artifact}")
        if (
            mapping_artifact.get("status") != "pass"
            or mapping_artifact.get("config_records") != 246
            or mapping_artifact.get("unrouted_edges") != 0
        ):
            raise AssertionError(f"rotate_bits mapping should preserve full route evidence: {mapping_artifact}")
        actual_edges = {
            route.get("edge_ref")
            for route in mapping_artifact.get("routes", [])
            if isinstance(route, dict)
        }
        if actual_edges != ROUTED_EDGE_REFS:
            raise AssertionError(f"rotate_bits mapping missed routed edge refs: {mapping_artifact}")
        for route in mapping_artifact.get("routes", []):
            segment_kinds = {
                segment.get("segment_kind")
                for segment in route.get("segments", [])
                if isinstance(segment, dict)
            }
            if "resource_edge" not in segment_kinds:
                raise AssertionError(f"rotate_bits route lacks resource-edge evidence: {route}")
        if mapping_artifact.get("unrouted_edge_details") != []:
            raise AssertionError(f"rotate_bits pass mapping should not expose unrouted details: {mapping_artifact}")

        dfg_report = json.loads((out_dir / "rotate_bits-dfg-sim-report.json").read_text())
        if dfg_report.get("status") != "pass" or dfg_report.get("workload") != "rotate_bits":
            raise AssertionError(f"unexpected rotate_bits DFG-sim report: {dfg_report}")
        if dfg_report.get("optimistic_cycles") != 544 or dfg_report.get("dynamic_work_items") != 32:
            raise AssertionError(f"unexpected rotate_bits DFG-sim cycles: {dfg_report}")

        cgra_report = json.loads((out_dir / "rotate_bits-cgra-sim-report.json").read_text())
        if cgra_report.get("status") != "pass" or cgra_report.get("workload") != "rotate_bits":
            raise AssertionError(f"unexpected rotate_bits CGRA-sim report: {cgra_report}")
        if cgra_report.get("mapping_id") != MAPPING_ID:
            raise AssertionError(f"unexpected rotate_bits CGRA mapping identity: {cgra_report}")
        if cgra_report.get("hardware_aware_cycles") != 600:
            raise AssertionError(f"unexpected rotate_bits CGRA-sim cycles: {cgra_report}")
        if cgra_report.get("performance_delta_cycles") != 56 or cgra_report.get("route_segments") != 44:
            raise AssertionError(f"unexpected rotate_bits route cost: {cgra_report}")
        if cgra_report.get("difference_classification") != "expected_hardware_constraint":
            raise AssertionError(f"rotate_bits CGRA report should classify expected hardware cost: {cgra_report}")
        if cgra_report.get("hardware_aware_cycles") < dfg_report.get("optimistic_cycles"):
            raise AssertionError(f"CGRA-sim must not be more optimistic than DFG-sim: {cgra_report}")

        comparison = json.loads((out_dir / "sim-comparison-report.json").read_text())
        if comparison.get("status") != "pass" or comparison.get("workload") != "rotate_bits":
            raise AssertionError(f"unexpected rotate_bits comparison report: {comparison}")
        if comparison.get("difference_classification") != "expected_hardware_constraint":
            raise AssertionError(f"unexpected rotate_bits comparison class: {comparison}")

        sim_rows = read_csv_rows(out_dir / "sim-cycle-summary.csv")
        rotate_sim = [row for row in sim_rows if row["kernel"] == "rotate_bits"]
        if len(rotate_sim) != 1:
            raise AssertionError(f"expected one rotate_bits sim row, got {sim_rows}")
        if rotate_sim[0]["dfg_sim_cycles"] != "544" or rotate_sim[0]["cgra_sim_cycles"] != "600":
            raise AssertionError(f"rotate_bits sim summary missed cycles: {rotate_sim[0]}")
        if rotate_sim[0]["status"] != "pass":
            raise AssertionError(f"rotate_bits sim summary should pass: {rotate_sim[0]}")

        runtime_package = json.loads((out_dir / "runtime-package.json").read_text())
        if runtime_package.get("status") != "blocked" or runtime_package.get("workload") != "rotate_bits":
            raise AssertionError(f"rotate_bits runtime package should stay structured blocked: {runtime_package}")
        if runtime_package.get("work_package_identity") != f"work-package::rotate_bits::{MAPPING_ID}":
            raise AssertionError(f"unexpected rotate_bits work package identity: {runtime_package}")
        if runtime_package.get("memory_descriptors") != []:
            raise AssertionError(f"rotate_bits runtime package should expose missing memory descriptors: {runtime_package}")

        dse_rows = read_csv_rows(out_dir / "dse-candidate-summary.csv")
        rotate_dse = [row for row in dse_rows if row["workload"] == "rotate_bits"]
        if len(rotate_dse) != 1:
            raise AssertionError(f"expected one rotate_bits DSE row, got {dse_rows}")
        dse_row = rotate_dse[0]
        expected_dse = {
            "mapping_id": MAPPING_ID,
            "cgra_sim_cycles": "600",
            "frequency_mhz": "50.000",
            "area_um2": "52250.000",
            "dynamic_power_mw": "42.000",
            "leakage_power_mw": "5.325",
            "energy_nj": "567.900",
            "selection_status": "selected",
            "hardware_evidence_kind": "analytic_model_only",
        }
        for key, value in expected_dse.items():
            if dse_row[key] != value:
                raise AssertionError(f"unexpected rotate_bits DSE {key}: {dse_row}")
        fidelity_records = dse_row.get("feedback_fidelity_records", "")
        if "energy_nj=analytic:derived_from_fpa_and_cgra_sim" not in fidelity_records:
            raise AssertionError(f"rotate_bits DSE row should mark analytic energy fidelity: {dse_row}")

        workload_bundle = json.loads((out_dir / "workload-report-bundle.json").read_text())
        if workload_bundle.get("report_status") != "blocked" or workload_bundle.get("workload") != "rotate_bits":
            raise AssertionError(f"rotate_bits workload bundle should preserve runtime blocker: {workload_bundle}")
        metric_ids = {
            metric.get("metric_id")
            for metric in workload_bundle.get("metric_records", [])
            if isinstance(metric, dict)
        }
        for metric_id in (
            "metric::rotate_bits::dfg_sim_cycles",
            "metric::rotate_bits::workload_size_items",
            "metric::rotate_bits::cgra_sim_cycles",
            "metric::rotate_bits::energy_nj",
            "metric::shared_reduction_adg::frequency_mhz",
        ):
            if metric_id not in metric_ids:
                raise AssertionError(f"rotate_bits workload report missed {metric_id}: {workload_bundle}")

        hardware_bundle = json.loads((out_dir / "hardware-report-bundle.json").read_text())
        if hardware_bundle.get("report_status") != "pass":
            raise AssertionError(f"unexpected rotate_bits hardware report status: {hardware_bundle}")
        if hardware_bundle.get("supported_workload_classes") != ["rotate_bits"]:
            raise AssertionError(f"hardware report should cite rotate_bits FPA support: {hardware_bundle}")
        fpa_rows = read_csv_rows(out_dir / "rtl-fpa-summary.csv")
        rotate_fpa = [row for row in fpa_rows if row["workload"] == "rotate_bits"]
        if len(rotate_fpa) != 1:
            raise AssertionError(f"expected one rotate_bits FPA row, got {fpa_rows}")
        if rotate_fpa[0]["status"] != "pass" or rotate_fpa[0]["fidelity_level"] != "analytic":
            raise AssertionError(f"rotate_bits FPA evidence should stay analytic and passing: {rotate_fpa[0]}")

        manifest = json.loads((out_dir / "full-stack-artifact-manifest.json").read_text())
        manifest_artifacts = {
            artifact.get("logical_path")
            for artifact in manifest.get("artifacts", [])
            if isinstance(artifact, dict)
        }
        for logical_path in (
            "rotate_bits-dfg-sim-report.json",
            "rotate_bits-cgra-sim-report.json",
            "sim-comparison-report.json",
            "runtime-package.json",
            "workload-report-bundle.json",
        ):
            if logical_path not in manifest_artifacts:
                raise AssertionError(f"manifest missed {logical_path}: {manifest}")
        edges = {
            (edge.get("from"), edge.get("to"))
            for edge in manifest.get("edges", [])
            if isinstance(edge, dict)
        }
        required_edges = {
            ("pnr-mapping", "rotate_bits-cgra-sim-report"),
            ("rotate_bits-dfg-sim-report", "sim-cycle-summary"),
            ("rotate_bits-cgra-sim-report", "sim-cycle-summary"),
            ("rotate_bits-dfg-sim-report", "sim-comparison-report"),
            ("rotate_bits-cgra-sim-report", "sim-comparison-report"),
            ("pnr-mapping", "sim-comparison-report"),
            ("pnr-mapping", "runtime-package"),
            ("rotate_bits-cgra-sim-report", "runtime-package"),
            ("sim-comparison-report", "runtime-package"),
            ("runtime-package", "workload-report-bundle"),
            ("rotate_bits-cgra-sim-report", "workload-report-bundle"),
            ("dse-candidate-summary", "workload-report-bundle"),
        }
        missing_edges = sorted(required_edges - edges)
        if missing_edges:
            raise AssertionError(f"rotate_bits manifest missed dependency edges: {missing_edges}")

        audit = json.loads((out_dir / "artifact-audit-summary.json").read_text())
        if audit.get("verdict") != "pass":
            raise AssertionError(f"expected rotate_bits chain audit pass, got {audit}")
        if audit.get("cross_artifact_findings"):
            raise AssertionError(f"rotate_bits chain should not have cross-artifact findings: {audit}")
        cgra_evidence = [
            check
            for check in audit.get("cross_artifact_checks", [])
            if check.get("rule") == "sim_cycle_report_mapping_evidence"
            and check.get("workload") == "rotate_bits"
        ]
        if not cgra_evidence:
            raise AssertionError(f"rotate_bits chain should carry CGRA mapping cross evidence: {audit}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
