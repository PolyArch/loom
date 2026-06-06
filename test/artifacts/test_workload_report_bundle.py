#!/usr/bin/env python3
"""Regression test for workload report bundle provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "bundle_id",
    "workload",
    "source_artifact_identity",
    "compiler_command_identity",
    "runtime_input_identity",
    "selected_hardware_candidate_identity",
    "selected_mapping_artifact_identity",
    "runtime_fallback_decision",
    "report_status",
    "diagnostics",
    "metric_records",
}


def metric_by_id(metrics: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(metric.get("metric_id")): metric for metric in metrics}


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-workload-report-") as tmp:
        out_dir = Path(tmp)
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
            ],
            "intermediate artifact chain",
        )

        report = out_dir / "workload-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_report_bundle.sh",
                "--output",
                str(report),
                "--artifact",
                str(out_dir / "source-compat-summary.csv"),
                "--artifact",
                str(out_dir / "compiler-pipeline-summary.csv"),
                "--artifact",
                str(out_dir / "dataflow-primitive-coverage.csv"),
                "--artifact",
                str(out_dir / "adg-hardware-summary.csv"),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(out_dir / "vecsum-dfg-sim-report.json"),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "sim-comparison-report.json"),
                "--artifact",
                str(out_dir / "runtime-package.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
            ],
            "workload report bundle",
        )

        data = json.loads(report.read_text())
        missing = REQUIRED_KEYS - set(data)
        if missing:
            raise AssertionError(f"report bundle missing keys: {sorted(missing)}")
        if data["kind"] != "workload_report_bundle":
            raise AssertionError(f"unexpected report kind: {data}")
        if data["report_status"] != "pass":
            raise AssertionError(f"report should pass with full vecsum evidence: {data}")
        if data["workload"] != "vecsum":
            raise AssertionError(f"unexpected workload: {data}")
        if data["selected_hardware_candidate_identity"] != "shared_reduction_adg":
            raise AssertionError(f"unexpected hardware identity: {data}")
        if data["selected_mapping_artifact_identity"] != "pnr-mapping":
            raise AssertionError(f"unexpected mapping artifact identity: {data}")
        optional_identities = data.get("optional_artifact_identities", {})
        if not isinstance(optional_identities, dict):
            raise AssertionError(f"report should include optional artifact identities: {data}")
        if optional_identities.get("simulation_comparison_report") != "sim-comparison-report":
            raise AssertionError(f"report should reference simulation comparison evidence: {data}")
        if optional_identities.get("runtime_package") != "runtime-package":
            raise AssertionError(f"report should reference runtime package evidence: {data}")
        expected_runtime_fallback = {
            "policy": "report_only",
            "decision": "report_only",
            "fallback_taken": False,
            "target_profile_id": "simulator::cgra_sim::mapping_constraint_estimate",
            "reason": "report-only runtime package records launch metadata without executing accelerator work",
        }
        if data["runtime_fallback_decision"] != expected_runtime_fallback:
            raise AssertionError(f"report should preserve runtime fallback decision: {data}")

        metrics = data.get("metric_records", [])
        if not isinstance(metrics, list) or not metrics:
            raise AssertionError(f"report should include metric records: {data}")
        metrics_by_id = metric_by_id(metrics)
        expected_metrics = {
            "metric::vecsum::dfg_sim_cycles": ("optimistic_steps", 579, "cycles", "dfg_software"),
            "metric::vecsum::cgra_sim_cycles": ("hardware_cycles", 589, "cycles", "cgra_mapped"),
            "metric::shared_reduction_adg::frequency_mhz": ("frequency", 250.0, "MHz", "custom_calibrated"),
            "metric::shared_reduction_adg::area_um2": ("area", 7250.0, "um2", "custom_calibrated"),
            "metric::shared_reduction_adg::dynamic_power_mw": (
                "dynamic_power",
                6.0,
                "mW",
                "custom_calibrated",
            ),
            "metric::shared_reduction_adg::leakage_power_mw": (
                "leakage_power",
                0.825,
                "mW",
                "custom_calibrated",
            ),
            "metric::vecsum::energy_nj": ("energy", 16.08, "nJ", "custom_calibrated"),
        }
        for metric_id, (metric_class, value, unit, fidelity) in expected_metrics.items():
            metric = metrics_by_id.get(metric_id)
            if metric is None:
                raise AssertionError(f"missing metric {metric_id}: {metrics}")
            if metric.get("metric_class") != metric_class:
                raise AssertionError(f"unexpected metric class for {metric_id}: {metric}")
            if abs(float(metric.get("value")) - float(value)) > 0.001:
                raise AssertionError(f"unexpected metric value for {metric_id}: {metric}")
            if metric.get("unit") != unit or metric.get("fidelity_level") != fidelity:
                raise AssertionError(f"unexpected unit or fidelity for {metric_id}: {metric}")
            if not metric.get("evidence_source_artifact_id"):
                raise AssertionError(f"metric lacks evidence source: {metric}")

        energy = metrics_by_id["metric::vecsum::energy_nj"]
        inputs = set(energy.get("input_metric_ids", []))
        required_inputs = {
            "metric::vecsum::cgra_sim_cycles",
            "metric::shared_reduction_adg::frequency_mhz",
            "metric::shared_reduction_adg::dynamic_power_mw",
            "metric::shared_reduction_adg::leakage_power_mw",
        }
        if inputs != required_inputs:
            raise AssertionError(f"energy metric should preserve input metric ids: {energy}")

        audit = out_dir / "artifact-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit),
                str(report),
            ],
            "workload report bundle audit",
        )
        audit_data = json.loads(audit.read_text())
        if audit_data.get("verdict") != "pass":
            raise AssertionError(f"expected report bundle audit pass: {audit_data}")
        reviews = audit_data.get("artifact_reviews", [])
        matching_reviews = [
            review for review in reviews
            if review.get("schema") == "workload_report_bundle"
        ]
        if len(matching_reviews) != 1:
            raise AssertionError(f"expected one report bundle review: {audit_data}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
