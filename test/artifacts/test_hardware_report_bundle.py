#!/usr/bin/env python3
"""Regression test for hardware candidate report bundles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "bundle_id",
    "hardware_candidate_identity",
    "fabric_adg_identity",
    "adg_builder_recipe_identity",
    "rtl_manifest_identity",
    "eda_report_identities",
    "fpa_report_identities",
    "supported_workload_classes",
    "report_status",
    "diagnostics",
    "metric_records",
}


def metric_by_id(metrics: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(metric.get("metric_id")): metric for metric in metrics}


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-hardware-report-") as tmp:
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

        report = out_dir / "hardware-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_hardware_report_bundle.sh",
                "--output",
                str(report),
                "--artifact",
                str(out_dir / "adg-hardware-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "hardware candidate report bundle",
        )

        data = json.loads(report.read_text())
        missing = REQUIRED_KEYS - set(data)
        if missing:
            raise AssertionError(f"hardware report bundle missing keys: {sorted(missing)}")
        if data["kind"] != "hardware_report_bundle":
            raise AssertionError(f"unexpected hardware report kind: {data}")
        if data["report_status"] != "pass":
            raise AssertionError(f"hardware report should pass with ADG and FPA evidence: {data}")
        expected_hardware = "test/pnr/shared_reduction_adg.mlir::shared_reduction_adg"
        if data["hardware_candidate_identity"] != expected_hardware:
            raise AssertionError(f"unexpected hardware identity: {data}")
        if data["fabric_adg_identity"] != "test/pnr/shared_reduction_adg.mlir":
            raise AssertionError(f"unexpected Fabric ADG identity: {data}")
        if data["fpa_report_identities"] != ["rtl-fpa-summary"]:
            raise AssertionError(f"unexpected FPA report identities: {data}")
        if data["supported_workload_classes"] != ["vecsum"]:
            raise AssertionError(f"unexpected supported workload classes: {data}")

        metrics = data.get("metric_records", [])
        if not isinstance(metrics, list) or not metrics:
            raise AssertionError(f"hardware report should include metric records: {data}")
        metrics_by_id = metric_by_id(metrics)
        expected_metrics = {
            "metric::test/pnr/shared_reduction_adg.mlir::shared_reduction_adg::node_count": (
                "hardware_nodes",
                25,
                "count",
                "fabric_verified",
            ),
            "metric::test/pnr/shared_reduction_adg.mlir::shared_reduction_adg::link_count": (
                "hardware_links",
                0,
                "count",
                "fabric_verified",
            ),
            "metric::test/pnr/shared_reduction_adg.mlir::shared_reduction_adg::frequency_mhz": (
                "frequency",
                250.0,
                "MHz",
                "custom_calibrated",
            ),
            "metric::test/pnr/shared_reduction_adg.mlir::shared_reduction_adg::area_um2": (
                "area",
                7250.0,
                "um2",
                "custom_calibrated",
            ),
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

        audit = out_dir / "hardware-artifact-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit),
                str(report),
            ],
            "hardware report bundle audit",
        )
        audit_data = json.loads(audit.read_text())
        if audit_data.get("verdict") != "pass":
            raise AssertionError(f"expected hardware report audit pass: {audit_data}")
        reviews = audit_data.get("artifact_reviews", [])
        matching_reviews = [
            review for review in reviews
            if review.get("schema") == "hardware_report_bundle"
        ]
        if len(matching_reviews) != 1:
            raise AssertionError(f"expected one hardware report bundle review: {audit_data}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
