#!/usr/bin/env python3
"""Regression test for DFG/CGRA simulation comparison reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "comparison_id",
    "workload",
    "runtime_input_identity",
    "dfg_sim_report_identity",
    "cgra_sim_report_identity",
    "mapping_artifact_identity",
    "functional_comparison_status",
    "memory_comparison_status",
    "performance_comparison_status",
    "performance_metric_definitions",
    "difference_classification",
    "explanation_categories",
    "diagnostics",
    "status",
}


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-sim-comparison-") as tmp:
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

        comparison = out_dir / "sim-comparison-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/simulator/run_sim_comparison_report.sh",
                "--dfg-report",
                str(out_dir / "vecsum-dfg-sim-report.json"),
                "--cgra-report",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--mapping-artifact",
                str(out_dir / "pnr-mapping.json"),
                "--output",
                str(comparison),
            ],
            "simulation comparison report",
        )

        data = json.loads(comparison.read_text())
        missing = REQUIRED_KEYS - set(data)
        if missing:
            raise AssertionError(f"simulation comparison report missing keys: {sorted(missing)}")
        if data["kind"] != "sim_comparison_report":
            raise AssertionError(f"unexpected comparison report kind: {data}")
        if data["status"] != "pass":
            raise AssertionError(f"comparison should pass for matched vecsum reports: {data}")
        if data["workload"] != "vecsum":
            raise AssertionError(f"unexpected comparison workload: {data}")
        if data["runtime_input_identity"] != "test-app-fixture::vecsum::default":
            raise AssertionError(f"unexpected runtime input identity: {data}")
        if data["dfg_sim_report_identity"] != "vecsum-dfg-sim-report":
            raise AssertionError(f"unexpected DFG report identity: {data}")
        if data["cgra_sim_report_identity"] != "vecsum-cgra-sim-report":
            raise AssertionError(f"unexpected CGRA report identity: {data}")
        if data["mapping_artifact_identity"] != "pnr-mapping":
            raise AssertionError(f"unexpected mapping artifact identity: {data}")
        expected_statuses = {
            "functional_comparison_status": "skipped",
            "memory_comparison_status": "skipped",
            "performance_comparison_status": "pass",
            "difference_classification": "expected_hardware_constraint",
        }
        for key, value in expected_statuses.items():
            if data[key] != value:
                raise AssertionError(f"unexpected {key}: {data}")
        definitions = data.get("performance_metric_definitions", {})
        expected_definitions = {
            "dfg": "optimistic_pipeline_latency_throughput_sum",
            "cgra": "mapping_constraint_estimate",
        }
        if definitions != expected_definitions:
            raise AssertionError(f"comparison should preserve metric definitions: {data}")
        if data.get("dfg_sim_cycles") != 579 or data.get("cgra_sim_cycles") != 589:
            raise AssertionError(f"comparison should preserve simulator cycle values: {data}")
        if data.get("performance_delta_cycles") != 10:
            raise AssertionError(f"comparison should preserve hardware delta: {data}")
        if "route_latency" not in data.get("explanation_categories", []):
            raise AssertionError(f"comparison should explain hardware overhead categories: {data}")

        audit = out_dir / "comparison-artifact-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit),
                str(comparison),
            ],
            "simulation comparison report audit",
        )
        audit_data = json.loads(audit.read_text())
        if audit_data.get("verdict") != "pass":
            raise AssertionError(f"expected comparison report audit pass: {audit_data}")

        mismatched_dfg = out_dir / "mismatch-dfg-sim-report.json"
        dfg_data = json.loads((out_dir / "vecsum-dfg-sim-report.json").read_text())
        dfg_data["workload"] = "other_workload"
        mismatched_dfg.write_text(json.dumps(dfg_data, indent=2, sort_keys=True) + "\n")
        mismatch_report = out_dir / "mismatch-sim-comparison-report.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/simulator/run_sim_comparison_report.sh",
                "--dfg-report",
                str(mismatched_dfg),
                "--cgra-report",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--mapping-artifact",
                str(out_dir / "pnr-mapping.json"),
                "--output",
                str(mismatch_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("mismatched reports unexpectedly produced a passing comparison")
        mismatch_data = json.loads(mismatch_report.read_text())
        if mismatch_data.get("status") != "fail":
            raise AssertionError(f"mismatched report should fail: {mismatch_data}")
        if mismatch_data.get("difference_classification") != "report_mismatch":
            raise AssertionError(f"mismatched report should classify report mismatch: {mismatch_data}")
        if mismatch_data.get("performance_comparison_status") != "blocked":
            raise AssertionError(f"mismatched report must not produce performance pass: {mismatch_data}")
        diagnostics = mismatch_data.get("diagnostics", [])
        if not any("workload identity mismatch" in str(item) for item in diagnostics):
            raise AssertionError(f"mismatched report should diagnose workload identity: {mismatch_data}")

        wrong_kind_dfg = out_dir / "wrong-kind-dfg-report.json"
        wrong_kind_dfg_data = json.loads((out_dir / "vecsum-dfg-sim-report.json").read_text())
        wrong_kind_dfg_data["kind"] = "cgra_sim_report"
        wrong_kind_dfg.write_text(json.dumps(wrong_kind_dfg_data, indent=2, sort_keys=True) + "\n")
        wrong_kind_dfg_report = out_dir / "wrong-kind-dfg-sim-comparison-report.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/simulator/run_sim_comparison_report.sh",
                "--dfg-report",
                str(wrong_kind_dfg),
                "--cgra-report",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--mapping-artifact",
                str(out_dir / "pnr-mapping.json"),
                "--output",
                str(wrong_kind_dfg_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("wrong-kind DFG report unexpectedly produced a passing comparison")
        wrong_kind_dfg_data = json.loads(wrong_kind_dfg_report.read_text())
        if wrong_kind_dfg_data.get("status") != "fail":
            raise AssertionError(f"wrong-kind DFG report should fail: {wrong_kind_dfg_data}")
        if wrong_kind_dfg_data.get("performance_comparison_status") != "blocked":
            raise AssertionError(f"wrong-kind DFG report must block performance comparison: {wrong_kind_dfg_data}")
        wrong_kind_dfg_audit = out_dir / "wrong-kind-dfg-artifact-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(wrong_kind_dfg_audit),
                str(wrong_kind_dfg_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("wrong-kind DFG comparison unexpectedly passed audit")

        wrong_kind_cgra = out_dir / "wrong-kind-cgra-report.json"
        wrong_kind_cgra_data = json.loads((out_dir / "vecsum-cgra-sim-report.json").read_text())
        wrong_kind_cgra_data["kind"] = "dfg_sim_report"
        wrong_kind_cgra.write_text(json.dumps(wrong_kind_cgra_data, indent=2, sort_keys=True) + "\n")
        wrong_kind_cgra_report = out_dir / "wrong-kind-cgra-sim-comparison-report.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/simulator/run_sim_comparison_report.sh",
                "--dfg-report",
                str(out_dir / "vecsum-dfg-sim-report.json"),
                "--cgra-report",
                str(wrong_kind_cgra),
                "--mapping-artifact",
                str(out_dir / "pnr-mapping.json"),
                "--output",
                str(wrong_kind_cgra_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("wrong-kind CGRA report unexpectedly produced a passing comparison")
        wrong_kind_cgra_data = json.loads(wrong_kind_cgra_report.read_text())
        if wrong_kind_cgra_data.get("status") != "fail":
            raise AssertionError(f"wrong-kind CGRA report should fail: {wrong_kind_cgra_data}")
        if wrong_kind_cgra_data.get("performance_comparison_status") != "blocked":
            raise AssertionError(f"wrong-kind CGRA report must block performance comparison: {wrong_kind_cgra_data}")
        wrong_kind_cgra_audit = out_dir / "wrong-kind-cgra-artifact-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(wrong_kind_cgra_audit),
                str(wrong_kind_cgra_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("wrong-kind CGRA comparison unexpectedly passed audit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
