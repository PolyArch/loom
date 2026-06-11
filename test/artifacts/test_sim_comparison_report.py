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


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def assert_generator_blocks_missing_final_state(
    repo: Path,
    out_dir: Path,
    dfg_data: dict[str, object],
    cgra_data: dict[str, object],
    *,
    label: str,
    expected_status_key: str,
) -> None:
    dfg_report = out_dir / f"{label}-dfg-sim-report.json"
    cgra_report = out_dir / f"{label}-cgra-sim-report.json"
    comparison_report = out_dir / f"{label}-sim-comparison-report.json"
    write_json(dfg_report, dfg_data)
    write_json(cgra_report, cgra_data)
    result = artifact_test_common.run_command(
        repo,
        [
            "bash",
            "test/simulator/run_sim_comparison_report.sh",
            "--dfg-report",
            str(dfg_report),
            "--cgra-report",
            str(cgra_report),
            "--output",
            str(comparison_report),
        ],
    )
    if result.returncode == 0:
        raise AssertionError(f"{label} comparison unexpectedly passed")
    data = json.loads(comparison_report.read_text())
    if data.get("status") != "blocked":
        raise AssertionError(f"{label} comparison should be blocked: {data}")
    if data.get("difference_classification") != "unsupported_scope":
        raise AssertionError(f"{label} comparison should classify unsupported scope: {data}")
    if data.get(expected_status_key) != "blocked":
        raise AssertionError(f"{label} comparison missed blocked {expected_status_key}: {data}")
    if "skipped" in {
        data.get("functional_comparison_status"),
        data.get("memory_comparison_status"),
    }:
        raise AssertionError(f"{label} comparison must not silently skip final-state checks: {data}")


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
        result = artifact_test_common.run_command(
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
        )
        if result.returncode == 0:
            raise AssertionError("blocked simulation comparison unexpectedly returned success")
        if not comparison.is_file():
            raise AssertionError("blocked simulation comparison did not write a report")

        data = json.loads(comparison.read_text())
        missing = REQUIRED_KEYS - set(data)
        if missing:
            raise AssertionError(f"simulation comparison report missing keys: {sorted(missing)}")
        if data["kind"] != "sim_comparison_report":
            raise AssertionError(f"unexpected comparison report kind: {data}")
        if data["status"] != "blocked":
            raise AssertionError(f"comparison should be blocked for unmapped vecsum reports: {data}")
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
            "functional_comparison_status": "pass",
            "memory_comparison_status": "pass",
            "performance_comparison_status": "blocked",
            "difference_classification": "unsupported_scope",
        }
        for key, value in expected_statuses.items():
            if data[key] != value:
                raise AssertionError(f"unexpected {key}: {data}")
        dfg_data = json.loads((out_dir / "vecsum-dfg-sim-report.json").read_text())
        cgra_data = json.loads((out_dir / "vecsum-cgra-sim-report.json").read_text())
        if not isinstance(dfg_data.get("final_memory_state"), dict):
            raise AssertionError(f"DFG-sim report must expose final_memory_state: {dfg_data}")
        if not isinstance(cgra_data.get("final_outputs"), list):
            raise AssertionError(f"CGRA-sim report must expose final_outputs: {cgra_data}")
        if not isinstance(cgra_data.get("final_memory_state"), dict):
            raise AssertionError(f"CGRA-sim report must expose final_memory_state: {cgra_data}")
        if cgra_data.get("functional_state_source") != "carried_from_dfg_sim_report":
            raise AssertionError(f"CGRA-sim must label carried functional state: {cgra_data}")
        if cgra_data["final_outputs"] != dfg_data.get("final_outputs"):
            raise AssertionError(f"CGRA-sim final outputs should match DFG-sim: {cgra_data}")
        if cgra_data["final_memory_state"] != dfg_data["final_memory_state"]:
            raise AssertionError(f"CGRA-sim final memory should match DFG-sim: {cgra_data}")

        missing_audit_state_dfg = out_dir / "missing-audit-state-dfg-sim-report.json"
        missing_audit_state_dfg_data = json.loads(json.dumps(dfg_data))
        del missing_audit_state_dfg_data["final_memory_state"]
        missing_audit_state_dfg.write_text(
            json.dumps(missing_audit_state_dfg_data, indent=2, sort_keys=True) + "\n"
        )
        missing_audit_state_summary = out_dir / "missing-audit-state-dfg-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_audit_state_summary),
                str(missing_audit_state_dfg),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DFG report without final_memory_state passed artifact audit")

        missing_provenance_cgra = out_dir / "missing-provenance-cgra-sim-report.json"
        missing_provenance_cgra_data = json.loads(json.dumps(cgra_data))
        del missing_provenance_cgra_data["functional_state_source"]
        missing_provenance_cgra.write_text(
            json.dumps(missing_provenance_cgra_data, indent=2, sort_keys=True) + "\n"
        )
        missing_provenance_summary = out_dir / "missing-provenance-cgra-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_provenance_summary),
                str(missing_provenance_cgra),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("CGRA report without functional_state_source passed artifact audit")
        missing_state_dfg = out_dir / "missing-state-dfg-sim-report.json"
        missing_state_dfg_data = json.loads(json.dumps(dfg_data))
        del missing_state_dfg_data["final_memory_state"]
        missing_state_dfg.write_text(
            json.dumps(missing_state_dfg_data, indent=2, sort_keys=True) + "\n"
        )
        missing_state_cgra = out_dir / "missing-state-cgra-sim-report.json"
        result = artifact_test_common.run_command(
            repo,
            [
                str(repo / "build/tools/loom-cgra-sim/loom-cgra-sim"),
                "--dfg-report",
                str(missing_state_dfg),
                "--mapping-artifact",
                str(out_dir / "pnr-mapping.json"),
                "--hardware-mlir",
                str(repo / "test/pnr/shared_reduction_adg.mlir"),
                "--output",
                str(missing_state_cgra),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("CGRA-sim accepted a DFG report without final_memory_state")
        definitions = data.get("performance_metric_definitions", {})
        expected_definitions = {
            "dfg": "optimistic_pipeline_latency_throughput_sum",
            "cgra": "mapping_constraint_estimate",
        }
        if definitions != expected_definitions:
            raise AssertionError(f"comparison should preserve metric definitions: {data}")
        if data.get("dfg_sim_cycles") != 579 or data.get("cgra_sim_cycles") != 579:
            raise AssertionError(f"comparison should preserve simulator cycle values: {data}")
        if data.get("performance_delta_cycles") != 0:
            raise AssertionError(f"comparison should preserve blocked hardware delta: {data}")
        if "explicit_fabric_route_paths" not in data.get("explanation_categories", []):
            raise AssertionError(f"comparison should explain unsupported route evidence: {data}")

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

        skipped_comparison = out_dir / "skipped-sim-comparison-report.json"
        skipped_data = json.loads(json.dumps(data))
        skipped_data["status"] = "pass"
        skipped_data["functional_comparison_status"] = "skipped"
        skipped_data["memory_comparison_status"] = "skipped"
        skipped_data["performance_comparison_status"] = "pass"
        skipped_data["difference_classification"] = "expected_hardware_constraint"
        skipped_data["diagnostics"] = [
            "functional output comparison skipped because one report lacks final_outputs",
            "visible memory-state comparison skipped because reports expose no final memory state",
        ]
        skipped_comparison.write_text(json.dumps(skipped_data, indent=2, sort_keys=True) + "\n")
        skipped_audit = out_dir / "skipped-comparison-artifact-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(skipped_audit),
                str(skipped_comparison),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("pass comparison with skipped functional or memory checks passed audit")

        missing_outputs_dfg = json.loads(json.dumps(dfg_data))
        missing_outputs_cgra = json.loads(json.dumps(cgra_data))
        missing_outputs_dfg.pop("final_outputs", None)
        assert_generator_blocks_missing_final_state(
            repo,
            out_dir,
            missing_outputs_dfg,
            missing_outputs_cgra,
            label="missing-final-outputs",
            expected_status_key="functional_comparison_status",
        )

        missing_memory_dfg = json.loads(json.dumps(dfg_data))
        missing_memory_cgra = json.loads(json.dumps(cgra_data))
        missing_memory_dfg.pop("final_memory_state", None)
        missing_memory_cgra.pop("final_memory_state", None)
        assert_generator_blocks_missing_final_state(
            repo,
            out_dir,
            missing_memory_dfg,
            missing_memory_cgra,
            label="missing-final-memory",
            expected_status_key="memory_comparison_status",
        )

        mismatched_runtime_input = out_dir / "mismatch-runtime-input-sim-comparison-report.json"
        mismatched_runtime_input_data = json.loads(json.dumps(data))
        mismatched_runtime_input_data["runtime_input_identity"] = "test-app-fixture::other::default"
        mismatched_runtime_input.write_text(
            json.dumps(mismatched_runtime_input_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_runtime_input_audit = out_dir / "mismatch-runtime-input-artifact-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_runtime_input_audit),
                str(mismatched_runtime_input),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("mismatched runtime input comparison unexpectedly passed audit")

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
