#!/usr/bin/env python3
"""Regression test for DSE report bundles."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import artifact_test_common


REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "dse_run_id",
    "objective_records",
    "candidate_list",
    "selected_candidates",
    "pareto_set",
    "rejected_candidate_summaries",
    "referenced_dse_candidate_artifact_identities",
    "referenced_workload_report_bundle_identities",
    "referenced_hardware_candidate_report_bundle_identities",
    "input_artifact_fingerprints",
    "runtime_evidence_summaries",
    "selected_policy_id",
    "policy_configuration",
    "candidate_ordering_rule",
    "report_status",
    "diagnostic_records",
    "diagnostics",
}


def selected_candidate_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("selection_status") == "selected":
                return row
    raise AssertionError(f"{path.name} has no selected candidate row")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-dse-report-") as tmp:
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

        report = out_dir / "dse-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(report),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
            "DSE report bundle",
        )

        data = json.loads(report.read_text())
        missing = REQUIRED_KEYS - set(data)
        if missing:
            raise AssertionError(f"DSE report bundle missing keys: {sorted(missing)}")
        if data["kind"] != "dse_report_bundle":
            raise AssertionError(f"unexpected DSE report bundle kind: {data}")
        if data["report_status"] != "pass":
            raise AssertionError(f"DSE report should pass with selected candidate evidence: {data}")
        if data["dse_run_id"] != "dse::deterministic_minimize_runtime_v1":
            raise AssertionError(f"unexpected DSE run id: {data}")
        if data["selected_policy_id"] != "deterministic_minimize_runtime_v1":
            raise AssertionError(f"unexpected DSE policy id: {data}")
        if data["candidate_ordering_rule"] != "runtime_score_then_candidate_id":
            raise AssertionError(f"unexpected DSE ordering rule: {data}")
        expected_policy_configuration = {
            "policy_kind": "deterministic",
            "random_seed": None,
            "conflict_resolution": "candidate_ordering_rule",
        }
        if data["policy_configuration"] != expected_policy_configuration:
            raise AssertionError(f"unexpected DSE policy configuration: {data}")
        if data["referenced_dse_candidate_artifact_identities"] != ["dse-candidate-summary"]:
            raise AssertionError(f"unexpected DSE candidate artifact references: {data}")
        if data["referenced_workload_report_bundle_identities"] != ["workload-report-bundle"]:
            raise AssertionError(f"unexpected workload report references: {data}")
        if data["referenced_hardware_candidate_report_bundle_identities"] != ["hardware-report-bundle"]:
            raise AssertionError(f"unexpected hardware report references: {data}")
        expected_report_fingerprints = {
            "dse-candidate-summary": artifact_test_common.fingerprint(out_dir / "dse-candidate-summary.csv"),
            "workload-report-bundle": artifact_test_common.fingerprint(out_dir / "workload-report-bundle.json"),
            "hardware-report-bundle": artifact_test_common.fingerprint(out_dir / "hardware-report-bundle.json"),
        }
        if data["input_artifact_fingerprints"] != expected_report_fingerprints:
            raise AssertionError(f"unexpected DSE report input fingerprints: {data}")
        workload_runtime_evidence = json.loads((out_dir / "workload-report-bundle.json").read_text())[
            "runtime_evidence"
        ]
        expected_runtime_summaries = [
            {
                "workload_report_bundle_identity": "workload-report-bundle",
                "runtime_package_identity": "runtime-package",
                "runtime_report_identity": "runtime-report::vecsum::vecsum__shared_reduction_adg::report_only",
                "host_program_identity": workload_runtime_evidence["host_program_identity"],
                "work_package_identity": workload_runtime_evidence["work_package_identity"],
                "launch_descriptor_identity": workload_runtime_evidence["launch_descriptor_identity"],
                "mapping_artifact_identity": workload_runtime_evidence["mapping_artifact_identity"],
                "fabric_adg_identity": workload_runtime_evidence["fabric_adg_identity"],
                "target_profile_id": workload_runtime_evidence["target_profile_id"],
                "launch_status": "not_run",
                "target_status": "not_run",
                "runtime_trace_identity": workload_runtime_evidence["runtime_trace_identity"],
                "profiling_record_identity": workload_runtime_evidence["profiling_record_identity"],
                "data_movement_policy": "simulated",
                "synchronization_mode": "host_wait",
                "required_data_movement_policies": ["simulated"],
                "required_synchronization_policies": ["host_wait"],
                "simulator_report_identities": workload_runtime_evidence["simulator_report_identities"],
                "output_buffer_identities": workload_runtime_evidence["output_buffer_identities"],
                "diagnostic_records": workload_runtime_evidence["diagnostic_records"],
                "input_artifact_fingerprints": workload_runtime_evidence["input_artifact_fingerprints"],
                "fallback_decision": {
                    "policy": "report_only",
                    "decision": "report_only",
                    "fallback_taken": False,
                    "target_profile_id": "simulator::cgra_sim::mapping_constraint_estimate",
                    "reason": "report-only runtime package records launch metadata without executing accelerator work",
                },
            }
        ]
        if data["runtime_evidence_summaries"] != expected_runtime_summaries:
            raise AssertionError(f"unexpected runtime evidence summaries: {data}")

        objectives = data.get("objective_records", [])
        if len(objectives) != 1:
            raise AssertionError(f"expected one objective record: {data}")
        objective = objectives[0]
        expected_objective = {
            "objective_id": "objective::minimize_runtime",
            "objective_kind": "minimize_runtime",
            "comparison_direction": "minimize",
            "units": "cycles",
        }
        for key, value in expected_objective.items():
            if objective.get(key) != value:
                raise AssertionError(f"unexpected objective {key}: {objective}")
        if "metric::vecsum::cgra_sim_cycles" not in objective.get("metric_inputs", []):
            raise AssertionError(f"objective should cite CGRA cycle metric input: {objective}")

        candidates = data.get("candidate_list", [])
        if len(candidates) != 1:
            raise AssertionError(f"expected one DSE candidate record: {data}")
        candidate = candidates[0]
        candidate_id = "candidate::vecsum::shared_reduction_adg::vecsum__shared_reduction_adg"
        if candidate.get("candidate_id") != candidate_id:
            raise AssertionError(f"unexpected candidate id: {candidate}")
        if candidate.get("candidate_kind") != "combined_full_stack_candidate":
            raise AssertionError(f"unexpected candidate kind: {candidate}")
        if candidate.get("status") != "selected":
            raise AssertionError(f"candidate should be selected: {candidate}")
        expected_input_fingerprints = artifact_test_common.semicolon_map(
            selected_candidate_row(out_dir / "dse-candidate-summary.csv")["input_artifact_fingerprints"]
        )
        if candidate.get("input_artifact_fingerprints") != expected_input_fingerprints:
            raise AssertionError(f"candidate missed input artifact fingerprints: {candidate}")
        if sorted(candidate.get("referenced_input_artifacts", [])) != sorted(expected_input_fingerprints):
            raise AssertionError(f"candidate fingerprints do not cover referenced inputs: {candidate}")
        for metric_id in (
            "metric::vecsum::cgra_sim_cycles",
            "metric::shared_reduction_adg::frequency_mhz",
            "metric::shared_reduction_adg::area_um2",
            "metric::shared_reduction_adg::dynamic_power_mw",
            "metric::vecsum::energy_nj",
        ):
            if metric_id not in candidate.get("metric_records_used", []):
                raise AssertionError(f"candidate missed metric {metric_id}: {candidate}")
        if data["selected_candidates"] != [candidate_id]:
            raise AssertionError(f"unexpected selected candidates: {data}")
        if data["pareto_set"] != [] or data["rejected_candidate_summaries"] != []:
            raise AssertionError(f"unexpected non-selected candidate summaries: {data}")
        if data["diagnostic_records"] != []:
            raise AssertionError(f"passing DSE report should have no diagnostic records: {data}")

        audit = out_dir / "dse-artifact-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit),
                str(report),
            ],
            "DSE report bundle audit",
        )
        audit_data = json.loads(audit.read_text())
        if audit_data.get("verdict") != "pass":
            raise AssertionError(f"expected DSE report audit pass: {audit_data}")
        reviews = audit_data.get("artifact_reviews", [])
        matching_reviews = [
            review for review in reviews
            if review.get("schema") == "dse_report_bundle"
        ]
        if len(matching_reviews) != 1:
            raise AssertionError(f"expected one DSE report bundle review: {audit_data}")

        bad_candidate_fingerprint = out_dir / "bad-candidate-fingerprint-dse-report-bundle.json"
        bad_candidate_fingerprint_data = json.loads(report.read_text())
        candidate_fingerprints = bad_candidate_fingerprint_data["candidate_list"][0]["input_artifact_fingerprints"]
        first_input = next(iter(candidate_fingerprints))
        candidate_fingerprints[first_input] = "bad"
        bad_candidate_fingerprint.write_text(
            json.dumps(bad_candidate_fingerprint_data, indent=2, sort_keys=True) + "\n"
        )
        bad_candidate_fingerprint_audit = out_dir / "bad-candidate-fingerprint-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_candidate_fingerprint_audit),
                str(bad_candidate_fingerprint),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed candidate input fingerprint unexpectedly passed audit")

        bad_runtime_summary = out_dir / "bad-runtime-summary-dse-report-bundle.json"
        bad_runtime_summary_data = json.loads(report.read_text())
        bad_runtime_summary_data["runtime_evidence_summaries"][0]["input_artifact_fingerprints"][
            "runtime-package"
        ] = "bad"
        bad_runtime_summary.write_text(json.dumps(bad_runtime_summary_data, indent=2, sort_keys=True) + "\n")
        bad_runtime_summary_audit = out_dir / "bad-runtime-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_summary_audit),
                str(bad_runtime_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed runtime summary fingerprint unexpectedly passed audit")

        stale_runtime_summary = out_dir / "stale-runtime-summary-dse-report-bundle.json"
        stale_runtime_summary_data = json.loads(report.read_text())
        stale_runtime_summary_data["runtime_evidence_summaries"][0]["input_artifact_fingerprints"][
            "pnr-mapping"
        ] = "0" * 64
        stale_runtime_summary.write_text(json.dumps(stale_runtime_summary_data, indent=2, sort_keys=True) + "\n")
        stale_runtime_summary_audit = out_dir / "stale-runtime-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_runtime_summary_audit),
                str(stale_runtime_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with stale runtime summary fingerprint unexpectedly passed audit")

        bad_runtime_policy_summary = out_dir / "bad-runtime-policy-summary-dse-report-bundle.json"
        bad_runtime_policy_summary_data = json.loads(report.read_text())
        bad_runtime_policy_summary_data["runtime_evidence_summaries"][0][
            "required_data_movement_policies"
        ] = ["shared_coherent"]
        bad_runtime_policy_summary_data["runtime_evidence_summaries"][0][
            "required_synchronization_policies"
        ] = ["device_poll"]
        bad_runtime_policy_summary.write_text(
            json.dumps(bad_runtime_policy_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_policy_summary_audit = out_dir / "bad-runtime-policy-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_policy_summary_audit),
                str(bad_runtime_policy_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched runtime summary policies unexpectedly passed audit")

        bad_runtime_identity_summary = out_dir / "bad-runtime-identity-summary-dse-report-bundle.json"
        bad_runtime_identity_summary_data = json.loads(report.read_text())
        bad_runtime_identity_summary_data["runtime_evidence_summaries"][0]["launch_descriptor_identity"] = []
        bad_runtime_identity_summary.write_text(
            json.dumps(bad_runtime_identity_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_identity_summary_audit = out_dir / "bad-runtime-identity-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_identity_summary_audit),
                str(bad_runtime_identity_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed runtime identity summary unexpectedly passed audit")

        bad_runtime_diagnostics_summary = out_dir / "bad-runtime-diagnostics-summary-dse-report-bundle.json"
        bad_runtime_diagnostics_summary_data = json.loads(report.read_text())
        bad_runtime_diagnostics_summary_data["runtime_evidence_summaries"][0]["diagnostic_records"] = (
            "runtime-package::1"
        )
        bad_runtime_diagnostics_summary.write_text(
            json.dumps(bad_runtime_diagnostics_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_diagnostics_summary_audit = out_dir / "bad-runtime-diagnostics-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_diagnostics_summary_audit),
                str(bad_runtime_diagnostics_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed runtime diagnostics summary unexpectedly passed audit")

        bad_runtime_output_summary = out_dir / "bad-runtime-output-summary-dse-report-bundle.json"
        bad_runtime_output_summary_data = json.loads(report.read_text())
        bad_runtime_output_summary_data["runtime_evidence_summaries"][0]["output_buffer_identities"] = "output"
        bad_runtime_output_summary.write_text(
            json.dumps(bad_runtime_output_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_output_summary_audit = out_dir / "bad-runtime-output-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_output_summary_audit),
                str(bad_runtime_output_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed runtime output summary unexpectedly passed audit")

        bad_runtime_simulator_summary = out_dir / "bad-runtime-simulator-summary-dse-report-bundle.json"
        bad_runtime_simulator_summary_data = json.loads(report.read_text())
        bad_runtime_simulator_summary_data["runtime_evidence_summaries"][0][
            "simulator_report_identities"
        ] = "vecsum-cgra-sim-report"
        bad_runtime_simulator_summary.write_text(
            json.dumps(bad_runtime_simulator_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_simulator_summary_audit = out_dir / "bad-runtime-simulator-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_simulator_summary_audit),
                str(bad_runtime_simulator_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed runtime simulator summary unexpectedly passed audit")

        extra_custom_policy_summary = out_dir / "extra-custom-policy-summary-dse-report-bundle.json"
        extra_custom_policy_summary_data = json.loads(report.read_text())
        extra_custom_policy_summary_data["runtime_evidence_summaries"][0][
            "custom_data_movement_policy_identity"
        ] = "runtime-policy::unexpected::vecsum"
        extra_custom_policy_summary.write_text(
            json.dumps(extra_custom_policy_summary_data, indent=2, sort_keys=True) + "\n"
        )
        extra_custom_policy_summary_audit = out_dir / "extra-custom-policy-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(extra_custom_policy_summary_audit),
                str(extra_custom_policy_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with custom policy for simulated runtime unexpectedly passed audit")

        unnamed_custom_policy_summary = out_dir / "unnamed-custom-policy-summary-dse-report-bundle.json"
        unnamed_custom_policy_summary_data = json.loads(report.read_text())
        unnamed_custom_policy_summary_data["runtime_evidence_summaries"][0]["data_movement_policy"] = "custom"
        unnamed_custom_policy_summary_data["runtime_evidence_summaries"][0][
            "required_data_movement_policies"
        ] = ["custom"]
        unnamed_custom_policy_summary.write_text(
            json.dumps(unnamed_custom_policy_summary_data, indent=2, sort_keys=True) + "\n"
        )
        unnamed_custom_policy_summary_audit = out_dir / "unnamed-custom-policy-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(unnamed_custom_policy_summary_audit),
                str(unnamed_custom_policy_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with unnamed custom runtime policy unexpectedly passed audit")

        stale_candidate_input = out_dir / "stale-candidate-input-dse-report-bundle.json"
        stale_candidate_input_data = json.loads(report.read_text())
        stale_candidate_input_data["input_artifact_fingerprints"]["dse-candidate-summary"] = "0" * 64
        stale_candidate_input.write_text(json.dumps(stale_candidate_input_data, indent=2, sort_keys=True) + "\n")
        stale_candidate_input_audit = out_dir / "stale-candidate-input-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_candidate_input_audit),
                str(stale_candidate_input),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with stale candidate input fingerprint unexpectedly passed audit")

        bad_report_fingerprint = out_dir / "bad-report-fingerprint-dse-report-bundle.json"
        bad_report_fingerprint_data = json.loads(report.read_text())
        bad_report_fingerprint_data["input_artifact_fingerprints"]["workload-report-bundle"] = "0" * 64
        bad_report_fingerprint.write_text(
            json.dumps(bad_report_fingerprint_data, indent=2, sort_keys=True) + "\n"
        )
        bad_report_fingerprint_audit = out_dir / "bad-report-fingerprint-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_report_fingerprint_audit),
                str(bad_report_fingerprint),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with stale report input fingerprint unexpectedly passed audit")

        custom_workload_report = out_dir / "custom-workload-evidence.json"
        custom_workload_report.write_text((out_dir / "workload-report-bundle.json").read_text())
        custom_name_report = out_dir / "custom-name-dse-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(custom_name_report),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
                "--artifact",
                str(custom_workload_report),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
            "DSE report bundle with embedded workload bundle kind",
        )
        custom_name_data = json.loads(custom_name_report.read_text())
        if custom_name_data["report_status"] != "pass":
            raise AssertionError(f"custom-named workload report should be accepted: {custom_name_data}")
        if custom_name_data["referenced_workload_report_bundle_identities"] != ["custom-workload-evidence"]:
            raise AssertionError(f"custom workload report path identity was not preserved: {custom_name_data}")
        if custom_name_data["input_artifact_fingerprints"].get(
            "custom-workload-evidence"
        ) != artifact_test_common.fingerprint(custom_workload_report):
            raise AssertionError(f"custom workload report fingerprint was not preserved: {custom_name_data}")

        missing_candidate_report = out_dir / "missing-candidate-dse-report-bundle.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(missing_candidate_report),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report without candidate summary unexpectedly passed")
        missing_candidate_data = json.loads(missing_candidate_report.read_text())
        records = missing_candidate_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "dse_candidate_missing"
            and record.get("component") == "dse_report_bundle"
            for record in records
        ):
            raise AssertionError(f"missing candidate report needs structured diagnostics: {missing_candidate_data}")
        missing_candidate_audit = out_dir / "missing-candidate-dse-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_candidate_audit),
                str(missing_candidate_report),
            ],
            "blocked DSE report bundle audit",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
