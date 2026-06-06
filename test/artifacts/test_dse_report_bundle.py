#!/usr/bin/env python3
"""Regression test for DSE report bundles."""

from __future__ import annotations

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
    "referenced_workload_report_bundle_identities",
    "referenced_hardware_candidate_report_bundle_identities",
    "runtime_evidence_summaries",
    "selected_policy_id",
    "policy_configuration",
    "candidate_ordering_rule",
    "report_status",
    "diagnostics",
}


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
        if data["referenced_workload_report_bundle_identities"] != ["workload-report-bundle"]:
            raise AssertionError(f"unexpected workload report references: {data}")
        if data["referenced_hardware_candidate_report_bundle_identities"] != ["hardware-report-bundle"]:
            raise AssertionError(f"unexpected hardware report references: {data}")
        workload_runtime_evidence = json.loads((out_dir / "workload-report-bundle.json").read_text())[
            "runtime_evidence"
        ]
        expected_runtime_summaries = [
            {
                "workload_report_bundle_identity": "workload-report-bundle",
                "runtime_package_identity": "runtime-package",
                "runtime_report_identity": "runtime-report::vecsum::vecsum__shared_reduction_adg::report_only",
                "launch_status": "not_run",
                "target_status": "not_run",
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
