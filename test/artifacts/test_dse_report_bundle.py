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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
