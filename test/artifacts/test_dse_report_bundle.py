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


def artifact_identity(reference: str) -> str:
    path = Path(reference)
    for suffix in (".csv", ".json"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def identity_list(raw: str) -> list[str]:
    return [artifact_identity(reference) for reference in raw.split(";") if reference]


def identity_map(raw: str) -> dict[str, str]:
    return {
        artifact_identity(reference): fingerprint
        for reference, fingerprint in artifact_test_common.semicolon_map(raw).items()
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
                "host_wrapper_identity": workload_runtime_evidence["host_wrapper_identity"],
                "host_interface": workload_runtime_evidence["host_interface"],
                "runtime_handle_model": workload_runtime_evidence["runtime_handle_model"],
                "work_package_metadata": workload_runtime_evidence["work_package_metadata"],
                "work_package_identity": workload_runtime_evidence["work_package_identity"],
                "launch_descriptor_identity": workload_runtime_evidence["launch_descriptor_identity"],
                "launch_descriptor": workload_runtime_evidence["launch_descriptor"],
                "mapping_artifact_identity": workload_runtime_evidence["mapping_artifact_identity"],
                "fabric_adg_identity": workload_runtime_evidence["fabric_adg_identity"],
                "target_profile_id": workload_runtime_evidence["target_profile_id"],
                "target_profile": workload_runtime_evidence["target_profile"],
                "fallback_policy": workload_runtime_evidence["fallback_policy"],
                "launch_status": "not_run",
                "target_status": "not_run",
                "runtime_trace_identity": workload_runtime_evidence["runtime_trace_identity"],
                "profiling_record_identity": workload_runtime_evidence["profiling_record_identity"],
                "data_movement_policy": "simulated",
                "synchronization_mode": "host_wait",
                "memory_descriptors": workload_runtime_evidence["memory_descriptors"],
                "argument_descriptors": workload_runtime_evidence["argument_descriptors"],
                "runtime_configuration": workload_runtime_evidence["runtime_configuration"],
                "required_runtime_features": workload_runtime_evidence["required_runtime_features"],
                "required_data_movement_policies": ["simulated"],
                "required_synchronization_policies": ["host_wait"],
                "simulator_report_identities": workload_runtime_evidence["simulator_report_identities"],
                "output_buffer_identities": workload_runtime_evidence["output_buffer_identities"],
                "diagnostic_records": workload_runtime_evidence["diagnostic_records"],
                "report_output_configuration": workload_runtime_evidence["report_output_configuration"],
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
            "constraint_or_optimization_mode": "optimization",
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
        selected_row = selected_candidate_row(out_dir / "dse-candidate-summary.csv")
        expected_input_fingerprints = identity_map(selected_row["input_artifact_fingerprints"])
        if candidate.get("input_artifact_fingerprints") != expected_input_fingerprints:
            raise AssertionError(f"candidate missed input artifact fingerprints: {candidate}")
        if sorted(candidate.get("referenced_input_artifacts", [])) != sorted(expected_input_fingerprints):
            raise AssertionError(f"candidate fingerprints do not cover referenced inputs: {candidate}")
        expected_outputs = identity_list(selected_row.get("output_artifacts", ""))
        if sorted(candidate.get("generated_output_artifacts", [])) != sorted(expected_outputs):
            raise AssertionError(f"candidate missed generated output artifacts: {candidate}")
        if str(out_dir) in json.dumps(candidate, sort_keys=True):
            raise AssertionError(f"candidate record should not expose private paths: {candidate}")
        for metric_id in (
            "metric::vecsum::cgra_sim_cycles",
            "metric::shared_reduction_adg::frequency_mhz",
            "metric::shared_reduction_adg::area_um2",
            "metric::shared_reduction_adg::dynamic_power_mw",
            "metric::shared_reduction_adg::leakage_power_mw",
            "metric::vecsum::energy_nj",
            "metric::vecsum::throughput_items_per_s",
            "metric::vecsum::performance_per_watt",
            "metric::vecsum::performance_per_area",
        ):
            if metric_id not in candidate.get("metric_records_used", []):
                raise AssertionError(f"candidate missed metric {metric_id}: {candidate}")
        if candidate.get("objective_records_used") != ["objective::minimize_runtime"]:
            raise AssertionError(f"candidate missed objective provenance: {candidate}")
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

        throughput_candidate_summary = out_dir / "throughput-dse-candidate-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/dse/run_candidate_summary.sh",
                "--objective",
                "maximize_throughput",
                "--output",
                str(throughput_candidate_summary),
                "--artifact",
                str(out_dir / "pnr-mapping-summary.csv"),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "throughput DSE candidate summary",
        )
        throughput_report = out_dir / "throughput-dse-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(throughput_report),
                "--artifact",
                str(throughput_candidate_summary),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
            "throughput DSE report bundle",
        )
        throughput_data = json.loads(throughput_report.read_text())
        throughput_objective = throughput_data.get("objective_records", [])[0]
        expected_throughput_objective = {
            "objective_id": "objective::maximize_throughput",
            "objective_kind": "maximize_throughput",
            "constraint_or_optimization_mode": "optimization",
            "comparison_direction": "maximize",
            "units": "items_per_s",
        }
        for key, value in expected_throughput_objective.items():
            if throughput_objective.get(key) != value:
                raise AssertionError(f"unexpected throughput objective {key}: {throughput_objective}")
        if throughput_objective.get("metric_inputs") != ["metric::vecsum::throughput_items_per_s"]:
            raise AssertionError(f"throughput objective should cite throughput metric: {throughput_objective}")
        if throughput_data.get("candidate_ordering_rule") != "throughput_score_then_candidate_id":
            raise AssertionError(f"unexpected throughput ordering rule: {throughput_data}")
        if throughput_data.get("selected_policy_id") != "deterministic_maximize_throughput_v1":
            raise AssertionError(f"unexpected throughput policy id: {throughput_data}")
        throughput_candidate = throughput_data.get("candidate_list", [])[0]
        if "metric::vecsum::throughput_items_per_s" not in throughput_candidate.get("metric_records_used", []):
            raise AssertionError(f"throughput candidate missed throughput metric: {throughput_candidate}")
        throughput_audit = out_dir / "throughput-dse-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(throughput_audit),
                str(throughput_report),
            ],
            "throughput DSE report bundle audit",
        )

        perf_watt_candidate_summary = out_dir / "perf-watt-dse-candidate-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/dse/run_candidate_summary.sh",
                "--objective",
                "maximize_performance_per_watt",
                "--output",
                str(perf_watt_candidate_summary),
                "--artifact",
                str(out_dir / "pnr-mapping-summary.csv"),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "performance per watt DSE candidate summary",
        )
        perf_watt_report = out_dir / "perf-watt-dse-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(perf_watt_report),
                "--artifact",
                str(perf_watt_candidate_summary),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
            "performance per watt DSE report bundle",
        )
        perf_watt_data = json.loads(perf_watt_report.read_text())
        perf_watt_objective = perf_watt_data.get("objective_records", [])[0]
        expected_perf_watt_objective = {
            "objective_id": "objective::maximize_performance_per_watt",
            "objective_kind": "maximize_performance_per_watt",
            "constraint_or_optimization_mode": "optimization",
            "comparison_direction": "maximize",
            "units": "items_per_s_per_w",
        }
        for key, value in expected_perf_watt_objective.items():
            if perf_watt_objective.get(key) != value:
                raise AssertionError(f"unexpected performance per watt objective {key}: {perf_watt_objective}")
        if perf_watt_objective.get("metric_inputs") != ["metric::vecsum::performance_per_watt"]:
            raise AssertionError(f"performance per watt objective should cite performance metric: {perf_watt_objective}")
        if perf_watt_data.get("candidate_ordering_rule") != "performance_per_watt_score_then_candidate_id":
            raise AssertionError(f"unexpected performance per watt ordering rule: {perf_watt_data}")
        if perf_watt_data.get("selected_policy_id") != "deterministic_maximize_performance_per_watt_v1":
            raise AssertionError(f"unexpected performance per watt policy id: {perf_watt_data}")
        perf_watt_candidate = perf_watt_data.get("candidate_list", [])[0]
        if "metric::vecsum::performance_per_watt" not in perf_watt_candidate.get("metric_records_used", []):
            raise AssertionError(f"performance per watt candidate missed performance metric: {perf_watt_candidate}")
        perf_watt_audit = out_dir / "perf-watt-dse-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(perf_watt_audit),
                str(perf_watt_report),
            ],
            "performance per watt DSE report bundle audit",
        )

        perf_area_candidate_summary = out_dir / "perf-area-dse-candidate-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/dse/run_candidate_summary.sh",
                "--objective",
                "maximize_performance_per_area",
                "--output",
                str(perf_area_candidate_summary),
                "--artifact",
                str(out_dir / "pnr-mapping-summary.csv"),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "performance per area DSE candidate summary",
        )
        perf_area_report = out_dir / "perf-area-dse-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(perf_area_report),
                "--artifact",
                str(perf_area_candidate_summary),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
            "performance per area DSE report bundle",
        )
        perf_area_data = json.loads(perf_area_report.read_text())
        perf_area_objective = perf_area_data.get("objective_records", [])[0]
        expected_perf_area_objective = {
            "objective_id": "objective::maximize_performance_per_area",
            "objective_kind": "maximize_performance_per_area",
            "constraint_or_optimization_mode": "optimization",
            "comparison_direction": "maximize",
            "units": "items_per_s_per_um2",
        }
        for key, value in expected_perf_area_objective.items():
            if perf_area_objective.get(key) != value:
                raise AssertionError(f"unexpected performance per area objective {key}: {perf_area_objective}")
        if perf_area_objective.get("metric_inputs") != ["metric::vecsum::performance_per_area"]:
            raise AssertionError(f"performance per area objective should cite performance metric: {perf_area_objective}")
        if perf_area_data.get("candidate_ordering_rule") != "performance_per_area_score_then_candidate_id":
            raise AssertionError(f"unexpected performance per area ordering rule: {perf_area_data}")
        if perf_area_data.get("selected_policy_id") != "deterministic_maximize_performance_per_area_v1":
            raise AssertionError(f"unexpected performance per area policy id: {perf_area_data}")
        perf_area_candidate = perf_area_data.get("candidate_list", [])[0]
        if "metric::vecsum::performance_per_area" not in perf_area_candidate.get("metric_records_used", []):
            raise AssertionError(f"performance per area candidate missed performance metric: {perf_area_candidate}")
        perf_area_audit = out_dir / "perf-area-dse-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(perf_area_audit),
                str(perf_area_report),
            ],
            "performance per area DSE report bundle audit",
        )

        area_candidate_summary = out_dir / "area-dse-candidate-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/dse/run_candidate_summary.sh",
                "--objective",
                "minimize_area",
                "--output",
                str(area_candidate_summary),
                "--artifact",
                str(out_dir / "pnr-mapping-summary.csv"),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "area DSE candidate summary",
        )
        area_report = out_dir / "area-dse-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(area_report),
                "--artifact",
                str(area_candidate_summary),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
            "area DSE report bundle",
        )
        area_data = json.loads(area_report.read_text())
        area_objective = area_data.get("objective_records", [])[0]
        expected_area_objective = {
            "objective_id": "objective::minimize_area",
            "objective_kind": "minimize_area",
            "constraint_or_optimization_mode": "optimization",
            "comparison_direction": "minimize",
            "units": "um2",
        }
        for key, value in expected_area_objective.items():
            if area_objective.get(key) != value:
                raise AssertionError(f"unexpected area objective {key}: {area_objective}")
        if area_objective.get("metric_inputs") != ["metric::shared_reduction_adg::area_um2"]:
            raise AssertionError(f"area objective should cite area metric: {area_objective}")
        if area_data.get("candidate_ordering_rule") != "area_score_then_candidate_id":
            raise AssertionError(f"unexpected area ordering rule: {area_data}")
        if area_data.get("selected_policy_id") != "deterministic_minimize_area_v1":
            raise AssertionError(f"unexpected area policy id: {area_data}")
        area_candidate = area_data.get("candidate_list", [])[0]
        if "metric::shared_reduction_adg::area_um2" not in area_candidate.get("metric_records_used", []):
            raise AssertionError(f"area candidate missed area metric: {area_candidate}")
        area_audit = out_dir / "area-dse-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(area_audit),
                str(area_report),
            ],
            "area DSE report bundle audit",
        )

        dynamic_power_candidate_summary = out_dir / "dynamic-power-dse-candidate-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/dse/run_candidate_summary.sh",
                "--objective",
                "minimize_dynamic_power",
                "--output",
                str(dynamic_power_candidate_summary),
                "--artifact",
                str(out_dir / "pnr-mapping-summary.csv"),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "dynamic power DSE candidate summary",
        )
        dynamic_power_report = out_dir / "dynamic-power-dse-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(dynamic_power_report),
                "--artifact",
                str(dynamic_power_candidate_summary),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
            "dynamic power DSE report bundle",
        )
        dynamic_power_data = json.loads(dynamic_power_report.read_text())
        dynamic_power_objective = dynamic_power_data.get("objective_records", [])[0]
        expected_dynamic_power_objective = {
            "objective_id": "objective::minimize_dynamic_power",
            "objective_kind": "minimize_dynamic_power",
            "constraint_or_optimization_mode": "optimization",
            "comparison_direction": "minimize",
            "units": "mW",
        }
        for key, value in expected_dynamic_power_objective.items():
            if dynamic_power_objective.get(key) != value:
                raise AssertionError(f"unexpected dynamic power objective {key}: {dynamic_power_objective}")
        expected_dynamic_power_metric = "metric::shared_reduction_adg::dynamic_power_mw"
        if dynamic_power_objective.get("metric_inputs") != [expected_dynamic_power_metric]:
            raise AssertionError(f"dynamic power objective should cite power metric: {dynamic_power_objective}")
        if dynamic_power_data.get("candidate_ordering_rule") != "dynamic_power_score_then_candidate_id":
            raise AssertionError(f"unexpected dynamic power ordering rule: {dynamic_power_data}")
        if dynamic_power_data.get("selected_policy_id") != "deterministic_minimize_dynamic_power_v1":
            raise AssertionError(f"unexpected dynamic power policy id: {dynamic_power_data}")
        dynamic_power_candidate = dynamic_power_data.get("candidate_list", [])[0]
        if expected_dynamic_power_metric not in dynamic_power_candidate.get("metric_records_used", []):
            raise AssertionError(f"dynamic power candidate missed power metric: {dynamic_power_candidate}")
        dynamic_power_audit = out_dir / "dynamic-power-dse-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(dynamic_power_audit),
                str(dynamic_power_report),
            ],
            "dynamic power DSE report bundle audit",
        )

        leakage_power_candidate_summary = out_dir / "leakage-power-dse-candidate-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/dse/run_candidate_summary.sh",
                "--objective",
                "minimize_leakage_power",
                "--output",
                str(leakage_power_candidate_summary),
                "--artifact",
                str(out_dir / "pnr-mapping-summary.csv"),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "leakage power DSE candidate summary",
        )
        leakage_power_report = out_dir / "leakage-power-dse-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(leakage_power_report),
                "--artifact",
                str(leakage_power_candidate_summary),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
            "leakage power DSE report bundle",
        )
        leakage_power_data = json.loads(leakage_power_report.read_text())
        leakage_power_objective = leakage_power_data.get("objective_records", [])[0]
        expected_leakage_power_objective = {
            "objective_id": "objective::minimize_leakage_power",
            "objective_kind": "minimize_leakage_power",
            "constraint_or_optimization_mode": "optimization",
            "comparison_direction": "minimize",
            "units": "mW",
        }
        for key, value in expected_leakage_power_objective.items():
            if leakage_power_objective.get(key) != value:
                raise AssertionError(f"unexpected leakage power objective {key}: {leakage_power_objective}")
        expected_leakage_power_metric = "metric::shared_reduction_adg::leakage_power_mw"
        if leakage_power_objective.get("metric_inputs") != [expected_leakage_power_metric]:
            raise AssertionError(f"leakage power objective should cite power metric: {leakage_power_objective}")
        if leakage_power_data.get("candidate_ordering_rule") != "leakage_power_score_then_candidate_id":
            raise AssertionError(f"unexpected leakage power ordering rule: {leakage_power_data}")
        if leakage_power_data.get("selected_policy_id") != "deterministic_minimize_leakage_power_v1":
            raise AssertionError(f"unexpected leakage power policy id: {leakage_power_data}")
        leakage_power_candidate = leakage_power_data.get("candidate_list", [])[0]
        if expected_leakage_power_metric not in leakage_power_candidate.get("metric_records_used", []):
            raise AssertionError(f"leakage power candidate missed power metric: {leakage_power_candidate}")
        leakage_power_audit = out_dir / "leakage-power-dse-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(leakage_power_audit),
                str(leakage_power_report),
            ],
            "leakage power DSE report bundle audit",
        )

        unsupported_scope_ledger = out_dir / "dse-objective-unsupported-scope-ledger.csv"
        unsupported_scope_ledger.write_text(
            "stage,case,artifact,reason,owner,blocking_input\n"
            "dse,candidate::vecsum::shared_reduction_adg::vecsum__shared_reduction_adg,"
            f"dse-candidate-summary,synthetic candidate diagnostic,implementation,{out_dir / 'pnr-mapping.json'}\n"
        )
        unsupported_scope_candidate_summary = out_dir / "unsupported-scope-dse-candidate-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/dse/run_candidate_summary.sh",
                "--objective",
                "minimize_unsupported_scope_diagnostics",
                "--output",
                str(unsupported_scope_candidate_summary),
                "--artifact",
                str(out_dir / "pnr-mapping-summary.csv"),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
                "--artifact",
                str(unsupported_scope_ledger),
            ],
            "unsupported-scope objective DSE candidate summary",
        )
        unsupported_scope_report = out_dir / "unsupported-scope-dse-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(unsupported_scope_report),
                "--artifact",
                str(unsupported_scope_candidate_summary),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
            "unsupported-scope objective DSE report bundle",
        )
        unsupported_scope_data = json.loads(unsupported_scope_report.read_text())
        unsupported_scope_objective = unsupported_scope_data.get("objective_records", [])[0]
        expected_unsupported_scope_objective = {
            "objective_id": "objective::minimize_unsupported_scope_diagnostics",
            "objective_kind": "minimize_unsupported_scope_diagnostics",
            "constraint_or_optimization_mode": "optimization",
            "comparison_direction": "minimize",
            "units": "count",
        }
        for key, value in expected_unsupported_scope_objective.items():
            if unsupported_scope_objective.get(key) != value:
                raise AssertionError(
                    f"unexpected unsupported-scope objective {key}: {unsupported_scope_objective}"
                )
        expected_unsupported_scope_metric = (
            "metric::vecsum::shared_reduction_adg::"
            "vecsum__shared_reduction_adg::unsupported_scope_diagnostics_count"
        )
        if unsupported_scope_objective.get("metric_inputs") != [expected_unsupported_scope_metric]:
            raise AssertionError(
                f"unsupported-scope objective should cite diagnostic metric: {unsupported_scope_objective}"
            )
        if (
            unsupported_scope_data.get("candidate_ordering_rule")
            != "unsupported_scope_diagnostics_score_then_candidate_id"
        ):
            raise AssertionError(f"unexpected unsupported-scope ordering rule: {unsupported_scope_data}")
        if (
            unsupported_scope_data.get("selected_policy_id")
            != "deterministic_minimize_unsupported_scope_diagnostics_v1"
        ):
            raise AssertionError(f"unexpected unsupported-scope policy id: {unsupported_scope_data}")
        unsupported_scope_candidate = unsupported_scope_data.get("candidate_list", [])[0]
        if expected_unsupported_scope_metric not in unsupported_scope_candidate.get("metric_records_used", []):
            raise AssertionError(
                f"unsupported-scope candidate missed diagnostic metric: {unsupported_scope_candidate}"
            )
        unsupported_scope_audit = out_dir / "unsupported-scope-dse-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(unsupported_scope_audit),
                str(unsupported_scope_report),
            ],
            "unsupported-scope DSE report bundle audit",
        )

        stochastic_without_seed = out_dir / "stochastic-without-seed-dse-report-bundle.json"
        stochastic_without_seed_data = json.loads(report.read_text())
        stochastic_without_seed_data["policy_configuration"]["policy_kind"] = "stochastic"
        stochastic_without_seed_data["policy_configuration"]["random_seed"] = None
        stochastic_without_seed.write_text(
            json.dumps(stochastic_without_seed_data, indent=2, sort_keys=True) + "\n"
        )
        stochastic_without_seed_audit = out_dir / "stochastic-without-seed-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stochastic_without_seed_audit),
                str(stochastic_without_seed),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with stochastic policy lacking seed unexpectedly passed audit")

        mismatched_conflict_resolution = out_dir / "mismatched-conflict-resolution-dse-report-bundle.json"
        mismatched_conflict_resolution_data = json.loads(report.read_text())
        mismatched_conflict_resolution_data["policy_configuration"]["conflict_resolution"] = "weighted_score"
        mismatched_conflict_resolution.write_text(
            json.dumps(mismatched_conflict_resolution_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_conflict_resolution_audit = (
            out_dir / "mismatched-conflict-resolution-dse-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_conflict_resolution_audit),
                str(mismatched_conflict_resolution),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched conflict resolution unexpectedly passed audit")

        mismatched_policy_id = out_dir / "mismatched-policy-id-dse-report-bundle.json"
        mismatched_policy_id_data = json.loads(report.read_text())
        mismatched_policy_id_data["selected_policy_id"] = "other_policy"
        mismatched_policy_id.write_text(json.dumps(mismatched_policy_id_data, indent=2, sort_keys=True) + "\n")
        mismatched_policy_id_audit = out_dir / "mismatched-policy-id-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_policy_id_audit),
                str(mismatched_policy_id),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched selected policy id unexpectedly passed audit")

        mismatched_policy_objective = out_dir / "mismatched-policy-objective-dse-report-bundle.json"
        mismatched_policy_objective_data = json.loads(report.read_text())
        mismatched_policy_objective_data["selected_policy_id"] = "deterministic_minimize_energy_v1"
        mismatched_policy_objective_data["dse_run_id"] = "dse::deterministic_minimize_energy_v1"
        mismatched_policy_objective.write_text(
            json.dumps(mismatched_policy_objective_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_policy_objective_audit = out_dir / "mismatched-policy-objective-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_policy_objective_audit),
                str(mismatched_policy_objective),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched policy objective unexpectedly passed audit")

        mismatched_ordering_rule = out_dir / "mismatched-ordering-rule-dse-report-bundle.json"
        mismatched_ordering_rule_data = json.loads(report.read_text())
        mismatched_ordering_rule_data["candidate_ordering_rule"] = "energy_score_then_candidate_id"
        mismatched_ordering_rule.write_text(
            json.dumps(mismatched_ordering_rule_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_ordering_rule_audit = out_dir / "mismatched-ordering-rule-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_ordering_rule_audit),
                str(mismatched_ordering_rule),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched candidate ordering rule unexpectedly passed audit")

        missing_candidate_metrics = out_dir / "missing-candidate-metrics-dse-report-bundle.json"
        missing_candidate_metrics_data = json.loads(report.read_text())
        missing_candidate_metrics_data["candidate_list"][0]["metric_records_used"] = []
        missing_candidate_metrics.write_text(
            json.dumps(missing_candidate_metrics_data, indent=2, sort_keys=True) + "\n"
        )
        missing_candidate_metrics_audit = out_dir / "missing-candidate-metrics-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_candidate_metrics_audit),
                str(missing_candidate_metrics),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with selected candidate lacking metric evidence unexpectedly passed audit")

        missing_objective_metric = out_dir / "missing-objective-metric-dse-report-bundle.json"
        missing_objective_metric_data = json.loads(report.read_text())
        objective_metric_inputs = set(missing_objective_metric_data["objective_records"][0]["metric_inputs"])
        missing_objective_metric_data["candidate_list"][0]["metric_records_used"] = [
            metric
            for metric in missing_objective_metric_data["candidate_list"][0]["metric_records_used"]
            if metric not in objective_metric_inputs
        ]
        missing_objective_metric.write_text(
            json.dumps(missing_objective_metric_data, indent=2, sort_keys=True) + "\n"
        )
        missing_objective_metric_audit = out_dir / "missing-objective-metric-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_objective_metric_audit),
                str(missing_objective_metric),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with selected candidate missing objective metric unexpectedly passed audit")

        bad_candidate_metric_reference = out_dir / "bad-candidate-metric-reference-dse-report-bundle.json"
        bad_candidate_metric_reference_data = json.loads(report.read_text())
        bad_candidate_metric_reference_data["candidate_list"][0]["metric_records_used"].append("metric::missing")
        bad_candidate_metric_reference.write_text(
            json.dumps(bad_candidate_metric_reference_data, indent=2, sort_keys=True) + "\n"
        )
        bad_candidate_metric_reference_audit = out_dir / "bad-candidate-metric-reference-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_candidate_metric_reference_audit),
                str(bad_candidate_metric_reference),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with unresolved candidate metric unexpectedly passed audit")

        bad_objective_metric_reference = out_dir / "bad-objective-metric-reference-dse-report-bundle.json"
        bad_objective_metric_reference_data = json.loads(report.read_text())
        bad_objective_metric_reference_data["objective_records"][0]["metric_inputs"].append("metric::missing")
        bad_objective_metric_reference.write_text(
            json.dumps(bad_objective_metric_reference_data, indent=2, sort_keys=True) + "\n"
        )
        bad_objective_metric_reference_audit = out_dir / "bad-objective-metric-reference-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_objective_metric_reference_audit),
                str(bad_objective_metric_reference),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with unresolved objective metric unexpectedly passed audit")

        mismatched_objective_identity = out_dir / "mismatched-objective-identity-dse-report-bundle.json"
        mismatched_objective_identity_data = json.loads(report.read_text())
        mismatched_objective_identity_data["objective_records"][0]["objective_id"] = "objective::other"
        mismatched_objective_identity_data["candidate_list"][0]["objective_records_used"] = ["objective::other"]
        mismatched_objective_identity.write_text(
            json.dumps(mismatched_objective_identity_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_objective_identity_audit = out_dir / "mismatched-objective-identity-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_objective_identity_audit),
                str(mismatched_objective_identity),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched objective identity unexpectedly passed audit")

        mismatched_objective_units = out_dir / "mismatched-objective-units-dse-report-bundle.json"
        mismatched_objective_units_data = json.loads(report.read_text())
        mismatched_objective_units_data["objective_records"][0]["units"] = "nJ"
        mismatched_objective_units.write_text(
            json.dumps(mismatched_objective_units_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_objective_units_audit = out_dir / "mismatched-objective-units-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_objective_units_audit),
                str(mismatched_objective_units),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched objective units unexpectedly passed audit")

        missing_objective_mode = out_dir / "missing-objective-mode-dse-report-bundle.json"
        missing_objective_mode_data = json.loads(report.read_text())
        del missing_objective_mode_data["objective_records"][0]["constraint_or_optimization_mode"]
        missing_objective_mode.write_text(
            json.dumps(missing_objective_mode_data, indent=2, sort_keys=True) + "\n"
        )
        missing_objective_mode_audit = out_dir / "missing-objective-mode-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_objective_mode_audit),
                str(missing_objective_mode),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with objective lacking mode unexpectedly passed audit")

        missing_candidate_outputs = out_dir / "missing-candidate-outputs-dse-report-bundle.json"
        missing_candidate_outputs_data = json.loads(report.read_text())
        missing_candidate_outputs_data["candidate_list"][0]["generated_output_artifacts"] = []
        missing_candidate_outputs.write_text(
            json.dumps(missing_candidate_outputs_data, indent=2, sort_keys=True) + "\n"
        )
        missing_candidate_outputs_audit = out_dir / "missing-candidate-outputs-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_candidate_outputs_audit),
                str(missing_candidate_outputs),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with selected candidate lacking output artifacts unexpectedly passed audit")

        bad_candidate_output_reference = out_dir / "bad-candidate-output-reference-dse-report-bundle.json"
        bad_candidate_output_reference_data = json.loads(report.read_text())
        bad_candidate_output_reference_data["candidate_list"][0]["generated_output_artifacts"] = [
            "missing-output.csv"
        ]
        bad_candidate_output_reference.write_text(
            json.dumps(bad_candidate_output_reference_data, indent=2, sort_keys=True) + "\n"
        )
        bad_candidate_output_reference_audit = out_dir / "bad-candidate-output-reference-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_candidate_output_reference_audit),
                str(bad_candidate_output_reference),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with unresolved candidate output artifact unexpectedly passed audit")

        missing_candidate_objectives = out_dir / "missing-candidate-objectives-dse-report-bundle.json"
        missing_candidate_objectives_data = json.loads(report.read_text())
        missing_candidate_objectives_data["candidate_list"][0]["objective_records_used"] = []
        missing_candidate_objectives.write_text(
            json.dumps(missing_candidate_objectives_data, indent=2, sort_keys=True) + "\n"
        )
        missing_candidate_objectives_audit = out_dir / "missing-candidate-objectives-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_candidate_objectives_audit),
                str(missing_candidate_objectives),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with selected candidate lacking objective evidence unexpectedly passed audit")

        mismatched_selected_status = out_dir / "mismatched-selected-status-dse-report-bundle.json"
        mismatched_selected_status_data = json.loads(report.read_text())
        mismatched_selected_status_data["candidate_list"][0]["status"] = "rejected"
        mismatched_selected_status.write_text(
            json.dumps(mismatched_selected_status_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_selected_status_audit = out_dir / "mismatched-selected-status-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_selected_status_audit),
                str(mismatched_selected_status),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with selected candidate status mismatch unexpectedly passed audit")

        duplicate_selected = out_dir / "duplicate-selected-dse-report-bundle.json"
        duplicate_selected_data = json.loads(report.read_text())
        duplicate_selected_data["selected_candidates"] = [candidate_id, candidate_id]
        duplicate_selected.write_text(json.dumps(duplicate_selected_data, indent=2, sort_keys=True) + "\n")
        duplicate_selected_audit = out_dir / "duplicate-selected-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(duplicate_selected_audit),
                str(duplicate_selected),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with duplicate selected candidate unexpectedly passed audit")

        overlapping_selection = out_dir / "overlapping-selection-dse-report-bundle.json"
        overlapping_selection_data = json.loads(report.read_text())
        overlapping_selection_data["pareto_set"] = [candidate_id]
        overlapping_selection.write_text(json.dumps(overlapping_selection_data, indent=2, sort_keys=True) + "\n")
        overlapping_selection_audit = out_dir / "overlapping-selection-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(overlapping_selection_audit),
                str(overlapping_selection),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with overlapping selected and Pareto candidates unexpectedly passed audit")

        unlisted_selected = out_dir / "unlisted-selected-dse-report-bundle.json"
        unlisted_selected_data = json.loads(report.read_text())
        extra_selected_candidate = dict(unlisted_selected_data["candidate_list"][0])
        extra_selected_candidate["candidate_id"] = "candidate::unlisted-selected"
        extra_selected_candidate["status"] = "selected"
        unlisted_selected_data["candidate_list"].append(extra_selected_candidate)
        unlisted_selected.write_text(json.dumps(unlisted_selected_data, indent=2, sort_keys=True) + "\n")
        unlisted_selected_audit = out_dir / "unlisted-selected-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(unlisted_selected_audit),
                str(unlisted_selected),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with unlisted selected candidate unexpectedly passed audit")

        unbacked_selected = out_dir / "unbacked-selected-dse-report-bundle.json"
        unbacked_selected_data = json.loads(report.read_text())
        unbacked_candidate = dict(unbacked_selected_data["candidate_list"][0])
        unbacked_candidate["candidate_id"] = "candidate::unbacked-selected"
        unbacked_selected_data["candidate_list"] = [unbacked_candidate]
        unbacked_selected_data["selected_candidates"] = ["candidate::unbacked-selected"]
        unbacked_selected.write_text(json.dumps(unbacked_selected_data, indent=2, sort_keys=True) + "\n")
        unbacked_selected_audit = out_dir / "unbacked-selected-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(unbacked_selected_audit),
                str(unbacked_selected),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with unbacked selected candidate unexpectedly passed audit")

        unlisted_pareto = out_dir / "unlisted-pareto-dse-report-bundle.json"
        unlisted_pareto_data = json.loads(report.read_text())
        extra_pareto_candidate = dict(unlisted_pareto_data["candidate_list"][0])
        extra_pareto_candidate["candidate_id"] = "candidate::unlisted-pareto"
        extra_pareto_candidate["status"] = "pareto"
        unlisted_pareto_data["candidate_list"].append(extra_pareto_candidate)
        unlisted_pareto.write_text(json.dumps(unlisted_pareto_data, indent=2, sort_keys=True) + "\n")
        unlisted_pareto_audit = out_dir / "unlisted-pareto-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(unlisted_pareto_audit),
                str(unlisted_pareto),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with unlisted Pareto candidate unexpectedly passed audit")

        missing_rejected_summary = out_dir / "missing-rejected-summary-dse-report-bundle.json"
        missing_rejected_summary_data = json.loads(report.read_text())
        rejected_candidate = dict(missing_rejected_summary_data["candidate_list"][0])
        rejected_candidate["candidate_id"] = "candidate::rejected"
        rejected_candidate["status"] = "rejected"
        rejected_candidate["diagnostics"] = ["dominated by selected candidate"]
        missing_rejected_summary_data["candidate_list"].append(rejected_candidate)
        missing_rejected_summary.write_text(
            json.dumps(missing_rejected_summary_data, indent=2, sort_keys=True) + "\n"
        )
        missing_rejected_summary_audit = out_dir / "missing-rejected-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_rejected_summary_audit),
                str(missing_rejected_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with rejected candidate missing summary unexpectedly passed audit")

        bad_rejected_summary = out_dir / "bad-rejected-summary-dse-report-bundle.json"
        bad_rejected_summary_data = json.loads(report.read_text())
        bad_rejected_summary_data["rejected_candidate_summaries"] = [
            {"candidate_id": candidate_id, "diagnostics": []}
        ]
        bad_rejected_summary.write_text(json.dumps(bad_rejected_summary_data, indent=2, sort_keys=True) + "\n")
        bad_rejected_summary_audit = out_dir / "bad-rejected-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_rejected_summary_audit),
                str(bad_rejected_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with rejected summary for non-rejected candidate unexpectedly passed audit")

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
        missing_runtime_summary = out_dir / "missing-runtime-summary-fingerprint-dse-report-bundle.json"
        missing_runtime_summary_data = json.loads(report.read_text())
        missing_runtime_summary_data["runtime_evidence_summaries"][0]["input_artifact_fingerprints"].pop(
            "pnr-mapping",
            None,
        )
        missing_runtime_summary.write_text(json.dumps(missing_runtime_summary_data, indent=2, sort_keys=True) + "\n")
        missing_runtime_summary_audit = out_dir / "missing-runtime-summary-fingerprint-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_runtime_summary_audit),
                str(missing_runtime_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report without runtime summary input fingerprint unexpectedly passed audit")

        mismatched_runtime_summary_source = out_dir / "mismatched-runtime-summary-source-dse-report-bundle.json"
        mismatched_runtime_summary_source_data = json.loads(report.read_text())
        mismatched_runtime_summary_source_data["runtime_evidence_summaries"][0][
            "workload_report_bundle_identity"
        ] = "workload-report-bundle-other"
        mismatched_runtime_summary_source.write_text(
            json.dumps(mismatched_runtime_summary_source_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_runtime_summary_source_audit = (
            out_dir / "mismatched-runtime-summary-source-dse-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_runtime_summary_source_audit),
                str(mismatched_runtime_summary_source),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched runtime summary source unexpectedly passed audit")

        mismatched_runtime_report_summary = out_dir / "mismatched-runtime-report-summary-dse-report-bundle.json"
        mismatched_runtime_report_summary_data = json.loads(report.read_text())
        mismatched_runtime_report_summary_data["runtime_evidence_summaries"][0][
            "runtime_report_identity"
        ] = "runtime-report::other"
        mismatched_runtime_report_summary_data["runtime_evidence_summaries"][0]["report_output_configuration"][
            "runtime_report_identity"
        ] = "runtime-report::other"
        mismatched_runtime_report_summary.write_text(
            json.dumps(mismatched_runtime_report_summary_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_runtime_report_summary_audit = (
            out_dir / "mismatched-runtime-report-summary-dse-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_runtime_report_summary_audit),
                str(mismatched_runtime_report_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError(
                "DSE report with mismatched referenced runtime report summary unexpectedly passed audit"
            )

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

        bad_runtime_fallback_summary = out_dir / "bad-runtime-fallback-summary-dse-report-bundle.json"
        bad_runtime_fallback_summary_data = json.loads(report.read_text())
        bad_runtime_fallback_summary_data["runtime_evidence_summaries"][0]["fallback_decision"][
            "policy"
        ] = "allow_host_fallback"
        bad_runtime_fallback_summary_data["runtime_evidence_summaries"][0]["fallback_decision"][
            "target_profile_id"
        ] = "simulator::dfg_sim::optimistic_pipeline_latency_throughput_sum"
        bad_runtime_fallback_summary.write_text(
            json.dumps(bad_runtime_fallback_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_fallback_summary_audit = out_dir / "bad-runtime-fallback-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_fallback_summary_audit),
                str(bad_runtime_fallback_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched runtime fallback summary unexpectedly passed audit")

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

        bad_runtime_launch_summary = out_dir / "bad-runtime-launch-summary-dse-report-bundle.json"
        bad_runtime_launch_summary_data = json.loads(report.read_text())
        bad_runtime_launch_summary_data["runtime_evidence_summaries"][0]["launch_descriptor"][
            "descriptor_id"
        ] = "launch::other"
        bad_runtime_launch_summary.write_text(
            json.dumps(bad_runtime_launch_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_launch_summary_audit = out_dir / "bad-runtime-launch-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_launch_summary_audit),
                str(bad_runtime_launch_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched runtime launch summary unexpectedly passed audit")

        bad_runtime_scalar_summary = out_dir / "bad-runtime-scalar-summary-dse-report-bundle.json"
        bad_runtime_scalar_summary_data = json.loads(report.read_text())
        bad_runtime_scalar_summary_data["runtime_evidence_summaries"][0]["launch_descriptor"][
            "scalar_value_descriptors"
        ] = "scalar"
        bad_runtime_scalar_summary.write_text(
            json.dumps(bad_runtime_scalar_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_scalar_summary_audit = out_dir / "bad-runtime-scalar-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_scalar_summary_audit),
                str(bad_runtime_scalar_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed runtime scalar summary unexpectedly passed audit")

        bad_runtime_wrapper_summary = out_dir / "bad-runtime-wrapper-summary-dse-report-bundle.json"
        bad_runtime_wrapper_summary_data = json.loads(report.read_text())
        bad_runtime_wrapper_summary_data["runtime_evidence_summaries"][0]["host_wrapper_identity"] = []
        bad_runtime_wrapper_summary.write_text(
            json.dumps(bad_runtime_wrapper_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_wrapper_summary_audit = out_dir / "bad-runtime-wrapper-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_wrapper_summary_audit),
                str(bad_runtime_wrapper_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed runtime wrapper summary unexpectedly passed audit")

        bad_runtime_host_summary = out_dir / "bad-runtime-host-summary-dse-report-bundle.json"
        bad_runtime_host_summary_data = json.loads(report.read_text())
        bad_runtime_host_summary_data["runtime_evidence_summaries"][0]["host_interface"]["invocation_abi"] = ""
        bad_runtime_host_summary.write_text(
            json.dumps(bad_runtime_host_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_host_summary_audit = out_dir / "bad-runtime-host-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_host_summary_audit),
                str(bad_runtime_host_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed runtime host summary unexpectedly passed audit")

        bad_runtime_host_source_summary = out_dir / "bad-runtime-host-source-summary-dse-report-bundle.json"
        bad_runtime_host_source_summary_data = json.loads(report.read_text())
        bad_runtime_host_source_summary_data["runtime_evidence_summaries"][0]["host_interface"][
            "source_provenance"
        ] = "test-app-fixture::other::default"
        bad_runtime_host_source_summary.write_text(
            json.dumps(bad_runtime_host_source_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_host_source_summary_audit = (
            out_dir / "bad-runtime-host-source-summary-dse-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_host_source_summary_audit),
                str(bad_runtime_host_source_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched runtime host source summary unexpectedly passed audit")

        bad_runtime_handle_summary = out_dir / "bad-runtime-handle-summary-dse-report-bundle.json"
        bad_runtime_handle_summary_data = json.loads(report.read_text())
        bad_runtime_handle_summary_data["runtime_evidence_summaries"][0]["runtime_handle_model"][
            "ir_token_kind"
        ] = "dataflow_thread_token"
        bad_runtime_handle_summary.write_text(
            json.dumps(bad_runtime_handle_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_handle_summary_audit = out_dir / "bad-runtime-handle-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_handle_summary_audit),
                str(bad_runtime_handle_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with dataflow-backed runtime handle summary unexpectedly passed audit")

        bad_runtime_work_package_summary = out_dir / "bad-runtime-work-package-summary-dse-report-bundle.json"
        bad_runtime_work_package_summary_data = json.loads(report.read_text())
        bad_runtime_work_package_summary_data["runtime_evidence_summaries"][0]["work_package_metadata"][
            "selected_mapping_artifact_identity"
        ] = "other-mapping"
        bad_runtime_work_package_summary.write_text(
            json.dumps(bad_runtime_work_package_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_work_package_summary_audit = (
            out_dir / "bad-runtime-work-package-summary-dse-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_work_package_summary_audit),
                str(bad_runtime_work_package_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched runtime work package summary unexpectedly passed audit")

        bad_runtime_report_output_summary = out_dir / "bad-runtime-report-output-summary-dse-report-bundle.json"
        bad_runtime_report_output_summary_data = json.loads(report.read_text())
        bad_runtime_report_output_summary_data["runtime_evidence_summaries"][0]["report_output_configuration"][
            "runtime_report_identity"
        ] = "runtime-report::other"
        bad_runtime_report_output_summary.write_text(
            json.dumps(bad_runtime_report_output_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_report_output_summary_audit = (
            out_dir / "bad-runtime-report-output-summary-dse-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_report_output_summary_audit),
                str(bad_runtime_report_output_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched runtime report output summary unexpectedly passed audit")

        bad_runtime_memory_summary = out_dir / "bad-runtime-memory-summary-dse-report-bundle.json"
        bad_runtime_memory_summary_data = json.loads(report.read_text())
        bad_runtime_memory_summary_data["runtime_evidence_summaries"][0]["memory_descriptors"][0][
            "host_buffer_identity"
        ] = []
        bad_runtime_memory_summary.write_text(
            json.dumps(bad_runtime_memory_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_memory_summary_audit = out_dir / "bad-runtime-memory-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_memory_summary_audit),
                str(bad_runtime_memory_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed runtime memory summary unexpectedly passed audit")

        bad_runtime_arguments_summary = out_dir / "bad-runtime-arguments-summary-dse-report-bundle.json"
        bad_runtime_arguments_summary_data = json.loads(report.read_text())
        bad_runtime_arguments_summary_data["runtime_evidence_summaries"][0][
            "argument_descriptors"
        ] = "runtime_input"
        bad_runtime_arguments_summary.write_text(
            json.dumps(bad_runtime_arguments_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_arguments_summary_audit = out_dir / "bad-runtime-arguments-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_arguments_summary_audit),
                str(bad_runtime_arguments_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed runtime arguments summary unexpectedly passed audit")

        bad_runtime_target_summary = out_dir / "bad-runtime-target-summary-dse-report-bundle.json"
        bad_runtime_target_summary_data = json.loads(report.read_text())
        bad_runtime_target_summary_data["runtime_evidence_summaries"][0]["target_profile"][
            "profile_id"
        ] = "simulator::other"
        bad_runtime_target_summary.write_text(
            json.dumps(bad_runtime_target_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_target_summary_audit = out_dir / "bad-runtime-target-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_target_summary_audit),
                str(bad_runtime_target_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched runtime target summary unexpectedly passed audit")

        bad_runtime_configuration_summary = out_dir / "bad-runtime-configuration-summary-dse-report-bundle.json"
        bad_runtime_configuration_summary_data = json.loads(report.read_text())
        bad_runtime_configuration_summary_data["runtime_evidence_summaries"][0]["runtime_configuration"][
            "synchronization_mode"
        ] = "device_poll"
        bad_runtime_configuration_summary.write_text(
            json.dumps(bad_runtime_configuration_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_configuration_summary_audit = (
            out_dir / "bad-runtime-configuration-summary-dse-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_configuration_summary_audit),
                str(bad_runtime_configuration_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched runtime configuration summary unexpectedly passed audit")

        bad_runtime_features_summary = out_dir / "bad-runtime-features-summary-dse-report-bundle.json"
        bad_runtime_features_summary_data = json.loads(report.read_text())
        bad_runtime_features_summary_data["runtime_evidence_summaries"][0]["required_runtime_features"] = [
            "simulator_dispatch",
            "",
        ]
        bad_runtime_features_summary.write_text(
            json.dumps(bad_runtime_features_summary_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_features_summary_audit = out_dir / "bad-runtime-features-summary-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_features_summary_audit),
                str(bad_runtime_features_summary),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with malformed runtime feature summary unexpectedly passed audit")

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

        mismatched_hardware_bundle = out_dir / "mismatched-hardware-report-bundle.json"
        mismatched_hardware_bundle_data = json.loads((out_dir / "hardware-report-bundle.json").read_text())
        mismatched_hardware_bundle_data["bundle_id"] = "hardware-report::other_hardware"
        mismatched_hardware_bundle_data["hardware_candidate_identity"] = "other_hardware"
        mismatched_hardware_bundle_data["fabric_adg_identity"] = "other_hardware"
        mismatched_hardware_bundle.write_text(
            json.dumps(mismatched_hardware_bundle_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_hardware_report = out_dir / "mismatched-hardware-dse-report-bundle.json"
        mismatched_hardware_report_data = json.loads(report.read_text())
        mismatched_hardware_report_data["referenced_hardware_candidate_report_bundle_identities"] = [
            "mismatched-hardware-report-bundle"
        ]
        mismatched_hardware_report_data["input_artifact_fingerprints"].pop("hardware-report-bundle", None)
        mismatched_hardware_report_data["input_artifact_fingerprints"][
            "mismatched-hardware-report-bundle"
        ] = artifact_test_common.fingerprint(mismatched_hardware_bundle)
        mismatched_hardware_report.write_text(
            json.dumps(mismatched_hardware_report_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_hardware_report_audit = out_dir / "mismatched-hardware-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_hardware_report_audit),
                str(mismatched_hardware_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched hardware report bundle unexpectedly passed audit")

        mismatched_workload_bundle = out_dir / "mismatched-workload-report-bundle.json"
        mismatched_workload_bundle_data = json.loads((out_dir / "workload-report-bundle.json").read_text())
        mismatched_workload_bundle_data["bundle_id"] = (
            "workload::other_workload::shared_reduction_adg::vecsum__shared_reduction_adg"
        )
        mismatched_workload_bundle_data["workload"] = "other_workload"
        mismatched_workload_bundle.write_text(
            json.dumps(mismatched_workload_bundle_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_workload_report = out_dir / "mismatched-workload-dse-report-bundle.json"
        mismatched_workload_report_data = json.loads(report.read_text())
        mismatched_workload_report_data["referenced_workload_report_bundle_identities"] = [
            "mismatched-workload-report-bundle"
        ]
        mismatched_workload_report_data["runtime_evidence_summaries"][0][
            "workload_report_bundle_identity"
        ] = "mismatched-workload-report-bundle"
        mismatched_workload_report_data["input_artifact_fingerprints"].pop("workload-report-bundle", None)
        mismatched_workload_report_data["input_artifact_fingerprints"][
            "mismatched-workload-report-bundle"
        ] = artifact_test_common.fingerprint(mismatched_workload_bundle)
        mismatched_workload_report.write_text(
            json.dumps(mismatched_workload_report_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_workload_report_audit = out_dir / "mismatched-workload-dse-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_workload_report_audit),
                str(mismatched_workload_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched workload report bundle unexpectedly passed audit")

        mismatched_workload_hardware_bundle = out_dir / "mismatched-workload-hardware-report-bundle.json"
        mismatched_workload_hardware_bundle_data = json.loads((out_dir / "workload-report-bundle.json").read_text())
        mismatched_workload_hardware_bundle_data["bundle_id"] = (
            "workload::vecsum::other_hardware::vecsum__shared_reduction_adg"
        )
        mismatched_workload_hardware_bundle_data["selected_hardware_candidate_identity"] = "other_hardware"
        mismatched_workload_hardware_bundle.write_text(
            json.dumps(mismatched_workload_hardware_bundle_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_workload_hardware_report = out_dir / "mismatched-workload-hardware-dse-report-bundle.json"
        mismatched_workload_hardware_report_data = json.loads(report.read_text())
        mismatched_workload_hardware_report_data["referenced_workload_report_bundle_identities"] = [
            "mismatched-workload-hardware-report-bundle"
        ]
        mismatched_workload_hardware_report_data["runtime_evidence_summaries"][0][
            "workload_report_bundle_identity"
        ] = "mismatched-workload-hardware-report-bundle"
        mismatched_workload_hardware_report_data["input_artifact_fingerprints"].pop(
            "workload-report-bundle",
            None,
        )
        mismatched_workload_hardware_report_data["input_artifact_fingerprints"][
            "mismatched-workload-hardware-report-bundle"
        ] = artifact_test_common.fingerprint(mismatched_workload_hardware_bundle)
        mismatched_workload_hardware_report.write_text(
            json.dumps(mismatched_workload_hardware_report_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_workload_hardware_report_audit = (
            out_dir / "mismatched-workload-hardware-dse-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_workload_hardware_report_audit),
                str(mismatched_workload_hardware_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with mismatched workload hardware unexpectedly passed audit")

        unrelated_workload_report = out_dir / "unrelated-workload-report-bundle.json"
        unrelated_workload_report_data = json.loads((out_dir / "workload-report-bundle.json").read_text())
        unrelated_workload_report_data["bundle_id"] = (
            "workload::other_workload::shared_reduction_adg::vecsum__shared_reduction_adg"
        )
        unrelated_workload_report_data["workload"] = "other_workload"
        unrelated_workload_report.write_text(
            json.dumps(unrelated_workload_report_data, indent=2, sort_keys=True) + "\n"
        )
        unrelated_hardware_report = out_dir / "unrelated-hardware-report-bundle.json"
        unrelated_hardware_report_data = json.loads((out_dir / "hardware-report-bundle.json").read_text())
        unrelated_hardware_report_data["bundle_id"] = "hardware::other_hardware"
        unrelated_hardware_report_data["hardware_candidate_identity"] = "other_hardware"
        unrelated_hardware_report_data["fabric_adg_identity"] = "other_hardware"
        unrelated_hardware_report.write_text(
            json.dumps(unrelated_hardware_report_data, indent=2, sort_keys=True) + "\n"
        )
        filtered_report = out_dir / "filtered-dse-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(filtered_report),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(unrelated_workload_report),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
                "--artifact",
                str(unrelated_hardware_report),
            ],
            "DSE report bundle with unrelated passing reports",
        )
        filtered_data = json.loads(filtered_report.read_text())
        if filtered_data["referenced_workload_report_bundle_identities"] != ["workload-report-bundle"]:
            raise AssertionError(f"DSE report should ignore unrelated workload reports: {filtered_data}")
        if filtered_data["referenced_hardware_candidate_report_bundle_identities"] != ["hardware-report-bundle"]:
            raise AssertionError(f"DSE report should ignore unrelated hardware reports: {filtered_data}")
        if sorted(filtered_data["input_artifact_fingerprints"]) != [
            "dse-candidate-summary",
            "hardware-report-bundle",
            "workload-report-bundle",
        ]:
            raise AssertionError(f"DSE report should fingerprint only selected report inputs: {filtered_data}")
        summary_ids = [
            summary.get("workload_report_bundle_identity")
            for summary in filtered_data.get("runtime_evidence_summaries", [])
            if isinstance(summary, dict)
        ]
        if summary_ids != ["workload-report-bundle"]:
            raise AssertionError(f"DSE report should summarize only selected workload runtime evidence: {filtered_data}")

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

        alternate_mapping_workload_report = out_dir / "alternate-mapping-workload-report-bundle.json"
        alternate_mapping_data = json.loads((out_dir / "workload-report-bundle.json").read_text())
        alternate_mapping_data["bundle_id"] = "workload::vecsum::shared_reduction_adg::alternate_mapping"
        alternate_mapping_data["selected_mapping_artifact_identity"] = "alternate-mapping"
        alternate_mapping_data["runtime_evidence"]["mapping_artifact_identity"] = "alternate-mapping"
        alternate_mapping_data["runtime_evidence"]["work_package_metadata"][
            "selected_mapping_artifact_identity"
        ] = "alternate-mapping"
        alternate_mapping_data["runtime_evidence"]["launch_descriptor"][
            "selected_mapping_artifact_identity"
        ] = "alternate-mapping"
        alternate_mapping_workload_report.write_text(
            json.dumps(alternate_mapping_data, indent=2, sort_keys=True) + "\n"
        )
        filtered_mapping_report = out_dir / "filtered-mapping-dse-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(filtered_mapping_report),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
                "--artifact",
                str(alternate_mapping_workload_report),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
            "DSE report bundle with alternate mapping workload report",
        )
        filtered_mapping_data = json.loads(filtered_mapping_report.read_text())
        if filtered_mapping_data["referenced_workload_report_bundle_identities"] != ["workload-report-bundle"]:
            raise AssertionError(
                f"DSE report should ignore workload reports for other mappings: {filtered_mapping_data}"
            )
        if "alternate-mapping-workload-report-bundle" in filtered_mapping_data["input_artifact_fingerprints"]:
            raise AssertionError(
                f"DSE report should not fingerprint alternate mapping workload report: {filtered_mapping_data}"
            )
        filtered_mapping_summary_ids = [
            summary.get("workload_report_bundle_identity")
            for summary in filtered_mapping_data.get("runtime_evidence_summaries", [])
            if isinstance(summary, dict)
        ]
        if filtered_mapping_summary_ids != ["workload-report-bundle"]:
            raise AssertionError(
                f"DSE report should summarize selected mapping runtime evidence: {filtered_mapping_data}"
            )

        wrong_kind_workload_report = out_dir / "wrong-kind-workload-report-bundle.json"
        wrong_kind_workload_report.write_text(
            json.dumps({"schema_version": 1, "kind": "runtime_package"}, indent=2, sort_keys=True) + "\n"
        )
        private_path_report = out_dir / "private-path-dse-report-bundle.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_dse_report_bundle.sh",
                "--output",
                str(private_path_report),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
                "--artifact",
                str(wrong_kind_workload_report),
                "--artifact",
                str(out_dir / "workload-report-bundle.json"),
                "--artifact",
                str(out_dir / "hardware-report-bundle.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DSE report with wrong-kind workload report unexpectedly passed")
        private_path_data = json.loads(private_path_report.read_text())
        private_path_text = json.dumps(private_path_data, sort_keys=True)
        if str(out_dir) in private_path_text:
            raise AssertionError(f"DSE report diagnostics should not expose private paths: {private_path_data}")
        if not any(
            "wrong-kind-workload-report-bundle is not a workload_report_bundle" in str(item)
            for item in private_path_data.get("diagnostics", [])
        ):
            raise AssertionError(f"DSE report should diagnose wrong-kind workload report by identity: {private_path_data}")
        private_path_audit = out_dir / "private-path-dse-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(private_path_audit),
                str(private_path_report),
            ],
            "blocked DSE report bundle with portable diagnostics audit",
        )

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
