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
    "runtime_host_interface",
    "runtime_evidence",
    "runtime_fallback_decision",
    "input_artifact_fingerprints",
    "report_status",
    "diagnostic_records",
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
        expected_input_fingerprints = {
            "source-compat-summary": artifact_test_common.fingerprint(out_dir / "source-compat-summary.csv"),
            "compiler-pipeline-summary": artifact_test_common.fingerprint(out_dir / "compiler-pipeline-summary.csv"),
            "pnr-mapping": artifact_test_common.fingerprint(out_dir / "pnr-mapping.json"),
            "vecsum-dfg-sim-report": artifact_test_common.fingerprint(out_dir / "vecsum-dfg-sim-report.json"),
            "vecsum-cgra-sim-report": artifact_test_common.fingerprint(out_dir / "vecsum-cgra-sim-report.json"),
            "sim-comparison-report": artifact_test_common.fingerprint(out_dir / "sim-comparison-report.json"),
            "runtime-package": artifact_test_common.fingerprint(out_dir / "runtime-package.json"),
            "rtl-fpa-summary": artifact_test_common.fingerprint(out_dir / "rtl-fpa-summary.csv"),
            "dse-candidate-summary": artifact_test_common.fingerprint(out_dir / "dse-candidate-summary.csv"),
        }
        if data["input_artifact_fingerprints"] != expected_input_fingerprints:
            raise AssertionError(f"unexpected report input fingerprints: {data}")
        optional_identities = data.get("optional_artifact_identities", {})
        if not isinstance(optional_identities, dict):
            raise AssertionError(f"report should include optional artifact identities: {data}")
        if optional_identities.get("simulation_comparison_report") != "sim-comparison-report":
            raise AssertionError(f"report should reference simulation comparison evidence: {data}")
        if optional_identities.get("runtime_package") != "runtime-package":
            raise AssertionError(f"report should reference runtime package evidence: {data}")
        expected_host_interface = {
            "host_program_identity": "test-app-host::vecsum::default",
            "host_wrapper_identity": "runtime-wrapper::vecsum::vecsum__shared_reduction_adg",
            "invocation_abi": "loom_runtime_package_v1",
            "compatibility_mode_requires_runtime": False,
            "acceleration_mode_requires_runtime_package": True,
            "source_provenance": "test-app-fixture::vecsum::default",
        }
        if data["runtime_host_interface"] != expected_host_interface:
            raise AssertionError(f"report should preserve runtime host interface: {data}")
        expected_runtime_fallback = {
            "policy": "report_only",
            "decision": "report_only",
            "fallback_taken": False,
            "target_profile_id": "simulator::cgra_sim::mapping_constraint_estimate",
            "reason": "report-only runtime package records launch metadata without executing accelerator work",
        }
        if data["runtime_fallback_decision"] != expected_runtime_fallback:
            raise AssertionError(f"report should preserve runtime fallback decision: {data}")
        runtime_package_data = json.loads((out_dir / "runtime-package.json").read_text())
        runtime_report = runtime_package_data["runtime_report"]
        expected_runtime_evidence = {
            "runtime_package_identity": "runtime-package",
            "runtime_report_identity": "runtime-report::vecsum::vecsum__shared_reduction_adg::report_only",
            "host_program_identity": runtime_report["host_program_identity"],
            "work_package_identity": runtime_report["work_package_identity"],
            "launch_descriptor_identity": runtime_report["launch_descriptor_identity"],
            "mapping_artifact_identity": runtime_report["mapping_artifact_identity"],
            "fabric_adg_identity": runtime_report["fabric_adg_identity"],
            "target_profile_id": runtime_report["target_profile_id"],
            "fallback_policy": runtime_package_data["fallback_policy"],
            "launch_status": "not_run",
            "target_status": "not_run",
            "runtime_trace_identity": "",
            "profiling_record_identity": "",
            "data_movement_policy": "simulated",
            "synchronization_mode": "host_wait",
            "required_data_movement_policies": ["simulated"],
            "required_synchronization_policies": ["host_wait"],
            "simulator_report_identities": runtime_report["simulator_report_identities"],
            "output_buffer_identities": [],
            "diagnostic_records": runtime_report["diagnostic_records"],
            "input_artifact_fingerprints": runtime_package_data["input_artifact_fingerprints"],
            "fallback_decision": expected_runtime_fallback,
        }
        if data["runtime_evidence"] != expected_runtime_evidence:
            raise AssertionError(f"report should preserve runtime evidence references: {data}")

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
        bad_host_report = out_dir / "bad-host-interface-workload-report-bundle.json"
        bad_host_data = json.loads(report.read_text())
        bad_host_data["runtime_host_interface"]["compatibility_mode_requires_runtime"] = True
        bad_host_report.write_text(json.dumps(bad_host_data, indent=2, sort_keys=True) + "\n")
        bad_host_audit = out_dir / "bad-host-interface-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_host_audit),
                str(bad_host_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report requiring runtime for compatibility mode unexpectedly passed audit")
        stale_report_fingerprint = out_dir / "stale-input-fingerprint-workload-report-bundle.json"
        stale_report_fingerprint_data = json.loads(report.read_text())
        stale_report_fingerprint_data["input_artifact_fingerprints"]["source-compat-summary"] = "0" * 64
        stale_report_fingerprint.write_text(
            json.dumps(stale_report_fingerprint_data, indent=2, sort_keys=True) + "\n"
        )
        stale_report_fingerprint_audit = out_dir / "stale-input-fingerprint-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_report_fingerprint_audit),
                str(stale_report_fingerprint),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with stale input fingerprint unexpectedly passed audit")
        bad_runtime_fingerprint_report = out_dir / "bad-runtime-fingerprint-workload-report-bundle.json"
        bad_runtime_fingerprint_data = json.loads(report.read_text())
        bad_runtime_fingerprint_data["runtime_evidence"]["input_artifact_fingerprints"]["runtime-package"] = "bad"
        bad_runtime_fingerprint_report.write_text(
            json.dumps(bad_runtime_fingerprint_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_fingerprint_audit = out_dir / "bad-runtime-fingerprint-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_fingerprint_audit),
                str(bad_runtime_fingerprint_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with malformed runtime input fingerprint unexpectedly passed audit")
        stale_runtime_fingerprint_report = out_dir / "stale-runtime-fingerprint-workload-report-bundle.json"
        stale_runtime_fingerprint_data = json.loads(report.read_text())
        stale_runtime_fingerprint_data["runtime_evidence"]["input_artifact_fingerprints"]["pnr-mapping"] = "0" * 64
        stale_runtime_fingerprint_report.write_text(
            json.dumps(stale_runtime_fingerprint_data, indent=2, sort_keys=True) + "\n"
        )
        stale_runtime_fingerprint_audit = out_dir / "stale-runtime-fingerprint-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_runtime_fingerprint_audit),
                str(stale_runtime_fingerprint_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with stale runtime input fingerprint unexpectedly passed audit")
        bad_runtime_policy_report = out_dir / "bad-runtime-policy-workload-report-bundle.json"
        bad_runtime_policy_data = json.loads(report.read_text())
        bad_runtime_policy_data["runtime_evidence"]["required_data_movement_policies"] = ["shared_coherent"]
        bad_runtime_policy_data["runtime_evidence"]["required_synchronization_policies"] = ["device_poll"]
        bad_runtime_policy_report.write_text(
            json.dumps(bad_runtime_policy_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_policy_audit = out_dir / "bad-runtime-policy-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_policy_audit),
                str(bad_runtime_policy_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime policies unexpectedly passed audit")
        bad_runtime_fallback_report = out_dir / "bad-runtime-fallback-workload-report-bundle.json"
        bad_runtime_fallback_data = json.loads(report.read_text())
        bad_runtime_fallback_data["runtime_evidence"]["fallback_decision"]["policy"] = "allow_host_fallback"
        bad_runtime_fallback_data["runtime_evidence"]["fallback_decision"][
            "target_profile_id"
        ] = "simulator::dfg_sim::optimistic_pipeline_latency_throughput_sum"
        bad_runtime_fallback_report.write_text(
            json.dumps(bad_runtime_fallback_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_fallback_audit = out_dir / "bad-runtime-fallback-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_fallback_audit),
                str(bad_runtime_fallback_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime fallback unexpectedly passed audit")
        bad_runtime_identity_report = out_dir / "bad-runtime-identity-workload-report-bundle.json"
        bad_runtime_identity_data = json.loads(report.read_text())
        bad_runtime_identity_data["runtime_evidence"]["launch_descriptor_identity"] = []
        bad_runtime_identity_report.write_text(
            json.dumps(bad_runtime_identity_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_identity_audit = out_dir / "bad-runtime-identity-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_identity_audit),
                str(bad_runtime_identity_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with malformed runtime identity unexpectedly passed audit")
        bad_runtime_diagnostics_report = out_dir / "bad-runtime-diagnostics-workload-report-bundle.json"
        bad_runtime_diagnostics_data = json.loads(report.read_text())
        bad_runtime_diagnostics_data["runtime_evidence"]["diagnostic_records"] = "runtime-package::1"
        bad_runtime_diagnostics_report.write_text(
            json.dumps(bad_runtime_diagnostics_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_diagnostics_audit = out_dir / "bad-runtime-diagnostics-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_diagnostics_audit),
                str(bad_runtime_diagnostics_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with malformed runtime diagnostics unexpectedly passed audit")
        bad_simulator_identities_report = out_dir / "bad-runtime-simulator-identities-workload-report-bundle.json"
        bad_simulator_identities_data = json.loads(report.read_text())
        bad_simulator_identities_data["runtime_evidence"]["simulator_report_identities"] = "vecsum-cgra-sim-report"
        bad_simulator_identities_report.write_text(
            json.dumps(bad_simulator_identities_data, indent=2, sort_keys=True) + "\n"
        )
        bad_simulator_identities_audit = out_dir / "bad-runtime-simulator-identities-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_simulator_identities_audit),
                str(bad_simulator_identities_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with malformed runtime simulator identities unexpectedly passed audit")
        extra_custom_policy_report = out_dir / "extra-custom-policy-workload-report-bundle.json"
        extra_custom_policy_data = json.loads(report.read_text())
        extra_custom_policy_data["runtime_evidence"][
            "custom_data_movement_policy_identity"
        ] = "runtime-policy::unexpected::vecsum"
        extra_custom_policy_report.write_text(
            json.dumps(extra_custom_policy_data, indent=2, sort_keys=True) + "\n"
        )
        extra_custom_policy_audit = out_dir / "extra-custom-policy-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(extra_custom_policy_audit),
                str(extra_custom_policy_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with custom policy for simulated runtime unexpectedly passed audit")
        reviews = audit_data.get("artifact_reviews", [])
        matching_reviews = [
            review for review in reviews
            if review.get("schema") == "workload_report_bundle"
        ]
        if len(matching_reviews) != 1:
            raise AssertionError(f"expected one report bundle review: {audit_data}")

        blocked_runtime = out_dir / "blocked-runtime-package.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--output",
                str(blocked_runtime),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package without simulator evidence unexpectedly passed")
        blocked_report = out_dir / "blocked-runtime-workload-report-bundle.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_report_bundle.sh",
                "--output",
                str(blocked_report),
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
                str(blocked_runtime),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("report bundle with blocked runtime package unexpectedly passed")
        blocked_data = json.loads(blocked_report.read_text())
        records = blocked_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "missing_simulator_report"
            and record.get("component") == "runtime_package"
            for record in records
        ):
            raise AssertionError(f"report bundle should preserve runtime diagnostic records: {blocked_data}")
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "runtime_package_failure"
            and record.get("component") == "workload_report_bundle"
            for record in records
        ):
            raise AssertionError(f"report bundle should add its own runtime failure diagnostic: {blocked_data}")

        missing_binding_runtime = out_dir / "missing-binding-runtime-package.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--data-movement-policy",
                "shared_noncoherent",
                "--output",
                str(missing_binding_runtime),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "sim-comparison-report.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package without platform binding unexpectedly passed")
        missing_binding_report = out_dir / "missing-binding-workload-report-bundle.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_report_bundle.sh",
                "--output",
                str(missing_binding_report),
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
                str(missing_binding_runtime),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("report bundle with missing runtime platform binding unexpectedly passed")
        missing_binding_data = json.loads(missing_binding_report.read_text())
        records = missing_binding_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "missing_platform_memory_binding"
            and record.get("component") == "runtime_package"
            for record in records
        ):
            raise AssertionError(f"report bundle should preserve platform binding diagnostics: {missing_binding_data}")

        named_custom_runtime = out_dir / "named-custom-runtime-package.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--target",
                "hardware",
                "--data-movement-policy",
                "custom",
                "--custom-data-movement-policy",
                "runtime-policy::dma-window::vecsum",
                "--platform-binding",
                "platform-binding::host-buffer::vecsum",
                "--output",
                str(named_custom_runtime),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("hardware runtime package without hardware backend unexpectedly passed")
        named_custom_report = out_dir / "named-custom-workload-report-bundle.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_report_bundle.sh",
                "--output",
                str(named_custom_report),
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
                str(named_custom_runtime),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("report bundle with unavailable hardware runtime unexpectedly passed")
        named_custom_data = json.loads(named_custom_report.read_text())
        named_custom_evidence = named_custom_data.get("runtime_evidence", {})
        if named_custom_evidence.get("data_movement_policy") != "custom":
            raise AssertionError(f"custom report should preserve runtime data movement policy: {named_custom_data}")
        if (
            named_custom_evidence.get("custom_data_movement_policy_identity")
            != "runtime-policy::dma-window::vecsum"
        ):
            raise AssertionError(f"custom report should preserve runtime policy identity: {named_custom_data}")
        named_custom_audit = out_dir / "named-custom-workload-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(named_custom_audit),
                str(named_custom_report),
            ],
            "named custom workload report bundle audit",
        )
        unnamed_custom_report = out_dir / "unnamed-custom-workload-report-bundle.json"
        unnamed_custom_data = json.loads(named_custom_report.read_text())
        unnamed_custom_data["runtime_evidence"].pop("custom_data_movement_policy_identity", None)
        unnamed_custom_report.write_text(json.dumps(unnamed_custom_data, indent=2, sort_keys=True) + "\n")
        unnamed_custom_audit = out_dir / "unnamed-custom-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(unnamed_custom_audit),
                str(unnamed_custom_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with unnamed custom runtime policy unexpectedly passed audit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
