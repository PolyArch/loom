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
                str(out_dir / "rtl-manifest.json"),
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
            "rtl-manifest": artifact_test_common.fingerprint(out_dir / "rtl-manifest.json"),
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
        if optional_identities.get("rtl_manifest") != "rtl-manifest":
            raise AssertionError(f"report should reference RTL manifest evidence: {data}")
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
            "host_wrapper_identity": runtime_report["host_wrapper_identity"],
            "host_interface": runtime_package_data["host_interface"],
            "runtime_handle_model": runtime_package_data["runtime_handle_model"],
            "work_package_metadata": runtime_package_data["work_package_metadata"],
            "work_package_identity": runtime_report["work_package_identity"],
            "launch_descriptor_identity": runtime_report["launch_descriptor_identity"],
            "launch_descriptor": runtime_package_data["launch_descriptor"],
            "mapping_artifact_identity": runtime_report["mapping_artifact_identity"],
            "fabric_adg_identity": runtime_report["fabric_adg_identity"],
            "target_profile_id": runtime_report["target_profile_id"],
            "target_profile": runtime_package_data["target_profile"],
            "fallback_policy": runtime_package_data["fallback_policy"],
            "launch_status": "not_run",
            "target_status": "not_run",
            "runtime_trace_identity": "",
            "profiling_record_identity": "",
            "data_movement_policy": "simulated",
            "synchronization_mode": "host_wait",
            "memory_descriptors": runtime_package_data["memory_descriptors"],
            "argument_descriptors": runtime_package_data["argument_descriptors"],
            "runtime_configuration": runtime_package_data["runtime_configuration"],
            "required_runtime_features": runtime_package_data["required_runtime_features"],
            "required_data_movement_policies": ["simulated"],
            "required_synchronization_policies": ["host_wait"],
            "simulator_report_identities": runtime_report["simulator_report_identities"],
            "output_buffer_identities": [],
            "diagnostic_records": runtime_report["diagnostic_records"],
            "report_output_configuration": runtime_package_data["report_output_configuration"],
            "input_artifact_fingerprints": runtime_package_data["input_artifact_fingerprints"],
            "fallback_decision": expected_runtime_fallback,
        }
        if data["runtime_evidence"] != expected_runtime_evidence:
            raise AssertionError(f"report should preserve runtime evidence references: {data}")

        alternate_runtime_package = out_dir / "alternate-runtime-input-runtime-package.json"
        alternate_runtime_data = json.loads((out_dir / "runtime-package.json").read_text())
        alternate_runtime_input = "test-app-fixture::vecsum::alternate"
        alternate_runtime_data["work_package_metadata"]["runtime_input_identity"] = alternate_runtime_input
        alternate_runtime_data["host_interface"]["source_provenance"] = alternate_runtime_input
        alternate_runtime_data["launch_descriptor_identity"] = (
            "launch::vecsum::vecsum__shared_reduction_adg::test-app-fixture::vecsum::alternate"
        )
        alternate_runtime_data["runtime_report"]["launch_descriptor_identity"] = (
            alternate_runtime_data["launch_descriptor_identity"]
        )
        for descriptor in alternate_runtime_data["argument_descriptors"]:
            if descriptor.get("name") == "runtime_input":
                descriptor["identity"] = alternate_runtime_input
        for descriptor in alternate_runtime_data["memory_descriptors"]:
            descriptor["runtime_input_identity"] = alternate_runtime_input
        launch_descriptor = alternate_runtime_data["launch_descriptor"]
        launch_descriptor["descriptor_id"] = alternate_runtime_data["launch_descriptor_identity"]
        for descriptor in launch_descriptor["argument_descriptors"]:
            if descriptor.get("name") == "runtime_input":
                descriptor["identity"] = alternate_runtime_input
        for descriptor in launch_descriptor["memory_descriptors"]:
            descriptor["runtime_input_identity"] = alternate_runtime_input
        dfg_path = out_dir / "vecsum-dfg-sim-report.json"
        cgra_path = out_dir / "vecsum-cgra-sim-report.json"
        comparison_path = out_dir / "sim-comparison-report.json"
        original_dfg_text = dfg_path.read_text()
        original_cgra_text = cgra_path.read_text()
        original_comparison_text = comparison_path.read_text()
        alternate_dfg_data = json.loads(original_dfg_text)
        alternate_dfg_data["runtime_input_identity"] = alternate_runtime_input
        dfg_path.write_text(json.dumps(alternate_dfg_data, indent=2, sort_keys=True) + "\n")
        alternate_cgra_data = json.loads(original_cgra_text)
        alternate_cgra_data["runtime_input_identity"] = alternate_runtime_input
        cgra_path.write_text(json.dumps(alternate_cgra_data, indent=2, sort_keys=True) + "\n")
        alternate_comparison_data = json.loads(original_comparison_text)
        alternate_comparison_data["runtime_input_identity"] = alternate_runtime_input
        comparison_path.write_text(json.dumps(alternate_comparison_data, indent=2, sort_keys=True) + "\n")
        alternate_runtime_data["input_artifact_fingerprints"][
            "vecsum-cgra-sim-report"
        ] = artifact_test_common.fingerprint(cgra_path)
        alternate_runtime_data["input_artifact_fingerprints"][
            "sim-comparison-report"
        ] = artifact_test_common.fingerprint(comparison_path)
        alternate_runtime_package.write_text(json.dumps(alternate_runtime_data, indent=2, sort_keys=True) + "\n")
        alternate_report = out_dir / "alternate-runtime-input-workload-report-bundle.json"
        try:
            artifact_test_common.require_success(
                repo,
                [
                    "bash",
                    "test/e2e/run_report_bundle.sh",
                    "--output",
                    str(alternate_report),
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
                    str(alternate_runtime_package),
                    "--artifact",
                    str(out_dir / "sim-cycle-summary.csv"),
                    "--artifact",
                    str(out_dir / "rtl-manifest.json"),
                    "--artifact",
                    str(out_dir / "rtl-fpa-summary.csv"),
                    "--artifact",
                    str(out_dir / "dse-candidate-summary.csv"),
                ],
                "workload report bundle with alternate runtime input",
            )
            alternate_data = json.loads(alternate_report.read_text())
            if alternate_data.get("runtime_input_identity") != alternate_runtime_input:
                raise AssertionError(f"report should use runtime package input identity: {alternate_data}")
            artifact_test_common.require_success(
                repo,
                [
                    "python3",
                    "test/e2e/audit_intermediate_artifacts.py",
                    "--output",
                    str(out_dir / "alternate-runtime-input-workload-report-bundle-audit.json"),
                    str(alternate_report),
                ],
                "alternate runtime input workload report audit",
            )
        finally:
            dfg_path.write_text(original_dfg_text)
            cgra_path.write_text(original_cgra_text)
            comparison_path.write_text(original_comparison_text)

        unrelated_runtime_package = out_dir / "unrelated-runtime-package.json"
        unrelated_runtime_data = json.loads((out_dir / "runtime-package.json").read_text())
        unrelated_runtime_data["package_id"] = "runtime-package::other_workload::other_mapping"
        unrelated_runtime_data["workload"] = "other_workload"
        unrelated_runtime_data["work_package_identity"] = "work-package::other_workload::other_mapping"
        unrelated_runtime_data["selected_mapping_artifact_identity"] = "other-mapping"
        unrelated_runtime_data["fabric_adg_identity"] = "other_hardware"
        unrelated_runtime_data["runtime_report"]["work_package_identity"] = unrelated_runtime_data[
            "work_package_identity"
        ]
        unrelated_runtime_data["runtime_report"]["mapping_artifact_identity"] = "other-mapping"
        unrelated_runtime_data["runtime_report"]["fabric_adg_identity"] = "other_hardware"
        unrelated_runtime_data["work_package_metadata"]["workload"] = "other_workload"
        unrelated_runtime_data["work_package_metadata"]["work_package_identity"] = unrelated_runtime_data[
            "work_package_identity"
        ]
        unrelated_runtime_data["work_package_metadata"]["selected_mapping_artifact_identity"] = "other-mapping"
        unrelated_runtime_data["work_package_metadata"]["fabric_adg_identity"] = "other_hardware"
        unrelated_runtime_package.write_text(json.dumps(unrelated_runtime_data, indent=2, sort_keys=True) + "\n")
        filtered_runtime_report = out_dir / "filtered-runtime-workload-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_report_bundle.sh",
                "--output",
                str(filtered_runtime_report),
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
                str(unrelated_runtime_package),
                "--artifact",
                str(out_dir / "runtime-package.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-manifest.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
            ],
            "workload report bundle with unrelated runtime package",
        )
        filtered_runtime_data = json.loads(filtered_runtime_report.read_text())
        filtered_runtime_identities = filtered_runtime_data.get("optional_artifact_identities", {})
        if filtered_runtime_identities.get("runtime_package") != "runtime-package":
            raise AssertionError(f"workload report should ignore unrelated runtime package: {filtered_runtime_data}")
        if sorted(filtered_runtime_data["input_artifact_fingerprints"]) != sorted(expected_input_fingerprints):
            raise AssertionError(
                f"workload report should fingerprint only selected runtime inputs: {filtered_runtime_data}"
            )
        if filtered_runtime_data["runtime_evidence"].get("runtime_package_identity") != "runtime-package":
            raise AssertionError(
                f"workload report should summarize only selected runtime evidence: {filtered_runtime_data}"
            )

        unrelated_cgra_report = out_dir / "unrelated-cgra-sim-report.json"
        unrelated_cgra_data = json.loads((out_dir / "vecsum-cgra-sim-report.json").read_text())
        unrelated_cgra_data["workload"] = "other_workload"
        unrelated_cgra_data["hardware"] = "other_hardware"
        unrelated_cgra_data["mapping_id"] = "other_mapping"
        unrelated_cgra_data["hardware_aware_cycles"] = 999999
        unrelated_cgra_report.write_text(json.dumps(unrelated_cgra_data, indent=2, sort_keys=True) + "\n")
        filtered_cgra_report = out_dir / "filtered-cgra-workload-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_report_bundle.sh",
                "--output",
                str(filtered_cgra_report),
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
                str(unrelated_cgra_report),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "sim-comparison-report.json"),
                "--artifact",
                str(out_dir / "runtime-package.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-manifest.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
            ],
            "workload report bundle with unrelated CGRA report",
        )
        filtered_cgra_data = json.loads(filtered_cgra_report.read_text())
        filtered_cgra_identities = filtered_cgra_data.get("optional_artifact_identities", {})
        if filtered_cgra_identities.get("cgra_sim_report") != "vecsum-cgra-sim-report":
            raise AssertionError(f"workload report should ignore unrelated CGRA report: {filtered_cgra_data}")
        filtered_metrics = metric_by_id(filtered_cgra_data.get("metric_records", []))
        filtered_cgra_metric = filtered_metrics.get("metric::vecsum::cgra_sim_cycles")
        if filtered_cgra_metric is None or filtered_cgra_metric.get("value") != 589:
            raise AssertionError(f"workload report should use selected CGRA cycles: {filtered_cgra_data}")
        if "unrelated-cgra-sim-report" in filtered_cgra_data["input_artifact_fingerprints"]:
            raise AssertionError(f"workload report should not fingerprint unrelated CGRA report: {filtered_cgra_data}")

        unrelated_comparison_report = out_dir / "unrelated-sim-comparison-report.json"
        unrelated_comparison_data = json.loads((out_dir / "sim-comparison-report.json").read_text())
        unrelated_comparison_data["workload"] = "other_workload"
        unrelated_comparison_data["mapping_artifact_identity"] = "other-mapping"
        unrelated_comparison_data["cgra_sim_report_identity"] = "unrelated-cgra-sim-report"
        unrelated_comparison_data["cgra_sim_cycles"] = 999999
        unrelated_comparison_report.write_text(json.dumps(unrelated_comparison_data, indent=2, sort_keys=True) + "\n")
        filtered_comparison_report = out_dir / "filtered-comparison-workload-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_report_bundle.sh",
                "--output",
                str(filtered_comparison_report),
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
                str(unrelated_comparison_report),
                "--artifact",
                str(out_dir / "sim-comparison-report.json"),
                "--artifact",
                str(out_dir / "runtime-package.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-manifest.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
            ],
            "workload report bundle with unrelated comparison report",
        )
        filtered_comparison_data = json.loads(filtered_comparison_report.read_text())
        filtered_comparison_identities = filtered_comparison_data.get("optional_artifact_identities", {})
        if filtered_comparison_identities.get("simulation_comparison_report") != "sim-comparison-report":
            raise AssertionError(
                f"workload report should ignore unrelated comparison report: {filtered_comparison_data}"
            )
        if "unrelated-sim-comparison-report" in filtered_comparison_data["input_artifact_fingerprints"]:
            raise AssertionError(
                f"workload report should not fingerprint unrelated comparison report: {filtered_comparison_data}"
            )

        metrics = data.get("metric_records", [])
        if not isinstance(metrics, list) or not metrics:
            raise AssertionError(f"report should include metric records: {data}")
        metrics_by_id = metric_by_id(metrics)
        expected_metrics = {
            "metric::vecsum::dfg_sim_cycles": ("optimistic_steps", 579, "cycles", "dfg_software"),
            "metric::vecsum::workload_size_items": ("workload_size", 64, "items", "dfg_software"),
            "metric::vecsum::cgra_sim_cycles": ("hardware_cycles", 589, "cycles", "cgra_mapped"),
            "metric::shared_reduction_adg::frequency_mhz": ("frequency", 250.0, "MHz", "analytic"),
            "metric::shared_reduction_adg::area_um2": ("area", 7250.0, "um2", "analytic"),
            "metric::shared_reduction_adg::dynamic_power_mw": (
                "dynamic_power",
                6.0,
                "mW",
                "analytic",
            ),
            "metric::shared_reduction_adg::leakage_power_mw": (
                "leakage_power",
                0.825,
                "mW",
                "analytic",
            ),
            "metric::vecsum::estimated_runtime_us": ("estimated_runtime", 2.356, "us", "analytic"),
            "metric::vecsum::energy_nj": ("energy", 16.08, "nJ", "analytic"),
            "metric::vecsum::throughput_items_per_s": (
                "throughput",
                27164685.908,
                "items_per_s",
                "analytic",
            ),
            "metric::vecsum::performance_per_watt": (
                "performance_per_watt",
                3980173759.46,
                "items_per_s_per_w",
                "analytic",
            ),
            "metric::vecsum::performance_per_area": (
                "performance_per_area",
                3746.853,
                "items_per_s_per_um2",
                "analytic",
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

        energy = metrics_by_id["metric::vecsum::energy_nj"]
        inputs = set(energy.get("input_metric_ids", []))
        required_inputs = {
            "metric::vecsum::estimated_runtime_us",
            "metric::shared_reduction_adg::dynamic_power_mw",
            "metric::shared_reduction_adg::leakage_power_mw",
        }
        if inputs != required_inputs:
            raise AssertionError(f"energy metric should preserve input metric ids: {energy}")
        if energy.get("derivation_kind") != "runtime_power_energy":
            raise AssertionError(f"energy metric should identify runtime and power derivation: {energy}")
        runtime = metrics_by_id["metric::vecsum::estimated_runtime_us"]
        runtime_inputs = set(runtime.get("input_metric_ids", []))
        required_runtime_inputs = {
            "metric::vecsum::cgra_sim_cycles",
            "metric::shared_reduction_adg::frequency_mhz",
        }
        if runtime_inputs != required_runtime_inputs:
            raise AssertionError(f"runtime metric should preserve input metric ids: {runtime}")
        throughput = metrics_by_id["metric::vecsum::throughput_items_per_s"]
        throughput_inputs = set(throughput.get("input_metric_ids", []))
        required_throughput_inputs = {
            "metric::vecsum::workload_size_items",
            "metric::vecsum::estimated_runtime_us",
        }
        if throughput_inputs != required_throughput_inputs:
            raise AssertionError(f"throughput metric should preserve input metric ids: {throughput}")
        performance_per_watt = metrics_by_id["metric::vecsum::performance_per_watt"]
        performance_inputs = set(performance_per_watt.get("input_metric_ids", []))
        required_performance_inputs = {
            "metric::vecsum::throughput_items_per_s",
            "metric::shared_reduction_adg::dynamic_power_mw",
            "metric::shared_reduction_adg::leakage_power_mw",
        }
        if performance_inputs != required_performance_inputs:
            raise AssertionError(f"performance per watt should preserve input metric ids: {performance_per_watt}")
        performance_per_area = metrics_by_id["metric::vecsum::performance_per_area"]
        area_performance_inputs = set(performance_per_area.get("input_metric_ids", []))
        required_area_performance_inputs = {
            "metric::vecsum::throughput_items_per_s",
            "metric::shared_reduction_adg::area_um2",
        }
        if area_performance_inputs != required_area_performance_inputs:
            raise AssertionError(f"performance per area should preserve input metric ids: {performance_per_area}")

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
        alternate_comparison_runtime_input = "test-app-fixture::vecsum::alternate"
        alternate_dfg_report = out_dir / "alternate-runtime-input-dfg-sim-report.json"
        alternate_dfg_data = json.loads((out_dir / "vecsum-dfg-sim-report.json").read_text())
        alternate_dfg_data["runtime_input_identity"] = alternate_comparison_runtime_input
        alternate_dfg_report.write_text(json.dumps(alternate_dfg_data, indent=2, sort_keys=True) + "\n")
        alternate_cgra_report = out_dir / "alternate-runtime-input-cgra-sim-report.json"
        alternate_cgra_data = json.loads((out_dir / "vecsum-cgra-sim-report.json").read_text())
        alternate_cgra_data["runtime_input_identity"] = alternate_comparison_runtime_input
        alternate_cgra_report.write_text(json.dumps(alternate_cgra_data, indent=2, sort_keys=True) + "\n")
        alternate_comparison = out_dir / "alternate-runtime-input-sim-comparison-report.json"
        alternate_comparison_data = json.loads((out_dir / "sim-comparison-report.json").read_text())
        alternate_comparison_data["comparison_id"] = (
            "sim-comparison::vecsum::alternate-runtime-input-cgra-sim-report"
        )
        alternate_comparison_data["runtime_input_identity"] = alternate_comparison_runtime_input
        alternate_comparison_data["dfg_sim_report_identity"] = "alternate-runtime-input-dfg-sim-report"
        alternate_comparison_data["cgra_sim_report_identity"] = "alternate-runtime-input-cgra-sim-report"
        alternate_comparison.write_text(json.dumps(alternate_comparison_data, indent=2, sort_keys=True) + "\n")
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "alternate-runtime-input-sim-comparison-report-audit.json"),
                str(alternate_comparison),
            ],
            "alternate runtime input comparison audit",
        )
        mismatched_comparison_reference = out_dir / "mismatched-comparison-reference-workload-report-bundle.json"
        mismatched_comparison_reference_data = json.loads(report.read_text())
        mismatched_comparison_reference_data["optional_artifact_identities"][
            "simulation_comparison_report"
        ] = "alternate-runtime-input-sim-comparison-report"
        mismatched_comparison_reference_data["input_artifact_fingerprints"].pop("sim-comparison-report", None)
        mismatched_comparison_reference_data["input_artifact_fingerprints"][
            "alternate-runtime-input-sim-comparison-report"
        ] = artifact_test_common.fingerprint(alternate_comparison)
        mismatched_comparison_reference.write_text(
            json.dumps(mismatched_comparison_reference_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_comparison_reference_audit = (
            out_dir / "mismatched-comparison-reference-workload-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_comparison_reference_audit),
                str(mismatched_comparison_reference),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with unrelated comparison reference unexpectedly passed audit")
        missing_runtime_input_report = out_dir / "missing-runtime-input-workload-report-bundle.json"
        missing_runtime_input_data = json.loads(report.read_text())
        for metric in missing_runtime_input_data["metric_records"]:
            if metric.get("metric_id") == "metric::vecsum::estimated_runtime_us":
                metric["input_metric_ids"] = [
                    metric_id
                    for metric_id in metric["input_metric_ids"]
                    if metric_id != "metric::shared_reduction_adg::frequency_mhz"
                ]
        missing_runtime_input_report.write_text(
            json.dumps(missing_runtime_input_data, indent=2, sort_keys=True) + "\n"
        )
        missing_runtime_input_audit = out_dir / "missing-runtime-input-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_runtime_input_audit),
                str(missing_runtime_input_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with incomplete runtime metric inputs unexpectedly passed audit")
        missing_energy_input_report = out_dir / "missing-energy-input-workload-report-bundle.json"
        missing_energy_input_data = json.loads(report.read_text())
        for metric in missing_energy_input_data["metric_records"]:
            if metric.get("metric_id") == "metric::vecsum::energy_nj":
                metric["input_metric_ids"] = [
                    metric_id
                    for metric_id in metric["input_metric_ids"]
                    if metric_id != "metric::shared_reduction_adg::dynamic_power_mw"
                ]
        missing_energy_input_report.write_text(
            json.dumps(missing_energy_input_data, indent=2, sort_keys=True) + "\n"
        )
        missing_energy_input_audit = out_dir / "missing-energy-input-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_energy_input_audit),
                str(missing_energy_input_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with incomplete energy metric inputs unexpectedly passed audit")
        stale_energy_derivation_report = out_dir / "stale-energy-derivation-workload-report-bundle.json"
        stale_energy_derivation_data = json.loads(report.read_text())
        for metric in stale_energy_derivation_data["metric_records"]:
            if metric.get("metric_id") == "metric::vecsum::energy_nj":
                metric["derivation_kind"] = "cycle_frequency_power_area"
        stale_energy_derivation_report.write_text(
            json.dumps(stale_energy_derivation_data, indent=2, sort_keys=True) + "\n"
        )
        stale_energy_derivation_audit = out_dir / "stale-energy-derivation-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_energy_derivation_audit),
                str(stale_energy_derivation_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with stale energy derivation unexpectedly passed audit")
        missing_throughput_input_report = out_dir / "missing-throughput-input-workload-report-bundle.json"
        missing_throughput_input_data = json.loads(report.read_text())
        for metric in missing_throughput_input_data["metric_records"]:
            if metric.get("metric_id") == "metric::vecsum::throughput_items_per_s":
                metric["input_metric_ids"] = [
                    metric_id
                    for metric_id in metric["input_metric_ids"]
                    if metric_id != "metric::vecsum::estimated_runtime_us"
                ]
        missing_throughput_input_report.write_text(
            json.dumps(missing_throughput_input_data, indent=2, sort_keys=True) + "\n"
        )
        missing_throughput_input_audit = out_dir / "missing-throughput-input-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_throughput_input_audit),
                str(missing_throughput_input_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with incomplete throughput metric inputs unexpectedly passed audit")
        missing_performance_input_report = out_dir / "missing-performance-input-workload-report-bundle.json"
        missing_performance_input_data = json.loads(report.read_text())
        for metric in missing_performance_input_data["metric_records"]:
            if metric.get("metric_id") == "metric::vecsum::performance_per_watt":
                metric["input_metric_ids"] = [
                    metric_id
                    for metric_id in metric["input_metric_ids"]
                    if metric_id != "metric::vecsum::throughput_items_per_s"
                ]
        missing_performance_input_report.write_text(
            json.dumps(missing_performance_input_data, indent=2, sort_keys=True) + "\n"
        )
        missing_performance_input_audit = out_dir / "missing-performance-input-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_performance_input_audit),
                str(missing_performance_input_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with incomplete performance metric inputs unexpectedly passed audit")
        missing_area_performance_input_report = out_dir / "missing-area-performance-input-workload-report-bundle.json"
        missing_area_performance_input_data = json.loads(report.read_text())
        for metric in missing_area_performance_input_data["metric_records"]:
            if metric.get("metric_id") == "metric::vecsum::performance_per_area":
                metric["input_metric_ids"] = [
                    metric_id
                    for metric_id in metric["input_metric_ids"]
                    if metric_id != "metric::shared_reduction_adg::area_um2"
                ]
        missing_area_performance_input_report.write_text(
            json.dumps(missing_area_performance_input_data, indent=2, sort_keys=True) + "\n"
        )
        missing_area_performance_input_audit = (
            out_dir / "missing-area-performance-input-workload-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_area_performance_input_audit),
                str(missing_area_performance_input_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with incomplete area performance inputs unexpectedly passed audit")
        mismatched_energy_value_report = out_dir / "mismatched-energy-value-workload-report-bundle.json"
        mismatched_energy_value_data = json.loads(report.read_text())
        for metric in mismatched_energy_value_data["metric_records"]:
            if metric.get("metric_id") == "metric::vecsum::energy_nj":
                metric["value"] = 1.0
        mismatched_energy_value_report.write_text(
            json.dumps(mismatched_energy_value_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_energy_value_audit = out_dir / "mismatched-energy-value-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_energy_value_audit),
                str(mismatched_energy_value_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched energy metric value unexpectedly passed audit")
        missing_metric_source_report = out_dir / "missing-metric-source-workload-report-bundle.json"
        missing_metric_source_data = json.loads(report.read_text())
        for metric in missing_metric_source_data["metric_records"]:
            if metric.get("metric_id") == "metric::vecsum::energy_nj":
                metric["evidence_source_artifact_id"] = "missing-metric-source"
        missing_metric_source_report.write_text(
            json.dumps(missing_metric_source_data, indent=2, sort_keys=True) + "\n"
        )
        missing_metric_source_audit = out_dir / "missing-metric-source-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_metric_source_audit),
                str(missing_metric_source_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with missing metric source unexpectedly passed audit")
        mismatched_metric_unit_report = out_dir / "mismatched-metric-unit-workload-report-bundle.json"
        mismatched_metric_unit_data = json.loads(report.read_text())
        for metric in mismatched_metric_unit_data["metric_records"]:
            if metric.get("metric_id") == "metric::vecsum::energy_nj":
                metric["unit"] = "cycles"
        mismatched_metric_unit_report.write_text(
            json.dumps(mismatched_metric_unit_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_metric_unit_audit = out_dir / "mismatched-metric-unit-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_metric_unit_audit),
                str(mismatched_metric_unit_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched metric unit unexpectedly passed audit")

        mismatched_comparison = out_dir / "mismatched-comparison-report.json"
        mismatched_comparison_data = json.loads((out_dir / "sim-comparison-report.json").read_text())
        mismatched_comparison_data["cgra_sim_report_identity"] = "other-cgra-sim-report"
        mismatched_comparison.write_text(json.dumps(mismatched_comparison_data, indent=2, sort_keys=True) + "\n")
        mismatched_comparison_report = out_dir / "mismatched-comparison-workload-report-bundle.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_report_bundle.sh",
                "--output",
                str(mismatched_comparison_report),
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
                str(mismatched_comparison),
                "--artifact",
                str(out_dir / "runtime-package.json"),
                "--artifact",
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-manifest.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched comparison evidence unexpectedly passed")
        mismatched_comparison_data = json.loads(mismatched_comparison_report.read_text())
        records = mismatched_comparison_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "simulation_comparison_failure"
            and "CGRA-sim report identity" in record.get("message", "")
            for record in records
        ):
            raise AssertionError(
                f"workload report should diagnose mismatched comparison evidence: {mismatched_comparison_data}"
            )
        mismatched_comparison_audit = out_dir / "mismatched-comparison-workload-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_comparison_audit),
                str(mismatched_comparison_report),
            ],
            "mismatched comparison workload report audit",
        )

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
        bad_host_wrapper_report = out_dir / "bad-host-wrapper-workload-report-bundle.json"
        bad_host_wrapper_data = json.loads(report.read_text())
        bad_host_wrapper_data["runtime_host_interface"][
            "host_wrapper_identity"
        ] = "runtime-wrapper::other::vecsum"
        bad_host_wrapper_report.write_text(json.dumps(bad_host_wrapper_data, indent=2, sort_keys=True) + "\n")
        bad_host_wrapper_audit = out_dir / "bad-host-wrapper-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_host_wrapper_audit),
                str(bad_host_wrapper_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime wrapper unexpectedly passed audit")
        bad_source_provenance_report = out_dir / "bad-source-provenance-workload-report-bundle.json"
        bad_source_provenance_data = json.loads(report.read_text())
        bad_source_provenance_data["runtime_host_interface"][
            "source_provenance"
        ] = "test-app-fixture::other::default"
        bad_source_provenance_report.write_text(
            json.dumps(bad_source_provenance_data, indent=2, sort_keys=True) + "\n"
        )
        bad_source_provenance_audit = out_dir / "bad-source-provenance-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_source_provenance_audit),
                str(bad_source_provenance_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime source provenance unexpectedly passed audit")
        bad_top_level_host_abi_report = out_dir / "bad-top-level-host-abi-workload-report-bundle.json"
        bad_top_level_host_abi_data = json.loads(report.read_text())
        bad_top_level_host_abi_data["runtime_host_interface"]["invocation_abi"] = "other_runtime_abi"
        bad_top_level_host_abi_report.write_text(
            json.dumps(bad_top_level_host_abi_data, indent=2, sort_keys=True) + "\n"
        )
        bad_top_level_host_abi_audit = out_dir / "bad-top-level-host-abi-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_top_level_host_abi_audit),
                str(bad_top_level_host_abi_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched top-level runtime host ABI unexpectedly passed audit")
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
        stale_rtl_manifest_fingerprint = out_dir / "stale-rtl-manifest-workload-report-bundle.json"
        stale_rtl_manifest_data = json.loads(report.read_text())
        stale_rtl_manifest_data["input_artifact_fingerprints"]["rtl-manifest"] = "0" * 64
        stale_rtl_manifest_fingerprint.write_text(
            json.dumps(stale_rtl_manifest_data, indent=2, sort_keys=True) + "\n"
        )
        stale_rtl_manifest_audit = out_dir / "stale-rtl-manifest-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_rtl_manifest_audit),
                str(stale_rtl_manifest_fingerprint),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with stale RTL manifest fingerprint unexpectedly passed audit")
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
        missing_runtime_fingerprint_report = out_dir / "missing-runtime-fingerprint-workload-report-bundle.json"
        missing_runtime_fingerprint_data = json.loads(report.read_text())
        missing_runtime_fingerprint_data["runtime_evidence"]["input_artifact_fingerprints"].pop("pnr-mapping", None)
        missing_runtime_fingerprint_report.write_text(
            json.dumps(missing_runtime_fingerprint_data, indent=2, sort_keys=True) + "\n"
        )
        missing_runtime_fingerprint_audit = out_dir / "missing-runtime-fingerprint-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_runtime_fingerprint_audit),
                str(missing_runtime_fingerprint_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report without runtime input fingerprint unexpectedly passed audit")
        mismatched_runtime_package_report = out_dir / "mismatched-runtime-package-workload-report-bundle.json"
        mismatched_runtime_package_data = json.loads(report.read_text())
        mismatched_runtime_package_data["runtime_evidence"]["runtime_package_identity"] = "runtime-package-other"
        mismatched_runtime_package_report.write_text(
            json.dumps(mismatched_runtime_package_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_runtime_package_audit = out_dir / "mismatched-runtime-package-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_runtime_package_audit),
                str(mismatched_runtime_package_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime package identity unexpectedly passed audit")
        mismatched_runtime_hardware_report = out_dir / "mismatched-runtime-hardware-workload-report-bundle.json"
        mismatched_runtime_hardware_data = json.loads(report.read_text())
        mismatched_runtime_hardware_data["selected_hardware_candidate_identity"] = "other_hardware"
        mismatched_runtime_hardware_report.write_text(
            json.dumps(mismatched_runtime_hardware_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_runtime_hardware_audit = (
            out_dir / "mismatched-runtime-hardware-workload-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_runtime_hardware_audit),
                str(mismatched_runtime_hardware_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime hardware unexpectedly passed audit")
        missing_runtime_package_reference = out_dir / "missing-runtime-package-reference-workload-report-bundle.json"
        missing_runtime_package_reference_data = json.loads(report.read_text())
        missing_runtime_package_reference_data["optional_artifact_identities"].pop("runtime_package", None)
        missing_runtime_package_reference_data["input_artifact_fingerprints"].pop("runtime-package", None)
        missing_runtime_package_reference.write_text(
            json.dumps(missing_runtime_package_reference_data, indent=2, sort_keys=True) + "\n"
        )
        missing_runtime_package_reference_audit = (
            out_dir / "missing-runtime-package-reference-workload-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_runtime_package_reference_audit),
                str(missing_runtime_package_reference),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report without runtime package input reference unexpectedly passed audit")

        stale_runtime_package_projection = out_dir / "stale-runtime-package-projection-workload-report-bundle.json"
        stale_runtime_package = out_dir / "stale-runtime-package.json"
        stale_runtime_package_data = json.loads((out_dir / "runtime-package.json").read_text())
        stale_runtime_package_data["host_wrapper_identity"] = "runtime-wrapper::vecsum::other"
        stale_runtime_package_data["runtime_report"]["host_wrapper_identity"] = "runtime-wrapper::vecsum::other"
        stale_runtime_package.write_text(json.dumps(stale_runtime_package_data, indent=2, sort_keys=True) + "\n")
        stale_runtime_package_projection_data = json.loads(report.read_text())
        stale_runtime_package_projection_data["optional_artifact_identities"]["runtime_package"] = (
            "stale-runtime-package"
        )
        stale_runtime_package_projection_data["runtime_evidence"]["runtime_package_identity"] = (
            "stale-runtime-package"
        )
        stale_runtime_package_projection_data["input_artifact_fingerprints"].pop("runtime-package", None)
        stale_runtime_package_projection_data["input_artifact_fingerprints"][
            "stale-runtime-package"
        ] = artifact_test_common.fingerprint(stale_runtime_package)
        stale_runtime_package_projection.write_text(
            json.dumps(stale_runtime_package_projection_data, indent=2, sort_keys=True) + "\n"
        )
        stale_runtime_package_projection_audit = (
            out_dir / "stale-runtime-package-projection-workload-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_runtime_package_projection_audit),
                str(stale_runtime_package_projection),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with stale runtime package projection unexpectedly passed audit")

        mismatched_runtime_input_report = out_dir / "mismatched-runtime-input-workload-report-bundle.json"
        mismatched_runtime_input_data = json.loads(report.read_text())
        mismatched_runtime_input_identity = "test-app-fixture::other::default"
        mismatched_runtime_input_data["runtime_evidence"]["work_package_metadata"][
            "runtime_input_identity"
        ] = mismatched_runtime_input_identity
        mismatched_runtime_input_data["runtime_evidence"]["host_interface"][
            "source_provenance"
        ] = mismatched_runtime_input_identity
        mismatched_runtime_input_report.write_text(
            json.dumps(mismatched_runtime_input_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_runtime_input_audit = out_dir / "mismatched-runtime-input-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_runtime_input_audit),
                str(mismatched_runtime_input_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime input identity unexpectedly passed audit")
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
        bad_top_level_runtime_fallback_report = (
            out_dir / "bad-top-level-runtime-fallback-workload-report-bundle.json"
        )
        bad_top_level_runtime_fallback_data = json.loads(report.read_text())
        bad_top_level_runtime_fallback_data["runtime_fallback_decision"]["decision"] = "none"
        bad_top_level_runtime_fallback_report.write_text(
            json.dumps(bad_top_level_runtime_fallback_data, indent=2, sort_keys=True) + "\n"
        )
        bad_top_level_runtime_fallback_audit = (
            out_dir / "bad-top-level-runtime-fallback-workload-report-bundle-audit.json"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_top_level_runtime_fallback_audit),
                str(bad_top_level_runtime_fallback_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched top-level runtime fallback unexpectedly passed audit")
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
        bad_runtime_launch_report = out_dir / "bad-runtime-launch-workload-report-bundle.json"
        bad_runtime_launch_data = json.loads(report.read_text())
        bad_runtime_launch_data["runtime_evidence"]["launch_descriptor"]["descriptor_id"] = "launch::other"
        bad_runtime_launch_report.write_text(
            json.dumps(bad_runtime_launch_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_launch_audit = out_dir / "bad-runtime-launch-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_launch_audit),
                str(bad_runtime_launch_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime launch descriptor unexpectedly passed audit")
        bad_runtime_scalar_report = out_dir / "bad-runtime-scalar-workload-report-bundle.json"
        bad_runtime_scalar_data = json.loads(report.read_text())
        bad_runtime_scalar_data["runtime_evidence"]["launch_descriptor"]["scalar_value_descriptors"] = "scalar"
        bad_runtime_scalar_report.write_text(
            json.dumps(bad_runtime_scalar_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_scalar_audit = out_dir / "bad-runtime-scalar-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_scalar_audit),
                str(bad_runtime_scalar_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with malformed runtime scalar descriptors unexpectedly passed audit")
        bad_runtime_wrapper_report = out_dir / "bad-runtime-wrapper-workload-report-bundle.json"
        bad_runtime_wrapper_data = json.loads(report.read_text())
        bad_runtime_wrapper_data["runtime_evidence"]["host_wrapper_identity"] = []
        bad_runtime_wrapper_report.write_text(
            json.dumps(bad_runtime_wrapper_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_wrapper_audit = out_dir / "bad-runtime-wrapper-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_wrapper_audit),
                str(bad_runtime_wrapper_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with malformed runtime wrapper unexpectedly passed audit")
        bad_runtime_host_report = out_dir / "bad-runtime-host-workload-report-bundle.json"
        bad_runtime_host_data = json.loads(report.read_text())
        bad_runtime_host_data["runtime_evidence"]["host_interface"]["invocation_abi"] = ""
        bad_runtime_host_report.write_text(
            json.dumps(bad_runtime_host_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_host_audit = out_dir / "bad-runtime-host-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_host_audit),
                str(bad_runtime_host_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with malformed runtime host interface unexpectedly passed audit")
        bad_runtime_host_source_report = out_dir / "bad-runtime-host-source-workload-report-bundle.json"
        bad_runtime_host_source_data = json.loads(report.read_text())
        bad_runtime_host_source_data["runtime_evidence"]["host_interface"][
            "source_provenance"
        ] = "test-app-fixture::other::default"
        bad_runtime_host_source_report.write_text(
            json.dumps(bad_runtime_host_source_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_host_source_audit = out_dir / "bad-runtime-host-source-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_host_source_audit),
                str(bad_runtime_host_source_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime host source unexpectedly passed audit")
        bad_runtime_handle_report = out_dir / "bad-runtime-handle-workload-report-bundle.json"
        bad_runtime_handle_data = json.loads(report.read_text())
        bad_runtime_handle_data["runtime_evidence"]["runtime_handle_model"]["ir_token_kind"] = "dataflow_thread_token"
        bad_runtime_handle_report.write_text(json.dumps(bad_runtime_handle_data, indent=2, sort_keys=True) + "\n")
        bad_runtime_handle_audit = out_dir / "bad-runtime-handle-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_handle_audit),
                str(bad_runtime_handle_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with dataflow-backed runtime handle unexpectedly passed audit")
        bad_runtime_work_package_report = out_dir / "bad-runtime-work-package-workload-report-bundle.json"
        bad_runtime_work_package_data = json.loads(report.read_text())
        bad_runtime_work_package_data["runtime_evidence"]["work_package_metadata"][
            "selected_mapping_artifact_identity"
        ] = "other-mapping"
        bad_runtime_work_package_report.write_text(
            json.dumps(bad_runtime_work_package_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_work_package_audit = out_dir / "bad-runtime-work-package-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_work_package_audit),
                str(bad_runtime_work_package_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime work package unexpectedly passed audit")
        bad_runtime_report_output_report = out_dir / "bad-runtime-report-output-workload-report-bundle.json"
        bad_runtime_report_output_data = json.loads(report.read_text())
        bad_runtime_report_output_data["runtime_evidence"]["report_output_configuration"][
            "runtime_report_identity"
        ] = "runtime-report::other"
        bad_runtime_report_output_report.write_text(
            json.dumps(bad_runtime_report_output_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_report_output_audit = out_dir / "bad-runtime-report-output-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_report_output_audit),
                str(bad_runtime_report_output_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime report output unexpectedly passed audit")
        bad_runtime_memory_report = out_dir / "bad-runtime-memory-workload-report-bundle.json"
        bad_runtime_memory_data = json.loads(report.read_text())
        bad_runtime_memory_data["runtime_evidence"]["memory_descriptors"][0]["host_buffer_identity"] = []
        bad_runtime_memory_report.write_text(json.dumps(bad_runtime_memory_data, indent=2, sort_keys=True) + "\n")
        bad_runtime_memory_audit = out_dir / "bad-runtime-memory-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_memory_audit),
                str(bad_runtime_memory_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with malformed runtime memory descriptor unexpectedly passed audit")
        bad_runtime_arguments_report = out_dir / "bad-runtime-arguments-workload-report-bundle.json"
        bad_runtime_arguments_data = json.loads(report.read_text())
        bad_runtime_arguments_data["runtime_evidence"]["argument_descriptors"] = "runtime_input"
        bad_runtime_arguments_report.write_text(
            json.dumps(bad_runtime_arguments_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_arguments_audit = out_dir / "bad-runtime-arguments-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_arguments_audit),
                str(bad_runtime_arguments_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with malformed runtime arguments unexpectedly passed audit")
        bad_runtime_target_report = out_dir / "bad-runtime-target-workload-report-bundle.json"
        bad_runtime_target_data = json.loads(report.read_text())
        bad_runtime_target_data["runtime_evidence"]["target_profile"]["profile_id"] = "simulator::other"
        bad_runtime_target_report.write_text(json.dumps(bad_runtime_target_data, indent=2, sort_keys=True) + "\n")
        bad_runtime_target_audit = out_dir / "bad-runtime-target-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_target_audit),
                str(bad_runtime_target_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime target profile unexpectedly passed audit")
        bad_runtime_configuration_report = out_dir / "bad-runtime-configuration-workload-report-bundle.json"
        bad_runtime_configuration_data = json.loads(report.read_text())
        bad_runtime_configuration_data["runtime_evidence"]["runtime_configuration"][
            "synchronization_mode"
        ] = "device_poll"
        bad_runtime_configuration_report.write_text(
            json.dumps(bad_runtime_configuration_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_configuration_audit = out_dir / "bad-runtime-configuration-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_configuration_audit),
                str(bad_runtime_configuration_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with mismatched runtime configuration unexpectedly passed audit")
        bad_runtime_features_report = out_dir / "bad-runtime-features-workload-report-bundle.json"
        bad_runtime_features_data = json.loads(report.read_text())
        bad_runtime_features_data["runtime_evidence"]["required_runtime_features"] = ["simulator_dispatch", ""]
        bad_runtime_features_report.write_text(
            json.dumps(bad_runtime_features_data, indent=2, sort_keys=True) + "\n"
        )
        bad_runtime_features_audit = out_dir / "bad-runtime-features-workload-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_runtime_features_audit),
                str(bad_runtime_features_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("workload report with malformed runtime feature requirements unexpectedly passed audit")
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

        missing_runtime_report = out_dir / "missing-runtime-workload-report-bundle.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_report_bundle.sh",
                "--output",
                str(missing_runtime_report),
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
                str(out_dir / "sim-cycle-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
                "--artifact",
                str(out_dir / "dse-candidate-summary.csv"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("report bundle without runtime package unexpectedly passed")
        missing_runtime_data = json.loads(missing_runtime_report.read_text())
        missing_runtime_records = missing_runtime_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "missing_runtime_package"
            and record.get("component") == "workload_report_bundle"
            for record in missing_runtime_records
        ):
            raise AssertionError(f"missing runtime package should be diagnosed: {missing_runtime_data}")

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
