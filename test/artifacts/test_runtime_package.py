#!/usr/bin/env python3
"""Regression test for runtime launch package artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "package_id",
    "workload",
    "work_package_identity",
    "launch_descriptor_identity",
    "host_program_identity",
    "host_wrapper_identity",
    "host_interface",
    "launch_descriptor",
    "runtime_handle_model",
    "selected_mapping_artifact_identity",
    "fabric_adg_identity",
    "target_profile",
    "runtime_configuration",
    "input_artifact_fingerprints",
    "runtime_report",
    "fallback_policy",
    "fallback_decision",
    "synchronization_mode",
    "data_movement_policy",
    "memory_descriptors",
    "argument_descriptors",
    "required_runtime_features",
    "required_data_movement_policies",
    "required_synchronization_policies",
    "simulator_report_identities",
    "diagnostic_records",
    "diagnostics",
    "status",
}

def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-runtime-package-") as tmp:
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

        package = out_dir / "runtime-package.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--output",
                str(package),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "sim-comparison-report.json"),
            ],
            "runtime package",
        )

        data = json.loads(package.read_text())
        missing = REQUIRED_KEYS - set(data)
        if missing:
            raise AssertionError(f"runtime package missing keys: {sorted(missing)}")
        if data["kind"] != "runtime_package":
            raise AssertionError(f"unexpected runtime package kind: {data}")
        if data["status"] != "pass":
            raise AssertionError(f"runtime package should pass with mapping and simulator evidence: {data}")
        if data["workload"] != "vecsum":
            raise AssertionError(f"unexpected runtime package workload: {data}")
        if data["package_id"] != "runtime-package::vecsum::vecsum__shared_reduction_adg":
            raise AssertionError(f"unexpected runtime package identity: {data}")
        if data["work_package_identity"] != "work-package::vecsum::vecsum__shared_reduction_adg":
            raise AssertionError(f"unexpected work package identity: {data}")
        if data["host_program_identity"] != "test-app-host::vecsum::default":
            raise AssertionError(f"unexpected host program identity: {data}")
        if data["host_wrapper_identity"] != "runtime-wrapper::vecsum::vecsum__shared_reduction_adg":
            raise AssertionError(f"unexpected host wrapper identity: {data}")
        expected_host_interface = {
            "host_program_identity": "test-app-host::vecsum::default",
            "host_wrapper_identity": "runtime-wrapper::vecsum::vecsum__shared_reduction_adg",
            "invocation_abi": "loom_runtime_package_v1",
            "compatibility_mode_requires_runtime": False,
            "acceleration_mode_requires_runtime_package": True,
            "source_provenance": "test-app-fixture::vecsum::default",
        }
        if data["host_interface"] != expected_host_interface:
            raise AssertionError(f"unexpected host interface metadata: {data}")
        expected_launch = "launch::vecsum::vecsum__shared_reduction_adg::test-app-fixture::vecsum::default"
        if data["launch_descriptor_identity"] != expected_launch:
            raise AssertionError(f"unexpected launch descriptor identity: {data}")
        launch_descriptor = data["launch_descriptor"]
        if launch_descriptor.get("descriptor_id") != expected_launch:
            raise AssertionError(f"launch descriptor id should match package identity: {data}")
        if launch_descriptor.get("work_package_identity") != data["work_package_identity"]:
            raise AssertionError(f"launch descriptor missed work package identity: {data}")
        if launch_descriptor.get("selected_mapping_artifact_identity") != "pnr-mapping":
            raise AssertionError(f"launch descriptor missed mapping identity: {data}")
        if launch_descriptor.get("target_profile_id") != "simulator::cgra_sim::mapping_constraint_estimate":
            raise AssertionError(f"launch descriptor missed target profile: {data}")
        if launch_descriptor.get("memory_descriptor_logical_arguments") != ["vecsum.default_input"]:
            raise AssertionError(f"launch descriptor missed memory descriptor bindings: {data}")
        if launch_descriptor.get("argument_descriptor_names") != ["runtime_input", "mapping_artifact"]:
            raise AssertionError(f"launch descriptor missed argument descriptors: {data}")
        if launch_descriptor.get("scalar_value_descriptors") != []:
            raise AssertionError(f"launch descriptor should expose scalar value descriptors: {data}")
        if launch_descriptor.get("fallback_policy") != "report_only":
            raise AssertionError(f"launch descriptor missed fallback policy: {data}")
        if launch_descriptor.get("synchronization_mode") != "host_wait":
            raise AssertionError(f"launch descriptor missed synchronization mode: {data}")
        handle_model = data["runtime_handle_model"]
        if handle_model.get("handle_kind") != "host_visible_launch_handle":
            raise AssertionError(f"runtime handle model should be host-visible: {data}")
        if handle_model.get("ir_token_kind") != "not_dataflow_thread_token":
            raise AssertionError(f"runtime handle must not be a dataflow token: {data}")
        if "wait_for_completion" not in handle_model.get("operations", []):
            raise AssertionError(f"runtime handle should expose wait operation: {data}")
        if data["selected_mapping_artifact_identity"] != "pnr-mapping":
            raise AssertionError(f"unexpected selected mapping identity: {data}")
        if data["fabric_adg_identity"] != "shared_reduction_adg":
            raise AssertionError(f"unexpected fabric ADG identity: {data}")
        expected_target = {
            "target_kind": "simulator",
            "simulator": "cgra_sim",
            "profile_id": "simulator::cgra_sim::mapping_constraint_estimate",
        }
        if data["target_profile"] != expected_target:
            raise AssertionError(f"unexpected target profile: {data}")
        expected_runtime_configuration = {
            "configuration_id": "runtime-config::report_only::simulated::host_wait",
            "target_profile_id": "simulator::cgra_sim::mapping_constraint_estimate",
            "data_movement_policy": "simulated",
            "platform_binding_identity": "",
            "fallback_policy": "report_only",
            "synchronization_mode": "host_wait",
        }
        if data["runtime_configuration"] != expected_runtime_configuration:
            raise AssertionError(f"unexpected runtime configuration: {data}")
        expected_input_fingerprints = {
            "pnr-mapping": artifact_test_common.fingerprint(out_dir / "pnr-mapping.json"),
            "vecsum-cgra-sim-report": artifact_test_common.fingerprint(out_dir / "vecsum-cgra-sim-report.json"),
            "sim-comparison-report": artifact_test_common.fingerprint(out_dir / "sim-comparison-report.json"),
        }
        if data["input_artifact_fingerprints"] != expected_input_fingerprints:
            raise AssertionError(f"unexpected runtime input fingerprints: {data}")
        if data["fallback_policy"] != "report_only":
            raise AssertionError(f"unexpected fallback policy: {data}")
        expected_fallback = {
            "policy": "report_only",
            "decision": "report_only",
            "fallback_taken": False,
            "target_profile_id": "simulator::cgra_sim::mapping_constraint_estimate",
            "reason": "report-only runtime package records launch metadata without executing accelerator work",
        }
        if data["fallback_decision"] != expected_fallback:
            raise AssertionError(f"unexpected fallback decision: {data}")

        fallback_features = {
            "require_acceleration": "require_acceleration_policy",
            "allow_host_fallback": "host_fallback_policy",
            "allow_scalar_fallback": "scalar_fallback_policy",
        }
        for fallback_policy, feature in fallback_features.items():
            policy_package = out_dir / f"{fallback_policy}-runtime-package.json"
            artifact_test_common.require_success(
                repo,
                [
                    "bash",
                    "test/e2e/run_runtime_package.sh",
                    "--fallback-policy",
                    fallback_policy,
                    "--output",
                    str(policy_package),
                    "--artifact",
                    str(out_dir / "pnr-mapping.json"),
                    "--artifact",
                    str(out_dir / "vecsum-cgra-sim-report.json"),
                    "--artifact",
                    str(out_dir / "sim-comparison-report.json"),
                ],
                f"runtime package with {fallback_policy}",
            )
            policy_data = json.loads(policy_package.read_text())
            if policy_data.get("fallback_policy") != fallback_policy:
                raise AssertionError(f"runtime package should preserve fallback policy: {policy_data}")
            expected_policy_fallback = {
                "policy": fallback_policy,
                "decision": "none",
                "fallback_taken": False,
                "target_profile_id": "simulator::cgra_sim::mapping_constraint_estimate",
                "reason": "selected target profile metadata is available; no fallback was selected",
            }
            if policy_data.get("fallback_decision") != expected_policy_fallback:
                raise AssertionError(f"unexpected explicit fallback decision: {policy_data}")
            expected_configuration_id = f"runtime-config::{fallback_policy}::simulated::host_wait"
            if policy_data.get("runtime_configuration", {}).get("configuration_id") != expected_configuration_id:
                raise AssertionError(f"runtime configuration should include fallback policy: {policy_data}")
            if policy_data.get("launch_descriptor", {}).get("fallback_policy") != fallback_policy:
                raise AssertionError(f"launch descriptor should include fallback policy: {policy_data}")
            runtime_report = policy_data.get("runtime_report", {})
            expected_report_id = f"runtime-report::vecsum::vecsum__shared_reduction_adg::{fallback_policy}"
            if runtime_report.get("report_id") != expected_report_id:
                raise AssertionError(f"runtime report id should include fallback policy: {policy_data}")
            if runtime_report.get("fallback_decision") != expected_policy_fallback:
                raise AssertionError(f"runtime report should preserve fallback decision: {policy_data}")
            required_features = set(policy_data.get("required_runtime_features", []))
            if required_features != {"simulator_dispatch", "explicit_mapping_artifact", feature}:
                raise AssertionError(f"runtime features should include explicit fallback policy: {policy_data}")
            policy_audit = out_dir / f"{fallback_policy}-runtime-package-audit-summary.json"
            artifact_test_common.require_success(
                repo,
                [
                    "python3",
                    "test/e2e/audit_intermediate_artifacts.py",
                    "--output",
                    str(policy_audit),
                    str(policy_package),
                ],
                f"runtime package audit with {fallback_policy}",
            )
        expected_runtime_report = {
            "report_id": "runtime-report::vecsum::vecsum__shared_reduction_adg::report_only",
            "host_program_identity": "test-app-host::vecsum::default",
            "work_package_identity": "work-package::vecsum::vecsum__shared_reduction_adg",
            "launch_descriptor_identity": expected_launch,
            "mapping_artifact_identity": "pnr-mapping",
            "fabric_adg_identity": "shared_reduction_adg",
            "target_profile_id": "simulator::cgra_sim::mapping_constraint_estimate",
            "memory_policy": "simulated",
            "synchronization_mode": "host_wait",
            "fallback_decision": expected_fallback,
            "simulator_report_identities": ["vecsum-cgra-sim-report", "sim-comparison-report"],
            "runtime_trace_identity": "",
            "profiling_record_identity": "",
            "output_buffer_identities": [],
            "launch_status": "not_run",
            "target_status": "not_run",
            "diagnostic_records": [],
        }
        if data["runtime_report"] != expected_runtime_report:
            raise AssertionError(f"unexpected runtime report: {data}")
        if data["synchronization_mode"] != "host_wait":
            raise AssertionError(f"unexpected synchronization mode: {data}")
        if data["data_movement_policy"] != "simulated":
            raise AssertionError(f"unexpected data movement policy: {data}")
        if data.get("required_data_movement_policies") != ["simulated"]:
            raise AssertionError(f"runtime package should record required data movement policies: {data}")
        if data.get("required_synchronization_policies") != ["host_wait"]:
            raise AssertionError(f"runtime package should record required synchronization policies: {data}")
        memory_descriptors = data.get("memory_descriptors", [])
        if not isinstance(memory_descriptors, list) or len(memory_descriptors) != 1:
            raise AssertionError(f"runtime package should include one memory descriptor: {data}")
        descriptor = memory_descriptors[0]
        expected_memory = {
            "logical_argument": "vecsum.default_input",
            "direction": "read_write",
            "policy": "simulated",
            "runtime_input_identity": "test-app-fixture::vecsum::default",
            "byte_size": 256,
            "element_layout": "u32[64]",
            "alignment_bytes": 4,
            "address_space": "simulator::memory_model",
            "coherence_requirement": "simulator_consistent",
            "transfer_policy": "simulated",
        }
        if descriptor != expected_memory:
            raise AssertionError(f"unexpected memory descriptor: {data}")
        arguments = data.get("argument_descriptors", [])
        argument_names = {item.get("name") for item in arguments if isinstance(item, dict)}
        if argument_names != {"runtime_input", "mapping_artifact"}:
            raise AssertionError(f"unexpected argument descriptors: {data}")
        required_features = set(data.get("required_runtime_features", []))
        if required_features != {"simulator_dispatch", "explicit_mapping_artifact", "report_only_fallback"}:
            raise AssertionError(f"unexpected runtime features: {data}")
        simulator_reports = set(data.get("simulator_report_identities", []))
        if simulator_reports != {"vecsum-cgra-sim-report", "sim-comparison-report"}:
            raise AssertionError(f"unexpected simulator report identities: {data}")

        audit = out_dir / "runtime-package-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit),
                str(package),
            ],
            "runtime package audit",
        )
        audit_data = json.loads(audit.read_text())
        if audit_data.get("verdict") != "pass":
            raise AssertionError(f"expected runtime package audit pass: {audit_data}")
        token_backed_package = out_dir / "token-backed-runtime-package.json"
        token_backed_data = json.loads(package.read_text())
        token_backed_data["runtime_handle_model"]["ir_token_kind"] = "dataflow_thread_token"
        token_backed_package.write_text(json.dumps(token_backed_data, indent=2, sort_keys=True) + "\n")
        token_backed_audit = out_dir / "token-backed-runtime-package-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(token_backed_audit),
                str(token_backed_package),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package backed by dataflow token unexpectedly passed audit")
        mismatched_config_package = out_dir / "mismatched-runtime-config-package.json"
        mismatched_config_data = json.loads(package.read_text())
        mismatched_config_data["runtime_configuration"]["data_movement_policy"] = "shared_coherent"
        mismatched_config_package.write_text(json.dumps(mismatched_config_data, indent=2, sort_keys=True) + "\n")
        mismatched_config_audit = out_dir / "mismatched-runtime-config-package-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_config_audit),
                str(mismatched_config_package),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package with mismatched runtime configuration unexpectedly passed audit")
        mismatched_required_policy_package = out_dir / "mismatched-required-policy-runtime-package.json"
        mismatched_required_policy_data = json.loads(package.read_text())
        mismatched_required_policy_data["required_data_movement_policies"] = ["shared_coherent"]
        mismatched_required_policy_data["required_synchronization_policies"] = ["device_poll"]
        mismatched_required_policy_package.write_text(
            json.dumps(mismatched_required_policy_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_required_policy_audit = out_dir / "mismatched-required-policy-runtime-package-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_required_policy_audit),
                str(mismatched_required_policy_package),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package with mismatched required policies unexpectedly passed audit")
        bad_fingerprint_package = out_dir / "bad-fingerprint-runtime-package.json"
        bad_fingerprint_data = json.loads(package.read_text())
        bad_fingerprint_data["input_artifact_fingerprints"]["pnr-mapping"] = "not-a-sha256"
        bad_fingerprint_package.write_text(json.dumps(bad_fingerprint_data, indent=2, sort_keys=True) + "\n")
        bad_fingerprint_audit = out_dir / "bad-fingerprint-runtime-package-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_fingerprint_audit),
                str(bad_fingerprint_package),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package with malformed input fingerprint unexpectedly passed audit")
        stale_fingerprint_package = out_dir / "stale-fingerprint-runtime-package.json"
        stale_fingerprint_data = json.loads(package.read_text())
        stale_fingerprint_data["input_artifact_fingerprints"]["pnr-mapping"] = "0" * 64
        stale_fingerprint_package.write_text(json.dumps(stale_fingerprint_data, indent=2, sort_keys=True) + "\n")
        stale_fingerprint_audit = out_dir / "stale-fingerprint-runtime-package-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_fingerprint_audit),
                str(stale_fingerprint_package),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package with stale input fingerprint unexpectedly passed audit")
        bad_host_interface_package = out_dir / "bad-host-interface-runtime-package.json"
        bad_host_interface_data = json.loads(package.read_text())
        bad_host_interface_data["host_interface"]["compatibility_mode_requires_runtime"] = True
        bad_host_interface_package.write_text(json.dumps(bad_host_interface_data, indent=2, sort_keys=True) + "\n")
        bad_host_interface_audit = out_dir / "bad-host-interface-runtime-package-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_host_interface_audit),
                str(bad_host_interface_package),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package requiring runtime for compatibility mode unexpectedly passed audit")
        fake_execution_package = out_dir / "fake-execution-runtime-package.json"
        fake_execution_data = json.loads(package.read_text())
        fake_execution_data["runtime_report"]["launch_status"] = "pass"
        fake_execution_package.write_text(json.dumps(fake_execution_data, indent=2, sort_keys=True) + "\n")
        fake_execution_audit = out_dir / "fake-execution-runtime-package-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(fake_execution_audit),
                str(fake_execution_package),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("report-only runtime package claiming execution unexpectedly passed audit")

        missing_cgra = out_dir / "missing-cgra-runtime-package.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--output",
                str(missing_cgra),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("CGRA-sim runtime package without simulator evidence unexpectedly passed")
        missing_cgra_data = json.loads(missing_cgra.read_text())
        if missing_cgra_data.get("status") != "blocked":
            raise AssertionError(f"missing CGRA-sim evidence should block runtime package: {missing_cgra_data}")
        if not any("CGRA-sim target requires CGRA-sim report" in str(item) for item in missing_cgra_data.get("diagnostics", [])):
            raise AssertionError(f"missing CGRA-sim evidence should be diagnosed: {missing_cgra_data}")
        missing_records = missing_cgra_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "missing_simulator_report"
            and record.get("component") == "runtime_package"
            for record in missing_records
        ):
            raise AssertionError(f"missing CGRA-sim evidence needs structured diagnostics: {missing_cgra_data}")

        missing_binding = out_dir / "missing-platform-binding-runtime-package.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--data-movement-policy",
                "shared_noncoherent",
                "--output",
                str(missing_binding),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(out_dir / "vecsum-cgra-sim-report.json"),
                "--artifact",
                str(out_dir / "sim-comparison-report.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package without platform memory binding unexpectedly passed")
        missing_binding_data = json.loads(missing_binding.read_text())
        if missing_binding_data.get("status") != "blocked":
            raise AssertionError(f"missing platform binding should block runtime package: {missing_binding_data}")
        if missing_binding_data.get("data_movement_policy") != "shared_noncoherent":
            raise AssertionError(f"runtime package should preserve requested memory policy: {missing_binding_data}")
        descriptors = missing_binding_data.get("memory_descriptors", [])
        if not descriptors or descriptors[0].get("policy") != "shared_noncoherent":
            raise AssertionError(f"memory descriptor should preserve requested memory policy: {missing_binding_data}")
        records = missing_binding_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "missing_platform_memory_binding"
            and record.get("component") == "runtime_package"
            for record in records
        ):
            raise AssertionError(f"missing platform binding needs structured diagnostics: {missing_binding_data}")
        missing_binding_audit = out_dir / "missing-platform-binding-runtime-package-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_binding_audit),
                str(missing_binding),
            ],
            "blocked runtime package audit",
        )
        mismatched_policy = out_dir / "mismatched-memory-policy-runtime-package.json"
        mismatched_policy_data = json.loads(missing_binding.read_text())
        mismatched_policy_data["memory_descriptors"][0]["policy"] = "simulated"
        mismatched_policy.write_text(json.dumps(mismatched_policy_data, indent=2, sort_keys=True) + "\n")
        mismatched_policy_audit = out_dir / "mismatched-memory-policy-runtime-package-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_policy_audit),
                str(mismatched_policy),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package with mismatched memory descriptor policy unexpectedly passed audit")
        invalid_extent = out_dir / "invalid-memory-extent-runtime-package.json"
        invalid_extent_data = json.loads(package.read_text())
        invalid_extent_data["memory_descriptors"][0]["byte_size"] = 0
        invalid_extent.write_text(json.dumps(invalid_extent_data, indent=2, sort_keys=True) + "\n")
        invalid_extent_audit = out_dir / "invalid-memory-extent-runtime-package-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(invalid_extent_audit),
                str(invalid_extent),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package with invalid memory descriptor extent unexpectedly passed audit")

        unknown_layout_report = out_dir / "toy-dfg-sim-report.json"
        unknown_layout_report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "dfg_sim_report",
                    "workload": "toy",
                    "status": "pass",
                    "runtime_input_identity": "test-app-fixture::toy::default",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        unknown_layout_package = out_dir / "unknown-layout-runtime-package.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--target",
                "dfg-sim",
                "--output",
                str(unknown_layout_package),
                "--artifact",
                str(unknown_layout_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("runtime package with unknown memory layout unexpectedly passed")
        unknown_layout_data = json.loads(unknown_layout_package.read_text())
        if unknown_layout_data.get("status") != "blocked":
            raise AssertionError(f"unknown memory layout should block runtime package: {unknown_layout_data}")
        if unknown_layout_data.get("memory_descriptors") != []:
            raise AssertionError(f"unknown memory layout must not invent memory descriptors: {unknown_layout_data}")
        if "runtime input memory layout is missing for toy" not in unknown_layout_data.get("diagnostics", []):
            raise AssertionError(f"unknown memory layout should be diagnosed: {unknown_layout_data}")

        mismatched_cgra = out_dir / "mismatch-cgra-sim-report.json"
        mismatched_cgra_data = json.loads((out_dir / "vecsum-cgra-sim-report.json").read_text())
        mismatched_cgra_data["workload"] = "other_workload"
        mismatched_cgra_data["mapping_id"] = "other_mapping"
        mismatched_cgra.write_text(json.dumps(mismatched_cgra_data, indent=2, sort_keys=True) + "\n")
        mismatched_package = out_dir / "mismatch-runtime-package.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--output",
                str(mismatched_package),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
                "--artifact",
                str(mismatched_cgra),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("CGRA-sim runtime package with mismatched report unexpectedly passed")
        mismatched_package_data = json.loads(mismatched_package.read_text())
        if mismatched_package_data.get("status") != "blocked":
            raise AssertionError(f"mismatched CGRA-sim report should block runtime package: {mismatched_package_data}")
        diagnostics = set(str(item) for item in mismatched_package_data.get("diagnostics", []))
        expected_diagnostics = {
            "CGRA-sim report workload identity mismatch",
            "CGRA-sim report mapping identity mismatch",
        }
        if not expected_diagnostics.issubset(diagnostics):
            raise AssertionError(f"mismatched CGRA-sim diagnostics are incomplete: {mismatched_package_data}")

        dfg_mapping_only = out_dir / "dfg-mapping-only-runtime-package.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--target",
                "dfg-sim",
                "--output",
                str(dfg_mapping_only),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("DFG-sim runtime package with mapping-only input unexpectedly passed")
        dfg_mapping_only_data = json.loads(dfg_mapping_only.read_text())
        if dfg_mapping_only_data.get("status") != "blocked":
            raise AssertionError(f"DFG-sim mapping-only input should block runtime package: {dfg_mapping_only_data}")
        diagnostics = [str(item) for item in dfg_mapping_only_data.get("diagnostics", [])]
        expected_diagnostics = {
            "DFG-sim target requires DFG-sim report",
            "DFG-sim target does not consume mapping artifacts",
        }
        if not expected_diagnostics.issubset(set(diagnostics)):
            raise AssertionError(f"DFG-sim mapping-only diagnostics are incomplete: {dfg_mapping_only_data}")

        dfg_package = out_dir / "dfg-runtime-package.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--target",
                "dfg-sim",
                "--output",
                str(dfg_package),
                "--artifact",
                str(out_dir / "vecsum-dfg-sim-report.json"),
            ],
            "DFG-sim runtime package",
        )
        dfg_data = json.loads(dfg_package.read_text())
        if dfg_data.get("status") != "pass" or dfg_data.get("workload") != "vecsum":
            raise AssertionError(f"DFG-sim runtime package should pass for software simulator report: {dfg_data}")
        expected_dfg_target = {
            "target_kind": "simulator",
            "simulator": "dfg_sim",
            "profile_id": "simulator::dfg_sim::optimistic_pipeline_latency_throughput_sum",
        }
        if dfg_data.get("target_profile") != expected_dfg_target:
            raise AssertionError(f"unexpected DFG-sim target profile: {dfg_data}")
        expected_dfg_runtime_configuration = {
            "configuration_id": "runtime-config::report_only::simulated::host_wait",
            "target_profile_id": "simulator::dfg_sim::optimistic_pipeline_latency_throughput_sum",
            "data_movement_policy": "simulated",
            "platform_binding_identity": "",
            "fallback_policy": "report_only",
            "synchronization_mode": "host_wait",
        }
        if dfg_data.get("runtime_configuration") != expected_dfg_runtime_configuration:
            raise AssertionError(f"unexpected DFG runtime configuration: {dfg_data}")
        if dfg_data.get("required_data_movement_policies") != ["simulated"]:
            raise AssertionError(f"DFG package should record required data movement policies: {dfg_data}")
        if dfg_data.get("required_synchronization_policies") != ["host_wait"]:
            raise AssertionError(f"DFG package should record required synchronization policies: {dfg_data}")
        dfg_launch_descriptor = dfg_data.get("launch_descriptor", {})
        if dfg_launch_descriptor.get("selected_mapping_artifact_identity") != "":
            raise AssertionError(f"DFG launch descriptor must not require mapping identity: {dfg_data}")
        if dfg_launch_descriptor.get("argument_descriptor_names") != ["runtime_input", "dfg_sim_report"]:
            raise AssertionError(f"DFG launch descriptor missed software report argument: {dfg_data}")
        dfg_handle_model = dfg_data.get("runtime_handle_model", {})
        if dfg_handle_model.get("ir_token_kind") != "not_dataflow_thread_token":
            raise AssertionError(f"DFG runtime handle must not be a dataflow token: {dfg_data}")
        expected_dfg_fallback = {
            "policy": "report_only",
            "decision": "report_only",
            "fallback_taken": False,
            "target_profile_id": "simulator::dfg_sim::optimistic_pipeline_latency_throughput_sum",
            "reason": "report-only runtime package records launch metadata without executing accelerator work",
        }
        if dfg_data.get("fallback_decision") != expected_dfg_fallback:
            raise AssertionError(f"unexpected DFG-sim fallback decision: {dfg_data}")
        if dfg_data.get("selected_mapping_artifact_identity") != "":
            raise AssertionError(f"DFG-sim package must not require a mapping artifact: {dfg_data}")
        if dfg_data.get("fabric_adg_identity") != "":
            raise AssertionError(f"DFG-sim package must not require Fabric ADG: {dfg_data}")
        if set(dfg_data.get("required_runtime_features", [])) != {
            "dfg_sim_dispatch",
            "software_dataflow_report",
            "report_only_fallback",
        }:
            raise AssertionError(f"unexpected DFG-sim runtime features: {dfg_data}")
        if dfg_data.get("simulator_report_identities") != ["vecsum-dfg-sim-report"]:
            raise AssertionError(f"unexpected DFG-sim report identities: {dfg_data}")
        dfg_audit = out_dir / "dfg-runtime-package-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(dfg_audit),
                str(dfg_package),
            ],
            "DFG-sim runtime package audit",
        )
        dfg_audit_data = json.loads(dfg_audit.read_text())
        if dfg_audit_data.get("verdict") != "pass":
            raise AssertionError(f"expected DFG-sim runtime package audit pass: {dfg_audit_data}")

        rtl_package = out_dir / "rtl-runtime-package.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--target",
                "rtl-sim",
                "--output",
                str(rtl_package),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("RTL-sim runtime package without RTL inputs unexpectedly passed")
        rtl_data = json.loads(rtl_package.read_text())
        expected_rtl_target = {
            "target_kind": "simulator",
            "simulator": "rtl_sim",
            "profile_id": "simulator::rtl_sim::generated_hardware",
        }
        if rtl_data.get("target_profile") != expected_rtl_target:
            raise AssertionError(f"unexpected RTL-sim target profile: {rtl_data}")
        if rtl_data.get("status") != "blocked":
            raise AssertionError(f"RTL-sim package should be blocked without RTL inputs: {rtl_data}")
        rtl_records = rtl_data.get("diagnostic_records", [])
        expected_rtl_classes = {"missing_rtl_artifact", "unavailable_accelerator_target"}
        actual_rtl_classes = {
            record.get("diagnostic_class")
            for record in rtl_records
            if isinstance(record, dict)
        }
        if not expected_rtl_classes <= actual_rtl_classes:
            raise AssertionError(f"RTL-sim package missed diagnostics: {rtl_data}")
        rtl_audit = out_dir / "rtl-runtime-package-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(rtl_audit),
                str(rtl_package),
            ],
            "blocked RTL-sim runtime package audit",
        )

        hardware_package = out_dir / "hardware-runtime-package.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--target",
                "hardware",
                "--data-movement-policy",
                "copy_in_copy_out",
                "--platform-binding",
                "platform-binding::host-buffer::vecsum",
                "--output",
                str(hardware_package),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("hardware runtime package without hardware backend unexpectedly passed")
        hardware_data = json.loads(hardware_package.read_text())
        expected_hardware_target = {
            "target_kind": "hardware",
            "hardware_backend": "physical_accelerator",
            "profile_id": "hardware::physical_accelerator::explicit_platform_binding",
        }
        if hardware_data.get("target_profile") != expected_hardware_target:
            raise AssertionError(f"unexpected hardware target profile: {hardware_data}")
        if hardware_data.get("status") != "blocked":
            raise AssertionError(f"hardware package should be blocked without backend inputs: {hardware_data}")
        if hardware_data.get("runtime_configuration", {}).get("platform_binding_identity") != "platform-binding::host-buffer::vecsum":
            raise AssertionError(f"hardware runtime configuration should preserve platform binding: {hardware_data}")
        if hardware_data.get("required_data_movement_policies") != ["copy_in_copy_out"]:
            raise AssertionError(f"hardware package should record required data movement policies: {hardware_data}")
        if hardware_data.get("required_synchronization_policies") != ["host_wait"]:
            raise AssertionError(f"hardware package should record required synchronization policies: {hardware_data}")
        hardware_records = hardware_data.get("diagnostic_records", [])
        actual_hardware_classes = {
            record.get("diagnostic_class")
            for record in hardware_records
            if isinstance(record, dict)
        }
        if "unavailable_accelerator_target" not in actual_hardware_classes:
            raise AssertionError(f"hardware package missed unavailable target diagnostic: {hardware_data}")
        hardware_audit = out_dir / "hardware-runtime-package-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(hardware_audit),
                str(hardware_package),
            ],
            "blocked hardware runtime package audit",
        )

        require_acceleration_package = out_dir / "require-acceleration-hardware-runtime-package.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_runtime_package.sh",
                "--target",
                "hardware",
                "--fallback-policy",
                "require_acceleration",
                "--data-movement-policy",
                "copy_in_copy_out",
                "--platform-binding",
                "platform-binding::host-buffer::vecsum",
                "--output",
                str(require_acceleration_package),
                "--artifact",
                str(out_dir / "pnr-mapping.json"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("require-acceleration hardware runtime package unexpectedly passed")
        require_acceleration_data = json.loads(require_acceleration_package.read_text())
        if require_acceleration_data.get("fallback_decision", {}).get("policy") != "require_acceleration":
            raise AssertionError(f"require-acceleration package should preserve policy: {require_acceleration_data}")
        if require_acceleration_data.get("fallback_decision", {}).get("decision") != "blocked":
            raise AssertionError(f"require-acceleration package should block without target: {require_acceleration_data}")
        require_acceleration_records = require_acceleration_data.get("diagnostic_records", [])
        require_acceleration_classes = {
            record.get("diagnostic_class")
            for record in require_acceleration_records
            if isinstance(record, dict)
        }
        if "user_requested_acceleration_failure" not in require_acceleration_classes:
            raise AssertionError(
                f"require-acceleration package should diagnose user-requested acceleration failure: "
                f"{require_acceleration_data}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
