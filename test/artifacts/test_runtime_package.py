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
    "launch_descriptor",
    "runtime_handle_model",
    "selected_mapping_artifact_identity",
    "fabric_adg_identity",
    "target_profile",
    "runtime_configuration",
    "fallback_policy",
    "fallback_decision",
    "synchronization_mode",
    "data_movement_policy",
    "memory_descriptors",
    "argument_descriptors",
    "required_runtime_features",
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
        if data["synchronization_mode"] != "host_wait":
            raise AssertionError(f"unexpected synchronization mode: {data}")
        if data["data_movement_policy"] != "simulated":
            raise AssertionError(f"unexpected data movement policy: {data}")
        memory_descriptors = data.get("memory_descriptors", [])
        if not isinstance(memory_descriptors, list) or len(memory_descriptors) != 1:
            raise AssertionError(f"runtime package should include one memory descriptor: {data}")
        descriptor = memory_descriptors[0]
        expected_memory = {
            "logical_argument": "vecsum.default_input",
            "direction": "read_write",
            "policy": "simulated",
            "runtime_input_identity": "test-app-fixture::vecsum::default",
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
