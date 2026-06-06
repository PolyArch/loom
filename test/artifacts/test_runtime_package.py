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
    "selected_mapping_artifact_identity",
    "fabric_adg_identity",
    "target_profile",
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
