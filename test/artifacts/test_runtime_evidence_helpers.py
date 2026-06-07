#!/usr/bin/env python3
"""Regression test for shared runtime evidence extraction helpers."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import runtime_evidence_helpers  # noqa: E402


def sample_runtime_package() -> dict[str, object]:
    return {
        "host_wrapper_identity": "runtime-wrapper::vecsum",
        "host_interface": {"invocation_abi": "loom_runtime_package_v1"},
        "runtime_handle_model": {"handle_semantics": "host_visible"},
        "work_package_metadata": {"workload": "vecsum"},
        "launch_descriptor": {"descriptor_id": "launch-descriptor::vecsum"},
        "target_profile": {"target_kind": "simulator"},
        "fallback_policy": "report_only",
        "data_movement_policy": "custom",
        "synchronization_mode": "host_wait",
        "memory_descriptors": [{"name": "input"}],
        "argument_descriptors": [{"name": "runtime_input"}],
        "runtime_configuration": {
            "configuration_id": "runtime-config::vecsum",
            "custom_data_movement_policy_identity": "runtime-policy::dma-window::vecsum",
        },
        "required_runtime_features": ["runtime_package_metadata"],
        "required_data_movement_policies": ["custom"],
        "required_synchronization_policies": ["host_wait"],
        "report_output_configuration": {"emit_runtime_report": True},
        "input_artifact_fingerprints": {"pnr-mapping": "a" * 64},
        "fallback_decision": {
            "policy": "report_only",
            "decision": "report_only",
            "fallback_taken": False,
        },
        "runtime_report": {
            "report_id": "runtime-report::vecsum",
            "host_program_identity": "test-app-host::vecsum",
            "host_wrapper_identity": "runtime-wrapper::vecsum",
            "work_package_identity": "work-package::vecsum",
            "launch_descriptor_identity": "launch-descriptor::vecsum",
            "mapping_artifact_identity": "pnr-mapping",
            "fabric_adg_identity": "shared_reduction_adg",
            "target_profile_id": "simulator::cgra_sim",
            "launch_status": "not_run",
            "target_status": "not_run",
            "runtime_trace_identity": "",
            "profiling_record_identity": "",
            "simulator_report_identities": ["vecsum-cgra-sim-report"],
            "output_buffer_identities": ["buffer::out"],
            "diagnostic_records": [{"diagnostic_id": "runtime::1"}],
            "fallback_decision": {
                "policy": "report_only",
                "decision": "report_only",
                "fallback_taken": False,
            },
            "custom_data_movement_policy_identity": "runtime-policy::dma-window::vecsum",
        },
    }


def main() -> int:
    evidence = runtime_evidence_helpers.runtime_evidence_from_package(
        sample_runtime_package(),
        "runtime-package",
    )
    expected_keys = set(runtime_evidence_helpers.RUNTIME_EVIDENCE_KEYS)
    expected_keys.add("custom_data_movement_policy_identity")
    if set(evidence) != expected_keys:
        raise AssertionError(f"runtime evidence keys drifted: {sorted(evidence)}")
    if evidence["runtime_package_identity"] != "runtime-package":
        raise AssertionError(f"runtime identity was not preserved: {evidence}")
    if evidence["custom_data_movement_policy_identity"] != "runtime-policy::dma-window::vecsum":
        raise AssertionError(f"custom runtime policy was not preserved: {evidence}")
    summary = runtime_evidence_helpers.runtime_evidence_summary(
        evidence,
        "workload-report-bundle",
    )
    if summary["workload_report_bundle_identity"] != "workload-report-bundle":
        raise AssertionError(f"summary missed workload report identity: {summary}")
    expected_summary_keys = set(runtime_evidence_helpers.RUNTIME_EVIDENCE_KEYS)
    expected_summary_keys.add("workload_report_bundle_identity")
    expected_summary_keys.add("custom_data_movement_policy_identity")
    if set(summary) != expected_summary_keys:
        raise AssertionError(f"runtime evidence summary keys drifted: {sorted(summary)}")
    for key in runtime_evidence_helpers.RUNTIME_EVIDENCE_KEYS:
        if summary[key] != evidence[key]:
            raise AssertionError(f"summary changed runtime evidence field {key}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
