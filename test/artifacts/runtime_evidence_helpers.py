#!/usr/bin/env python3
"""Runtime evidence projection helpers shared by report generators."""

from __future__ import annotations


RUNTIME_EVIDENCE_KEYS = (
    "runtime_package_identity",
    "runtime_report_identity",
    "host_program_identity",
    "host_wrapper_identity",
    "host_interface",
    "runtime_handle_model",
    "work_package_metadata",
    "work_package_identity",
    "launch_descriptor_identity",
    "launch_descriptor",
    "mapping_artifact_identity",
    "fabric_adg_identity",
    "target_profile_id",
    "target_profile",
    "fallback_policy",
    "launch_status",
    "target_status",
    "runtime_trace_identity",
    "profiling_record_identity",
    "data_movement_policy",
    "synchronization_mode",
    "memory_descriptors",
    "argument_descriptors",
    "runtime_configuration",
    "required_runtime_features",
    "required_data_movement_policies",
    "required_synchronization_policies",
    "simulator_report_identities",
    "input_artifact_fingerprints",
    "output_buffer_identities",
    "diagnostic_records",
    "report_output_configuration",
    "fallback_decision",
)


def dict_field(value: dict[str, object], key: str) -> dict[str, object]:
    field = value.get(key, {})
    return field if isinstance(field, dict) else {}


def list_field(value: dict[str, object], key: str) -> list[object]:
    field = value.get(key, [])
    return field if isinstance(field, list) else []


def string_list_field(value: dict[str, object], key: str) -> list[str]:
    return [str(item) for item in list_field(value, key) if isinstance(item, str)]


def object_list_field(value: dict[str, object], key: str) -> list[dict[str, object]]:
    return [item for item in list_field(value, key) if isinstance(item, dict)]


def string_map_field(value: dict[str, object], key: str) -> dict[str, str]:
    field = dict_field(value, key)
    return {
        str(identity): str(fingerprint)
        for identity, fingerprint in field.items()
        if isinstance(identity, str) and isinstance(fingerprint, str)
    }


def fallback_decision(runtime_package: dict[str, object], report: dict[str, object]) -> dict[str, object]:
    fallback = report.get("fallback_decision")
    if not isinstance(fallback, dict):
        fallback = runtime_package.get("fallback_decision", {})
    return fallback if isinstance(fallback, dict) else {}


def custom_data_movement_policy(runtime_package: dict[str, object], report: dict[str, object]) -> str:
    if str(runtime_package.get("data_movement_policy", "")) != "custom":
        return ""
    report_policy = report.get("custom_data_movement_policy_identity")
    if isinstance(report_policy, str) and report_policy:
        return report_policy
    runtime_configuration = dict_field(runtime_package, "runtime_configuration")
    runtime_policy = runtime_configuration.get("custom_data_movement_policy_identity")
    if isinstance(runtime_policy, str) and runtime_policy:
        return runtime_policy
    return ""


def runtime_evidence_from_package(
    runtime_package: dict[str, object],
    runtime_package_identity: str,
) -> dict[str, object]:
    report = dict_field(runtime_package, "runtime_report")
    evidence = {
        "runtime_package_identity": runtime_package_identity,
        "runtime_report_identity": str(report.get("report_id", "")),
        "host_program_identity": str(report.get("host_program_identity", "")),
        "host_wrapper_identity": str(runtime_package.get("host_wrapper_identity", "")),
        "host_interface": dict_field(runtime_package, "host_interface"),
        "runtime_handle_model": dict_field(runtime_package, "runtime_handle_model"),
        "work_package_metadata": dict_field(runtime_package, "work_package_metadata"),
        "work_package_identity": str(report.get("work_package_identity", "")),
        "launch_descriptor_identity": str(report.get("launch_descriptor_identity", "")),
        "launch_descriptor": dict_field(runtime_package, "launch_descriptor"),
        "mapping_artifact_identity": str(report.get("mapping_artifact_identity", "")),
        "fabric_adg_identity": str(report.get("fabric_adg_identity", "")),
        "target_profile_id": str(report.get("target_profile_id", "")),
        "target_profile": dict_field(runtime_package, "target_profile"),
        "fallback_policy": str(runtime_package.get("fallback_policy", "")),
        "launch_status": str(report.get("launch_status", "")),
        "target_status": str(report.get("target_status", "")),
        "runtime_trace_identity": str(report.get("runtime_trace_identity", "")),
        "profiling_record_identity": str(report.get("profiling_record_identity", "")),
        "data_movement_policy": str(runtime_package.get("data_movement_policy", "")),
        "synchronization_mode": str(runtime_package.get("synchronization_mode", "")),
        "memory_descriptors": object_list_field(runtime_package, "memory_descriptors"),
        "argument_descriptors": object_list_field(runtime_package, "argument_descriptors"),
        "runtime_configuration": dict_field(runtime_package, "runtime_configuration"),
        "required_runtime_features": string_list_field(runtime_package, "required_runtime_features"),
        "required_data_movement_policies": string_list_field(
            runtime_package,
            "required_data_movement_policies",
        ),
        "required_synchronization_policies": string_list_field(
            runtime_package,
            "required_synchronization_policies",
        ),
        "simulator_report_identities": string_list_field(report, "simulator_report_identities"),
        "input_artifact_fingerprints": string_map_field(runtime_package, "input_artifact_fingerprints"),
        "output_buffer_identities": string_list_field(report, "output_buffer_identities"),
        "diagnostic_records": object_list_field(report, "diagnostic_records"),
        "report_output_configuration": dict_field(runtime_package, "report_output_configuration"),
        "fallback_decision": fallback_decision(runtime_package, report),
    }
    custom_policy = custom_data_movement_policy(runtime_package, report)
    if custom_policy:
        evidence["custom_data_movement_policy_identity"] = custom_policy
    return evidence


def runtime_evidence_summary(
    evidence: dict[str, object],
    workload_report_bundle_identity: str,
) -> dict[str, object]:
    summary = {"workload_report_bundle_identity": workload_report_bundle_identity}
    for key in RUNTIME_EVIDENCE_KEYS:
        summary[key] = evidence.get(key, default_value_for_key(key))
    custom_policy = evidence.get("custom_data_movement_policy_identity")
    if isinstance(custom_policy, str) and custom_policy:
        summary["custom_data_movement_policy_identity"] = custom_policy
    return summary


def default_value_for_key(key: str) -> object:
    if key in {
        "host_interface",
        "runtime_handle_model",
        "work_package_metadata",
        "launch_descriptor",
        "target_profile",
        "runtime_configuration",
        "input_artifact_fingerprints",
        "report_output_configuration",
        "fallback_decision",
    }:
        return {}
    if key in {
        "memory_descriptors",
        "argument_descriptors",
        "required_runtime_features",
        "required_data_movement_policies",
        "required_synchronization_policies",
        "simulator_report_identities",
        "output_buffer_identities",
        "diagnostic_records",
    }:
        return []
    return ""
