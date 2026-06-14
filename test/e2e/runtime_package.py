#!/usr/bin/env python3
"""Emit runtime package descriptors from mapped simulator artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402
import artifact_io_helpers  # noqa: E402


artifact_id = intermediate_artifacts.artifact_id_for_path
input_artifact_fingerprints = intermediate_artifacts.input_artifact_fingerprints
read_json = artifact_io_helpers.read_json
group_paths = artifact_io_helpers.group_paths
first_path = artifact_io_helpers.first_path
matching_rtl_manifest_path = artifact_io_helpers.matching_rtl_manifest_path
hardware_matches = artifact_io_helpers.hardware_matches


DATA_MOVEMENT_POLICIES = (
    "shared_coherent",
    "shared_noncoherent",
    "copy_in_copy_out",
    "device_local",
    "simulated",
    "custom",
)
FALLBACK_POLICIES = (
    "require_acceleration",
    "allow_host_fallback",
    "allow_scalar_fallback",
    "report_only",
)
SYNCHRONIZATION_MODES = (
    "host_wait",
    "host_fence",
    "device_poll",
)

WORKLOAD_MEMORY_LAYOUTS: dict[str, dict[str, object]] = {
    "vecsum": {
        "byte_size": 256,
        "element_layout": "u32[64]",
        "alignment_bytes": 4,
    },
    "dotproduct": {
        "byte_size": 512,
        "element_layout": "f32[64];f32[64]",
        "alignment_bytes": 4,
    },
    "dot_product_3d": {
        "byte_size": 448,
        "element_layout": "f32[48];f32[48];f32[16]",
        "alignment_bytes": 4,
    },
    "vecadd": {
        "byte_size": 768,
        "element_layout": "f32[64];f32[64];f32[64]",
        "alignment_bytes": 4,
    },
    "vecmul": {
        "byte_size": 192,
        "element_layout": "f32[16];f32[16];f32[16]",
        "alignment_bytes": 4,
    },
    "xor_block": {
        "byte_size": 384,
        "element_layout": "u32[32];u32[32];u32[32]",
        "alignment_bytes": 4,
    },
    "byte_swap": {
        "byte_size": 256,
        "element_layout": "u32[32];u32[32]",
        "alignment_bytes": 4,
    },
    "downsample": {
        "byte_size": 80,
        "element_layout": "f32[16];f32[4]",
        "alignment_bytes": 4,
    },
    "upsample": {
        "byte_size": 80,
        "element_layout": "f32[4];f32[16]",
        "alignment_bytes": 4,
    },
    "delta_encode": {
        "byte_size": 80,
        "element_layout": "u32[10];u32[10]",
        "alignment_bytes": 4,
    },
    "relu": {
        "byte_size": 256,
        "element_layout": "f32[32];f32[32]",
        "alignment_bytes": 4,
    },
    "variance": {
        "byte_size": 64,
        "element_layout": "f32[16]",
        "alignment_bytes": 4,
    },
    "prefix_sum": {
        "byte_size": 512,
        "element_layout": "i32[64];i32[64]",
        "alignment_bytes": 4,
    },
    "cumsum": {
        "byte_size": 8192,
        "element_layout": "f32[1024];f32[1024]",
        "alignment_bytes": 4,
    },
    "prefix_sum_inclusive": {
        "byte_size": 8192,
        "element_layout": "u32[1024];u32[1024]",
        "alignment_bytes": 4,
    },
    "integrate_trapz": {
        "byte_size": 72,
        "element_layout": "f32[9];f32[9]",
        "alignment_bytes": 4,
    },
    "reduction": {
        "byte_size": 512,
        "element_layout": "i32[128]",
        "alignment_bytes": 4,
    },
    "mean": {
        "byte_size": 256,
        "element_layout": "f32[64]",
        "alignment_bytes": 4,
    },
    "vecnorm_l1": {
        "byte_size": 256,
        "element_layout": "i32[64]",
        "alignment_bytes": 4,
    },
    "vecnorm_l2": {
        "byte_size": 256,
        "element_layout": "i32[64]",
        "alignment_bytes": 4,
    },
    "correlation": {
        "byte_size": 1028,
        "element_layout": "f32[128];f32[16];f32[113]",
        "alignment_bytes": 4,
    },
    "spmv": {
        "byte_size": 128,
        "element_layout": "u32[9];u32[9];u32[5];u32[5];u32[4]",
        "alignment_bytes": 4,
    },
    "convolve_1d": {
        "byte_size": 1028,
        "element_layout": "f32[128];f32[7];f32[122]",
        "alignment_bytes": 4,
    },
    "matvec": {
        "byte_size": 116,
        "element_layout": "u32[20];u32[5];u32[4]",
        "alignment_bytes": 4,
    },
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("cgra-sim", "dfg-sim", "rtl-sim", "hardware"), default="cgra-sim")
    parser.add_argument("--data-movement-policy", choices=DATA_MOVEMENT_POLICIES, default="simulated")
    parser.add_argument("--custom-data-movement-policy", default="")
    parser.add_argument("--fallback-policy", choices=FALLBACK_POLICIES, default="report_only")
    parser.add_argument("--synchronization-mode", choices=SYNCHRONIZATION_MODES, default="host_wait")
    parser.add_argument("--platform-binding", default="")
    parser.add_argument("--enable-runtime-trace", action="store_true")
    parser.add_argument("--enable-runtime-profiling", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    return parser.parse_args(argv)


def string_field(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    return value if isinstance(value, str) else ""


def runtime_input_identity(workload: str, comparison: dict[str, object], dfg: dict[str, object]) -> str:
    identity = string_field(comparison, "runtime_input_identity")
    if identity:
        return identity
    identity = string_field(dfg, "runtime_input_identity")
    if identity:
        return identity
    return f"test-app-fixture::{workload}::default"


def report_identities(paths: list[Path | None]) -> list[str]:
    identities = []
    for path in paths:
        identity = artifact_id(path)
        if identity:
            identities.append(identity)
    return identities


def path_with_artifact_identity(paths: list[Path], identity: str) -> Path | None:
    if not identity:
        return None
    for path in paths:
        if artifact_id(path) == identity:
            return path
    return None


def matching_cgra_comparison_inputs(
    grouped: dict[str, list[Path]],
) -> tuple[Path, Path, Path, dict[str, object]] | None:
    mapping_paths = grouped.get("pnr_mapping_artifact", [])
    cgra_paths = grouped.get("cgra_sim_report", [])
    for comparison_path in grouped.get("sim_comparison_report", []):
        comparison = read_json(comparison_path)
        mapping_path = path_with_artifact_identity(
            mapping_paths,
            string_field(comparison, "mapping_artifact_identity"),
        )
        cgra_path = path_with_artifact_identity(
            cgra_paths,
            string_field(comparison, "cgra_sim_report_identity"),
        )
        if mapping_path is not None and cgra_path is not None:
            return comparison_path, mapping_path, cgra_path, comparison
    return None


def cgra_report_matches_mapping(cgra: dict[str, object], mapping: dict[str, object]) -> bool:
    for key in ("workload", "mapping_id"):
        cgra_value = string_field(cgra, key)
        mapping_value = string_field(mapping, key)
        if cgra_value and mapping_value and cgra_value != mapping_value:
            return False
    cgra_hardware = string_field(cgra, "hardware")
    mapping_hardware = string_field(mapping, "hardware")
    if cgra_hardware and mapping_hardware and not hardware_matches(cgra_hardware, mapping_hardware):
        return False
    return True


def matching_cgra_mapping_inputs(
    grouped: dict[str, list[Path]],
) -> tuple[Path, Path] | None:
    mapping_paths = grouped.get("pnr_mapping_artifact", [])
    for cgra_path in grouped.get("cgra_sim_report", []):
        cgra = read_json(cgra_path)
        mapping_matches = [
            path
            for path in mapping_paths
            if cgra_report_matches_mapping(cgra, read_json(path))
        ]
        if len(mapping_matches) == 1:
            return mapping_matches[0], cgra_path
    return None


def matching_rtl_inputs(
    grouped: dict[str, list[Path]],
) -> tuple[Path, Path] | None:
    mapping_paths = grouped.get("pnr_mapping_artifact", [])
    for rtl_manifest_path in grouped.get("rtl_manifest", []):
        rtl_manifest = read_json(rtl_manifest_path)
        mapping_path = path_with_artifact_identity(
            mapping_paths,
            string_field(rtl_manifest, "mapping_artifact_identity"),
        )
        if mapping_path is not None:
            return mapping_path, rtl_manifest_path
    for rtl_manifest_path in grouped.get("rtl_manifest", []):
        rtl_manifest = read_json(rtl_manifest_path)
        if string_field(rtl_manifest, "mapping_artifact_identity"):
            continue
        source_fabric = string_field(rtl_manifest, "source_fabric_adg_identity")
        hardware_matches_manifest = [
            path
            for path in mapping_paths
            if hardware_matches(source_fabric, string_field(read_json(path), "hardware"))
        ]
        if len(hardware_matches_manifest) == 1:
            return hardware_matches_manifest[0], rtl_manifest_path
    return None


def platform_binding_workload(platform_binding: str) -> str:
    return platform_binding.rsplit("::", 1)[-1] if platform_binding else ""


def matching_hardware_mapping_input(
    grouped: dict[str, list[Path]],
    platform_binding: str,
) -> tuple[Path | None, bool]:
    mapping_paths = grouped.get("pnr_mapping_artifact", [])
    if len(mapping_paths) == 1:
        return mapping_paths[0], False
    workload = platform_binding_workload(platform_binding)
    if workload:
        matches = [
            path
            for path in mapping_paths
            if string_field(read_json(path), "workload") == workload
        ]
        if len(matches) == 1:
            return matches[0], False
    return None, len(mapping_paths) > 1


def diagnostic_class(message: str) -> str:
    if "requires RTL" in message:
        return "missing_rtl_artifact"
    if "target is not available" in message:
        return "unavailable_accelerator_target"
    if "platform memory binding is missing" in message:
        return "missing_platform_memory_binding"
    if "memory layout is missing" in message:
        return "missing_runtime_input_layout"
    if "requires CGRA-sim report" in message or "requires DFG-sim report" in message:
        return "missing_simulator_report"
    if "mapping artifact is not passing" in message:
        return "mapping_artifact_failure"
    if "requires mapping artifact" in message or "mapping identity is missing" in message:
        return "missing_mapping_artifact"
    if "fabric ADG identity is missing" in message:
        return "missing_fabric_adg"
    if "requires simulated data movement policy" in message:
        return "unsupported_data_movement_policy"
    if "custom data movement policy" in message:
        return "unsupported_data_movement_policy"
    if "user-requested acceleration failed" in message:
        return "user_requested_acceleration_failure"
    if "does not consume" in message:
        return "unsupported_target_profile"
    if "identity mismatch" in message or "different mapping artifact" in message:
        return "stale_artifact_fingerprint"
    if "sim report is not passing" in message or "simulation comparison report is not passing" in message:
        return "simulator_target_failure"
    return "runtime_configuration_failure"


def failure_domain(message: str) -> str:
    diagnostic = diagnostic_class(message)
    if diagnostic in {
        "missing_mapping_artifact",
        "mapping_artifact_failure",
        "missing_fabric_adg",
        "stale_artifact_fingerprint",
        "unsupported_target_profile",
    }:
        return "compiler_artifacts"
    if diagnostic in {"missing_platform_memory_binding", "missing_runtime_input_layout"}:
        return "platform_services"
    if diagnostic in {"missing_simulator_report", "simulator_target_failure"}:
        return "simulator_execution"
    if diagnostic in {"missing_rtl_artifact", "unavailable_accelerator_target"}:
        return "hardware_execution"
    return "runtime_configuration"


def diagnostic_records(
    diagnostics: list[str],
    *,
    source_provenance: str,
    host_wrapper_identity: str,
) -> list[dict[str, str]]:
    records = []
    for index, message in enumerate(diagnostics, start=1):
        record = {
            "diagnostic_id": f"runtime-package::{index}",
            "diagnostic_class": diagnostic_class(message),
            "component": "runtime_package",
            "severity": "error",
            "message": message,
            "failure_domain": failure_domain(message),
        }
        if source_provenance:
            record["source_provenance"] = source_provenance
        if host_wrapper_identity:
            record["host_wrapper_identity"] = host_wrapper_identity
        records.append(record)
    return records


def fallback_decision(
    *,
    policy: str,
    target_profile: dict[str, str],
    status: str,
    diagnostics: list[str],
) -> dict[str, object]:
    target_profile_id = target_profile.get("profile_id", "")
    if status == "pass" and policy == "report_only":
        return {
            "policy": policy,
            "decision": "report_only",
            "fallback_taken": False,
            "target_profile_id": target_profile_id,
            "reason": "report-only runtime package records launch metadata without executing accelerator work",
        }
    if status == "pass":
        return {
            "policy": policy,
            "decision": "none",
            "fallback_taken": False,
            "target_profile_id": target_profile_id,
            "reason": "selected target profile metadata is available; no fallback was selected",
        }
    return {
        "policy": policy,
        "decision": "blocked",
        "fallback_taken": False,
        "target_profile_id": target_profile_id,
        "reason": "; ".join(diagnostics) if diagnostics else "runtime package is blocked",
    }


def fallback_feature(policy: str) -> str:
    return {
        "require_acceleration": "require_acceleration_policy",
        "allow_host_fallback": "host_fallback_policy",
        "allow_scalar_fallback": "scalar_fallback_policy",
        "report_only": "report_only_fallback",
    }[policy]


def build_launch_descriptor(
    *,
    descriptor_id: str,
    work_package_identity: str,
    workload: str,
    mapping_identity: str,
    target_profile: dict[str, str],
    memory_descriptors: list[dict[str, object]],
    argument_descriptors: list[dict[str, str]],
    fallback_policy: str,
    synchronization_mode: str,
    runtime_trace_enabled: bool,
    runtime_profiling_enabled: bool,
) -> dict[str, object]:
    return {
        "descriptor_id": descriptor_id,
        "work_package_identity": work_package_identity,
        "selected_accelerator_region": f"accelerator-region::{workload}" if workload != "unknown" else "",
        "logical_thread_domain": f"thread-domain::{workload}" if workload != "unknown" else "",
        "argument_descriptor_names": [
            descriptor["name"]
            for descriptor in argument_descriptors
            if "name" in descriptor
        ],
        "memory_descriptor_logical_arguments": [
            descriptor["logical_argument"]
            for descriptor in memory_descriptors
            if "logical_argument" in descriptor
        ],
        "argument_descriptors": argument_descriptors,
        "memory_descriptors": memory_descriptors,
        "scalar_value_descriptors": [],
        "selected_mapping_artifact_identity": mapping_identity,
        "target_profile_id": target_profile.get("profile_id", ""),
        "fallback_policy": fallback_policy,
        "synchronization_mode": synchronization_mode,
        "profiling_settings": {"enabled": runtime_profiling_enabled},
        "trace_settings": {"enabled": runtime_trace_enabled},
    }


def work_package_metadata(
    *,
    work_package_identity: str,
    workload: str,
    selected_mapping_identity: str,
    fabric_adg_identity: str,
    runtime_input: str,
) -> dict[str, str]:
    return {
        "work_package_identity": work_package_identity,
        "workload": workload,
        "selected_accelerator_region": f"accelerator-region::{workload}" if workload != "unknown" else "",
        "logical_thread_domain": f"thread-domain::{workload}" if workload != "unknown" else "",
        "selected_mapping_artifact_identity": selected_mapping_identity,
        "fabric_adg_identity": fabric_adg_identity,
        "runtime_input_identity": runtime_input,
    }


def runtime_handle_model() -> dict[str, object]:
    return {
        "handle_kind": "host_visible_launch_handle",
        "ir_token_kind": "not_dataflow_thread_token",
        "completion_source": "runtime_target_status",
        "operations": [
            "query_status",
            "wait_for_completion",
            "collect_diagnostics",
            "collect_profiling_data",
        ],
    }


def runtime_configuration(
    *,
    target_profile: dict[str, str],
    data_movement_policy: str,
    custom_data_movement_policy: str,
    platform_binding: str,
    fallback_policy: str,
    synchronization_mode: str,
) -> dict[str, str]:
    configuration_id = f"runtime-config::{fallback_policy}::{data_movement_policy}::{synchronization_mode}"
    if data_movement_policy == "custom" and custom_data_movement_policy:
        configuration_id = f"{configuration_id}::{custom_data_movement_policy}"
    configuration = {
        "configuration_id": configuration_id,
        "target_profile_id": target_profile.get("profile_id", ""),
        "data_movement_policy": data_movement_policy,
        "platform_binding_identity": platform_binding,
        "fallback_policy": fallback_policy,
        "synchronization_mode": synchronization_mode,
    }
    if data_movement_policy == "custom" and custom_data_movement_policy:
        configuration["custom_data_movement_policy_identity"] = custom_data_movement_policy
    return configuration


def host_interface(
    *,
    host_program_identity: str,
    host_wrapper_identity: str,
    runtime_input: str,
) -> dict[str, object]:
    return {
        "host_program_identity": host_program_identity,
        "host_wrapper_identity": host_wrapper_identity,
        "invocation_abi": "loom_runtime_package_v1",
        "compatibility_mode_requires_runtime": False,
        "acceleration_mode_requires_runtime_package": True,
        "source_provenance": runtime_input,
    }


def memory_layout_for_workload(workload: str) -> dict[str, object] | None:
    return WORKLOAD_MEMORY_LAYOUTS.get(workload)


def address_space_for_policy(data_movement_policy: str, platform_binding: str) -> str:
    if data_movement_policy == "simulated":
        return "simulator::memory_model"
    if platform_binding:
        return f"{platform_binding}::address_space"
    return "platform::unbound_address_space"


def coherence_requirement_for_policy(data_movement_policy: str) -> str:
    return {
        "shared_coherent": "shared_coherent",
        "shared_noncoherent": "explicit_flush_invalidate",
        "copy_in_copy_out": "copy_boundary",
        "device_local": "device_local",
        "simulated": "simulator_consistent",
        "custom": "custom_policy",
    }[data_movement_policy]


def memory_descriptor(
    *,
    workload: str,
    data_movement_policy: str,
    custom_data_movement_policy: str,
    runtime_input: str,
    platform_binding: str,
    layout: dict[str, object],
) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "logical_argument": f"{workload}.default_input",
        "host_buffer_identity": f"runtime-buffer::{workload}::default_input",
        "direction": "read_write",
        "policy": data_movement_policy,
        "runtime_input_identity": runtime_input,
        "layout_source_kind": "static_workload_fixture",
        "layout_source_identity": runtime_input,
        **layout,
        "address_space": address_space_for_policy(data_movement_policy, platform_binding),
        "coherence_requirement": coherence_requirement_for_policy(data_movement_policy),
        "transfer_policy": data_movement_policy,
    }
    if platform_binding:
        descriptor["platform_binding_identity"] = platform_binding
    if data_movement_policy == "custom" and custom_data_movement_policy:
        descriptor["custom_data_movement_policy_identity"] = custom_data_movement_policy
    return descriptor


def runtime_report(
    *,
    workload: str,
    mapping_id: str,
    host_program_identity: str,
    host_wrapper_identity: str,
    work_package_identity: str,
    launch_descriptor_identity: str,
    selected_mapping_identity: str,
    fabric_adg_identity: str,
    target_profile: dict[str, str],
    data_movement_policy: str,
    custom_data_movement_policy: str,
    synchronization_mode: str,
    fallback_policy: str,
    fallback: dict[str, object],
    simulator_report_identities: list[str],
    source_provenance: str,
    diagnostics: list[str],
) -> dict[str, object]:
    report = {
        "report_id": f"runtime-report::{workload}::{mapping_id}::{fallback_policy}",
        "host_program_identity": host_program_identity,
        "host_wrapper_identity": host_wrapper_identity,
        "work_package_identity": work_package_identity,
        "launch_descriptor_identity": launch_descriptor_identity,
        "mapping_artifact_identity": selected_mapping_identity,
        "fabric_adg_identity": fabric_adg_identity,
        "target_profile_id": target_profile.get("profile_id", ""),
        "memory_policy": data_movement_policy,
        "synchronization_mode": synchronization_mode,
        "fallback_decision": fallback,
        "simulator_report_identities": simulator_report_identities,
        "runtime_trace_identity": "",
        "profiling_record_identity": "",
        "output_buffer_identities": [],
        "launch_status": "not_run",
        "target_status": "not_run",
        "diagnostic_records": diagnostic_records(
            diagnostics,
            source_provenance=source_provenance,
            host_wrapper_identity=host_wrapper_identity,
        ),
    }
    if data_movement_policy == "custom" and custom_data_movement_policy:
        report["custom_data_movement_policy_identity"] = custom_data_movement_policy
    return report


def report_output_configuration(
    report: dict[str, object],
    launch_descriptor: dict[str, object],
) -> dict[str, object]:
    trace_settings = launch_descriptor.get("trace_settings", {})
    if not isinstance(trace_settings, dict):
        trace_settings = {}
    profiling_settings = launch_descriptor.get("profiling_settings", {})
    if not isinstance(profiling_settings, dict):
        profiling_settings = {}
    return {
        "runtime_report_identity": str(report.get("report_id", "")),
        "diagnostic_output_enabled": True,
        "trace_output_enabled": bool(trace_settings.get("enabled", False)),
        "profiling_output_enabled": bool(profiling_settings.get("enabled", False)),
    }


def build_package(
    paths: list[Path],
    target: str,
    data_movement_policy: str,
    custom_data_movement_policy: str,
    platform_binding: str,
    fallback_policy: str,
    synchronization_mode: str,
    runtime_trace_enabled: bool,
    runtime_profiling_enabled: bool,
) -> dict[str, object]:
    grouped = group_paths(paths)
    mapping_path = first_path(grouped, "pnr_mapping_artifact")
    dfg_path = first_path(grouped, "dfg_sim_report")
    cgra_path = first_path(grouped, "cgra_sim_report")
    comparison_path = first_path(grouped, "sim_comparison_report")
    rtl_manifest_path = None
    hardware_mapping_inputs_ambiguous = False

    comparison = read_json(comparison_path)
    if target == "cgra-sim":
        comparison_inputs = matching_cgra_comparison_inputs(grouped)
        if comparison_inputs is not None:
            comparison_path, mapping_path, cgra_path, comparison = comparison_inputs
        else:
            cgra_inputs = matching_cgra_mapping_inputs(grouped)
            if cgra_inputs is not None:
                mapping_path, cgra_path = cgra_inputs
    elif target == "rtl-sim":
        rtl_inputs = matching_rtl_inputs(grouped)
        if rtl_inputs is not None:
            mapping_path, rtl_manifest_path = rtl_inputs
    elif target == "hardware":
        mapping_path, hardware_mapping_inputs_ambiguous = matching_hardware_mapping_input(grouped, platform_binding)

    mapping = read_json(mapping_path)
    dfg = read_json(dfg_path)
    cgra = read_json(cgra_path)
    consumed_mapping_path = mapping_path if target in {"cgra-sim", "rtl-sim", "hardware"} else None
    consumed_dfg_path = dfg_path if target == "dfg-sim" else None
    consumed_cgra_path = cgra_path if target == "cgra-sim" else None
    consumed_comparison_path = comparison_path if target == "cgra-sim" else None
    mapping_for_identity = mapping if consumed_mapping_path is not None else {}
    dfg_for_identity = dfg if consumed_dfg_path is not None else {}
    cgra_for_identity = cgra if consumed_cgra_path is not None else {}
    comparison_for_identity = comparison if consumed_comparison_path is not None else {}
    workload = (
        string_field(mapping_for_identity, "workload")
        or string_field(dfg_for_identity, "workload")
        or string_field(cgra_for_identity, "workload")
        or "unknown"
    )
    hardware = string_field(mapping_for_identity, "hardware") or string_field(cgra_for_identity, "hardware")
    runtime_input = runtime_input_identity(workload, comparison_for_identity, dfg_for_identity)
    if target == "rtl-sim" and rtl_manifest_path is None and hardware:
        rtl_manifest_path = matching_rtl_manifest_path(grouped.get("rtl_manifest", []), hardware)

    diagnostics: list[str] = []
    if data_movement_policy == "custom" and not custom_data_movement_policy:
        diagnostics.append("custom data movement policy requires policy identity")
    if data_movement_policy != "custom" and custom_data_movement_policy:
        diagnostics.append("custom data movement policy identity is only valid for custom data movement")
    if target == "cgra-sim":
        if not mapping:
            diagnostics.append("CGRA-sim target requires mapping artifact")
        elif string_field(mapping, "status") != "pass":
            diagnostics.append("PnR mapping artifact is not passing")
        if dfg_path is not None:
            diagnostics.append("CGRA-sim target does not consume DFG-sim reports")
        if cgra_path is None:
            diagnostics.append("CGRA-sim target requires CGRA-sim report")
        elif string_field(cgra, "status") != "pass":
            diagnostics.append("CGRA-sim report is not passing")
        if mapping and cgra:
            if string_field(cgra, "workload") and string_field(cgra, "workload") != string_field(mapping, "workload"):
                diagnostics.append("CGRA-sim report workload identity mismatch")
            if string_field(cgra, "hardware") and string_field(cgra, "hardware") != string_field(mapping, "hardware"):
                diagnostics.append("CGRA-sim report hardware identity mismatch")
            if string_field(cgra, "mapping_id") and string_field(cgra, "mapping_id") != string_field(mapping, "mapping_id"):
                diagnostics.append("CGRA-sim report mapping identity mismatch")
        if not hardware:
            diagnostics.append("fabric ADG identity is missing")
        if not string_field(mapping, "mapping_id"):
            diagnostics.append("mapping identity is missing")
    elif target == "dfg-sim":
        if dfg_path is None:
            diagnostics.append("DFG-sim target requires DFG-sim report")
        elif string_field(dfg, "status") != "pass":
            diagnostics.append("DFG-sim report is not passing")
        if mapping_path is not None:
            diagnostics.append("DFG-sim target does not consume mapping artifacts")
        if cgra_path is not None:
            diagnostics.append("DFG-sim target does not consume CGRA-sim reports")
        if comparison_path is not None:
            diagnostics.append("DFG-sim target does not consume simulation comparison reports")
    elif target == "rtl-sim":
        if not mapping:
            diagnostics.append("RTL-sim target requires mapping artifact")
        if not hardware:
            diagnostics.append("fabric ADG identity is missing")
        if rtl_manifest_path is None:
            diagnostics.append("RTL-sim target requires RTL manifest artifact")
        diagnostics.append("RTL-sim target is not available until RTL simulation backend is provided")
    else:
        if not mapping:
            if hardware_mapping_inputs_ambiguous:
                diagnostics.append("hardware target requires mapping artifact input to be unambiguous")
            else:
                diagnostics.append("hardware target requires mapping artifact")
        if not hardware:
            diagnostics.append("fabric ADG identity is missing")
        if not platform_binding:
            diagnostics.append(f"platform memory binding is missing for {data_movement_policy} data movement")
        diagnostics.append("hardware target is not available until a hardware backend is provided")
    if not workload or workload == "unknown":
        diagnostics.append("workload identity is missing")
    if comparison and string_field(comparison, "status") != "pass":
        diagnostics.append("simulation comparison report is not passing")
    if target == "cgra-sim" and comparison:
        comparison_mapping = string_field(comparison, "mapping_artifact_identity")
        if comparison_mapping and comparison_mapping != artifact_id(mapping_path):
            diagnostics.append("simulation comparison report references a different mapping artifact")
    if data_movement_policy != "simulated":
        if not platform_binding:
            diagnostics.append(f"platform memory binding is missing for {data_movement_policy} data movement")
        if target in {"cgra-sim", "dfg-sim"}:
            diagnostics.append("simulator target requires simulated data movement policy")

    consumed_rtl_manifest_path = rtl_manifest_path if target == "rtl-sim" else None
    package_mapping_id = (
        string_field(mapping_for_identity, "mapping_id")
        if consumed_mapping_path is not None and string_field(mapping_for_identity, "mapping_id")
        else target.replace("-", "_")
    )

    package_id = (
        f"runtime-package::{workload}::{package_mapping_id}" if workload != "unknown" else "runtime-package::blocked"
    )
    work_package_identity = f"work-package::{workload}::{package_mapping_id}" if workload != "unknown" else ""
    host_program_identity = f"test-app-host::{workload}::default" if workload != "unknown" else ""
    host_wrapper_identity = f"runtime-wrapper::{workload}::{package_mapping_id}" if workload != "unknown" else ""
    launch_descriptor_identity = (
        f"launch::{workload}::{package_mapping_id}::{runtime_input}" if workload != "unknown" else ""
    )
    memory_descriptors: list[dict[str, object]] = []
    if workload != "unknown" and runtime_input:
        layout = memory_layout_for_workload(workload)
        if layout is None:
            diagnostics.append(f"runtime input memory layout is missing for {workload}")
        else:
            memory_descriptors.append(
                memory_descriptor(
                    workload=workload,
                    data_movement_policy=data_movement_policy,
                    custom_data_movement_policy=custom_data_movement_policy,
                    runtime_input=runtime_input,
                    platform_binding=platform_binding,
                    layout=layout,
                )
            )

    argument_descriptors = []
    if runtime_input:
        argument_descriptors.append(
            {
                "name": "runtime_input",
                "identity": runtime_input,
                "descriptor_kind": "test_fixture",
            }
        )
    if consumed_mapping_path is not None:
        argument_descriptors.append(
            {
                "name": "mapping_artifact",
                "identity": artifact_id(consumed_mapping_path),
                "descriptor_kind": "pnr_mapping_artifact",
            }
        )
    if consumed_dfg_path is not None:
        argument_descriptors.append(
            {
                "name": "dfg_sim_report",
                "identity": artifact_id(consumed_dfg_path),
                "descriptor_kind": "dfg_sim_report",
            }
        )
    if consumed_cgra_path is not None:
        argument_descriptors.append(
            {
                "name": "cgra_sim_report",
                "identity": artifact_id(consumed_cgra_path),
                "descriptor_kind": "cgra_sim_report",
            }
        )
    if consumed_comparison_path is not None:
        argument_descriptors.append(
            {
                "name": "sim_comparison_report",
                "identity": artifact_id(consumed_comparison_path),
                "descriptor_kind": "sim_comparison_report",
            }
        )
    if consumed_rtl_manifest_path is not None:
        argument_descriptors.append(
            {
                "name": "rtl_manifest",
                "identity": artifact_id(consumed_rtl_manifest_path),
                "descriptor_kind": "rtl_manifest",
            }
        )

    if target == "cgra-sim":
        target_profile = {
            "target_kind": "simulator",
            "simulator": "cgra_sim",
            "profile_id": "simulator::cgra_sim::mapping_constraint_estimate",
        }
        required_runtime_features = [
            "simulator_dispatch",
            "explicit_mapping_artifact",
            fallback_feature(fallback_policy),
        ]
        simulator_report_identities = report_identities([consumed_cgra_path, consumed_comparison_path])
        selected_mapping_identity = artifact_id(consumed_mapping_path)
        fabric_adg_identity = hardware
    elif target == "dfg-sim":
        target_profile = {
            "target_kind": "simulator",
            "simulator": "dfg_sim",
            "profile_id": "simulator::dfg_sim::optimistic_pipeline_latency_throughput_sum",
        }
        required_runtime_features = [
            "dfg_sim_dispatch",
            "software_dataflow_report",
            fallback_feature(fallback_policy),
        ]
        simulator_report_identities = report_identities([consumed_dfg_path])
        selected_mapping_identity = ""
        fabric_adg_identity = ""
    elif target == "rtl-sim":
        target_profile = {
            "target_kind": "simulator",
            "simulator": "rtl_sim",
            "profile_id": "simulator::rtl_sim::generated_hardware",
        }
        required_runtime_features = [
            "rtl_sim_dispatch",
            "explicit_mapping_artifact",
            "rtl_artifact_inputs",
            fallback_feature(fallback_policy),
        ]
        simulator_report_identities = []
        selected_mapping_identity = artifact_id(consumed_mapping_path)
        fabric_adg_identity = hardware
    else:
        target_profile = {
            "target_kind": "hardware",
            "hardware_backend": "physical_accelerator",
            "profile_id": "hardware::physical_accelerator::explicit_platform_binding",
        }
        required_runtime_features = [
            "hardware_dispatch",
            "explicit_mapping_artifact",
            "platform_memory_binding",
            fallback_feature(fallback_policy),
        ]
        simulator_report_identities = []
        selected_mapping_identity = artifact_id(consumed_mapping_path)
        fabric_adg_identity = hardware

    if runtime_trace_enabled:
        required_runtime_features.append("runtime_trace_output")
    if runtime_profiling_enabled:
        required_runtime_features.append("runtime_profiling_output")

    status = "pass" if not diagnostics else "blocked"
    if status == "blocked" and fallback_policy == "require_acceleration":
        diagnostics.append("user-requested acceleration failed")
    fallback = fallback_decision(
        policy=fallback_policy,
        target_profile=target_profile,
        status=status,
        diagnostics=diagnostics,
    )
    launch_descriptor = build_launch_descriptor(
        descriptor_id=launch_descriptor_identity,
        work_package_identity=work_package_identity,
        workload=workload,
        mapping_identity=selected_mapping_identity,
        target_profile=target_profile,
        memory_descriptors=memory_descriptors,
        argument_descriptors=argument_descriptors,
        fallback_policy=fallback_policy,
        synchronization_mode=synchronization_mode,
        runtime_trace_enabled=runtime_trace_enabled,
        runtime_profiling_enabled=runtime_profiling_enabled,
    )
    runtime_report_data = runtime_report(
        workload=workload,
        mapping_id=package_mapping_id,
        host_program_identity=host_program_identity,
        host_wrapper_identity=host_wrapper_identity,
        work_package_identity=work_package_identity,
        launch_descriptor_identity=launch_descriptor_identity,
        selected_mapping_identity=selected_mapping_identity,
        fabric_adg_identity=fabric_adg_identity,
        target_profile=target_profile,
        data_movement_policy=data_movement_policy,
        custom_data_movement_policy=custom_data_movement_policy,
        synchronization_mode=synchronization_mode,
        fallback_policy=fallback_policy,
        fallback=fallback,
        simulator_report_identities=simulator_report_identities,
        source_provenance=runtime_input,
        diagnostics=diagnostics,
    )

    return {
        "schema_version": 1,
        "kind": "runtime_package",
        "package_id": package_id,
        "workload": workload,
        "work_package_identity": work_package_identity,
        "work_package_metadata": work_package_metadata(
            work_package_identity=work_package_identity,
            workload=workload,
            selected_mapping_identity=selected_mapping_identity,
            fabric_adg_identity=fabric_adg_identity,
            runtime_input=runtime_input,
        ),
        "launch_descriptor_identity": launch_descriptor_identity,
        "host_program_identity": host_program_identity,
        "host_wrapper_identity": host_wrapper_identity,
        "host_interface": host_interface(
            host_program_identity=host_program_identity,
            host_wrapper_identity=host_wrapper_identity,
            runtime_input=runtime_input,
        ),
        "launch_descriptor": launch_descriptor,
        "runtime_handle_model": runtime_handle_model(),
        "selected_mapping_artifact_identity": selected_mapping_identity,
        "fabric_adg_identity": fabric_adg_identity,
        "target_profile": target_profile,
        "runtime_configuration": runtime_configuration(
            target_profile=target_profile,
            data_movement_policy=data_movement_policy,
            custom_data_movement_policy=custom_data_movement_policy,
            platform_binding=platform_binding,
            fallback_policy=fallback_policy,
            synchronization_mode=synchronization_mode,
        ),
        "input_artifact_fingerprints": input_artifact_fingerprints(
            [
                consumed_mapping_path,
                consumed_dfg_path,
                consumed_cgra_path,
                consumed_comparison_path,
                consumed_rtl_manifest_path,
            ]
        ),
        "runtime_report": runtime_report_data,
        "report_output_configuration": report_output_configuration(runtime_report_data, launch_descriptor),
        "fallback_policy": fallback_policy,
        "fallback_decision": fallback,
        "synchronization_mode": synchronization_mode,
        "data_movement_policy": data_movement_policy,
        "memory_descriptors": memory_descriptors,
        "argument_descriptors": argument_descriptors,
        "required_runtime_features": required_runtime_features,
        "required_data_movement_policies": [data_movement_policy],
        "required_synchronization_policies": [synchronization_mode],
        "simulator_report_identities": simulator_report_identities,
        "diagnostic_records": diagnostic_records(
            diagnostics,
            source_provenance=runtime_input,
            host_wrapper_identity=host_wrapper_identity,
        ),
        "diagnostics": diagnostics,
        "status": status,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    paths = [Path(value) for value in args.artifact]
    package = build_package(
        paths,
        args.target,
        args.data_movement_policy,
        args.custom_data_movement_policy,
        args.platform_binding,
        args.fallback_policy,
        args.synchronization_mode,
        args.enable_runtime_trace,
        args.enable_runtime_profiling,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    return 0 if package["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
