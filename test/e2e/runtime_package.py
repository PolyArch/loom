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


artifact_id = intermediate_artifacts.artifact_id_for_path
input_artifact_fingerprints = intermediate_artifacts.input_artifact_fingerprints


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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("cgra-sim", "dfg-sim", "rtl-sim", "hardware"), default="cgra-sim")
    parser.add_argument("--data-movement-policy", choices=DATA_MOVEMENT_POLICIES, default="simulated")
    parser.add_argument("--custom-data-movement-policy", default="")
    parser.add_argument("--fallback-policy", choices=FALLBACK_POLICIES, default="report_only")
    parser.add_argument("--synchronization-mode", choices=SYNCHRONIZATION_MODES, default="host_wait")
    parser.add_argument("--platform-binding", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    return parser.parse_args(argv)


def read_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text())


def group_paths(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(intermediate_artifacts.artifact_kind_for_path(path), []).append(path)
    return grouped


def first_path(grouped: dict[str, list[Path]], kind: str) -> Path | None:
    paths = grouped.get(kind, [])
    return paths[0] if paths else None


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
        "scalar_value_descriptors": [],
        "selected_mapping_artifact_identity": mapping_identity,
        "target_profile_id": target_profile.get("profile_id", ""),
        "fallback_policy": fallback_policy,
        "synchronization_mode": synchronization_mode,
        "profiling_settings": {"enabled": False},
        "trace_settings": {"enabled": False},
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
    configuration = {
        "configuration_id": f"runtime-config::{fallback_policy}::{data_movement_policy}::{synchronization_mode}",
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
    if workload == "vecsum":
        return {
            "byte_size": 256,
            "element_layout": "u32[64]",
            "alignment_bytes": 4,
        }
    return None


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


def build_package(
    paths: list[Path],
    target: str,
    data_movement_policy: str,
    custom_data_movement_policy: str,
    platform_binding: str,
    fallback_policy: str,
    synchronization_mode: str,
) -> dict[str, object]:
    grouped = group_paths(paths)
    mapping_path = first_path(grouped, "pnr_mapping_artifact")
    dfg_path = first_path(grouped, "dfg_sim_report")
    cgra_path = first_path(grouped, "cgra_sim_report")
    comparison_path = first_path(grouped, "sim_comparison_report")

    mapping = read_json(mapping_path)
    dfg = read_json(dfg_path)
    cgra = read_json(cgra_path)
    comparison = read_json(comparison_path)
    workload = (
        string_field(mapping, "workload")
        or string_field(dfg, "workload")
        or string_field(cgra, "workload")
        or "unknown"
    )
    hardware = string_field(mapping, "hardware") or string_field(cgra, "hardware")
    mapping_id = string_field(mapping, "mapping_id") or target.replace("-", "_")
    runtime_input = runtime_input_identity(workload, comparison, dfg)

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
        diagnostics.append("RTL-sim target requires RTL simulation artifacts")
        diagnostics.append("RTL-sim target is not available until RTL artifacts are provided")
    else:
        if not mapping:
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

    package_id = f"runtime-package::{workload}::{mapping_id}" if workload != "unknown" else "runtime-package::blocked"
    work_package_identity = f"work-package::{workload}::{mapping_id}" if workload != "unknown" else ""
    host_program_identity = f"test-app-host::{workload}::default" if workload != "unknown" else ""
    host_wrapper_identity = f"runtime-wrapper::{workload}::{mapping_id}" if workload != "unknown" else ""
    launch_descriptor_identity = (
        f"launch::{workload}::{mapping_id}::{runtime_input}" if workload != "unknown" else ""
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
    if mapping_path is not None:
        argument_descriptors.append(
            {
                "name": "mapping_artifact",
                "identity": artifact_id(mapping_path),
                "descriptor_kind": "pnr_mapping_artifact",
            }
        )
    if dfg_path is not None:
        argument_descriptors.append(
            {
                "name": "dfg_sim_report",
                "identity": artifact_id(dfg_path),
                "descriptor_kind": "dfg_sim_report",
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
        simulator_report_identities = report_identities([cgra_path, comparison_path])
        selected_mapping_identity = artifact_id(mapping_path)
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
        simulator_report_identities = report_identities([dfg_path])
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
        selected_mapping_identity = artifact_id(mapping_path)
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
        selected_mapping_identity = artifact_id(mapping_path)
        fabric_adg_identity = hardware

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
            [mapping_path, dfg_path, cgra_path, comparison_path]
        ),
        "runtime_report": runtime_report(
            workload=workload,
            mapping_id=mapping_id,
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
        ),
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
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    return 0 if package["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
