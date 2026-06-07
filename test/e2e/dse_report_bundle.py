#!/usr/bin/env python3
"""Emit DSE report bundles from candidate and report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


artifact_id = intermediate_artifacts.artifact_id_for_path
artifact_fingerprint = intermediate_artifacts.artifact_fingerprint
input_artifact_fingerprints = intermediate_artifacts.input_artifact_fingerprints


METRIC_ID_BY_NAME = {
    "cgra_sim_cycles": "metric::{workload}::cgra_sim_cycles",
    "frequency_mhz": "metric::{hardware}::frequency_mhz",
    "area_um2": "metric::{hardware}::area_um2",
    "dynamic_power_mw": "metric::{hardware}::dynamic_power_mw",
    "energy_nj": "metric::{workload}::energy_nj",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def group_paths(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(intermediate_artifacts.artifact_kind_for_path(path), []).append(path)
    return grouped


def selected_candidate_row(paths: list[Path]) -> tuple[Path, dict[str, str]] | None:
    for path in paths:
        for row in read_csv(path):
            if row.get("selection_status") == "selected":
                return path, row
    return None


def parse_metric_names(raw: str) -> list[str]:
    names: list[str] = []
    for entry in raw.split(";"):
        if not entry or "=" not in entry:
            continue
        name, _ = entry.split("=", 1)
        if name:
            names.append(name)
    return names


def metric_ids_for_candidate(row: dict[str, str]) -> list[str]:
    workload = row.get("workload", "")
    hardware = row.get("hardware", "")
    ids: list[str] = []
    for name in parse_metric_names(row.get("metric_records", "")):
        template = METRIC_ID_BY_NAME.get(name)
        if template is None:
            continue
        ids.append(template.format(workload=workload, hardware=hardware))
    return ids


def semicolon_list(raw: str) -> list[str]:
    return [entry for entry in raw.split(";") if entry]


def semicolon_map(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in raw.split(";"):
        if not entry:
            continue
        if "=" not in entry:
            continue
        key, value = entry.rsplit("=", 1)
        if key and value:
            parsed[key] = value
    return parsed


def diagnostic_class(message: str) -> str:
    if "selected DSE candidate" in message:
        return "dse_candidate_missing"
    if "workload report bundle" in message:
        return "workload_report_bundle_failure"
    if "hardware report bundle" in message:
        return "hardware_report_bundle_failure"
    return "dse_report_bundle_failure"


def diagnostic_records(diagnostics: list[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for index, message in enumerate(diagnostics, start=1):
        records.append(
            {
                "diagnostic_id": f"dse-report-bundle::{index}",
                "diagnostic_class": diagnostic_class(message),
                "component": "dse_report_bundle",
                "severity": "error",
                "message": message,
            }
        )
    return records


def objective_record(row: dict[str, str]) -> dict[str, object]:
    objective = row.get("objective", "")
    objective_id = row.get("objective_record", "") or f"objective::{objective}"
    metric_inputs = metric_ids_for_candidate(row)
    if objective == "minimize_runtime":
        metric_inputs = [metric for metric in metric_inputs if metric.endswith("::cgra_sim_cycles")]
        direction = "minimize"
        units = "cycles"
    elif objective in {"minimize_energy", "minimize_power"}:
        metric_inputs = [metric for metric in metric_inputs if metric.endswith("::energy_nj")]
        direction = "minimize"
        units = "nJ"
    else:
        direction = "minimize"
        units = "score"
    return {
        "objective_id": objective_id,
        "objective_kind": objective,
        "metric_inputs": metric_inputs,
        "priority": 1,
        "constraint_or_optimization_mode": "optimization",
        "comparison_direction": direction,
        "units": units,
        "validity_conditions": ["candidate metric records are present"],
    }


def candidate_record(row: dict[str, str]) -> dict[str, object]:
    return {
        "candidate_id": row.get("candidate", ""),
        "candidate_kind": row.get("candidate_kind", ""),
        "parent_candidate_ids": [],
        "referenced_input_artifacts": semicolon_list(row.get("input_artifacts", "")),
        "input_artifact_fingerprints": semicolon_map(row.get("input_artifact_fingerprints", "")),
        "generated_output_artifacts": semicolon_list(row.get("output_artifacts", "")),
        "objective_records_used": [row.get("objective_record", "")],
        "metric_records_used": metric_ids_for_candidate(row),
        "status": row.get("selection_status", "blocked"),
        "diagnostics": [row.get("diagnostic", "")] if row.get("diagnostic", "") else [],
    }


def report_bundle_references(paths: list[Path], expected_kind: str) -> tuple[list[str], dict[str, str], list[str]]:
    ids: list[str] = []
    fingerprints: dict[str, str] = {}
    diagnostics: list[str] = []
    for path in paths:
        data = read_json(path)
        if data.get("kind") != expected_kind:
            diagnostics.append(f"{path} is not a {expected_kind}")
            continue
        if data.get("report_status") != "pass":
            diagnostics.append(f"{artifact_id(path)} is not a passing report bundle")
            continue
        identity = artifact_id(path)
        ids.append(identity)
        fingerprints[identity] = artifact_fingerprint(path)
    return ids, fingerprints, diagnostics


def input_artifact_references(paths: list[Path]) -> tuple[list[str], dict[str, str]]:
    ids: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        ids.append(artifact_id(path))
    return ids, input_artifact_fingerprints(paths)


def runtime_evidence_summaries(paths: list[Path]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for path in paths:
        data = read_json(path)
        if data.get("kind") != "workload_report_bundle" or data.get("report_status") != "pass":
            continue
        evidence = data.get("runtime_evidence", {})
        if not isinstance(evidence, dict):
            continue
        fallback = evidence.get("fallback_decision", {})
        if not isinstance(fallback, dict):
            fallback = {}
        input_fingerprints = evidence.get("input_artifact_fingerprints", {})
        if not isinstance(input_fingerprints, dict):
            input_fingerprints = {}
        output_buffers = evidence.get("output_buffer_identities", [])
        if not isinstance(output_buffers, list):
            output_buffers = []
        simulator_reports = evidence.get("simulator_report_identities", [])
        if not isinstance(simulator_reports, list):
            simulator_reports = []
        diagnostic_records = evidence.get("diagnostic_records", [])
        if not isinstance(diagnostic_records, list):
            diagnostic_records = []
        work_package_metadata = evidence.get("work_package_metadata", {})
        if not isinstance(work_package_metadata, dict):
            work_package_metadata = {}
        report_output_configuration = evidence.get("report_output_configuration", {})
        if not isinstance(report_output_configuration, dict):
            report_output_configuration = {}
        memory_descriptors = evidence.get("memory_descriptors", [])
        if not isinstance(memory_descriptors, list):
            memory_descriptors = []
        argument_descriptors = evidence.get("argument_descriptors", [])
        if not isinstance(argument_descriptors, list):
            argument_descriptors = []
        required_data_movement_policies = evidence.get("required_data_movement_policies", [])
        if not isinstance(required_data_movement_policies, list):
            required_data_movement_policies = []
        required_synchronization_policies = evidence.get("required_synchronization_policies", [])
        if not isinstance(required_synchronization_policies, list):
            required_synchronization_policies = []
        summary = {
            "workload_report_bundle_identity": artifact_id(path),
            "runtime_package_identity": str(evidence.get("runtime_package_identity", "")),
            "runtime_report_identity": str(evidence.get("runtime_report_identity", "")),
            "host_program_identity": str(evidence.get("host_program_identity", "")),
            "host_wrapper_identity": str(evidence.get("host_wrapper_identity", "")),
            "runtime_handle_model": evidence.get("runtime_handle_model", {}),
            "work_package_metadata": work_package_metadata,
            "work_package_identity": str(evidence.get("work_package_identity", "")),
            "launch_descriptor_identity": str(evidence.get("launch_descriptor_identity", "")),
            "mapping_artifact_identity": str(evidence.get("mapping_artifact_identity", "")),
            "fabric_adg_identity": str(evidence.get("fabric_adg_identity", "")),
            "target_profile_id": str(evidence.get("target_profile_id", "")),
            "fallback_policy": str(evidence.get("fallback_policy", "")),
            "launch_status": str(evidence.get("launch_status", "")),
            "target_status": str(evidence.get("target_status", "")),
            "runtime_trace_identity": str(evidence.get("runtime_trace_identity", "")),
            "profiling_record_identity": str(evidence.get("profiling_record_identity", "")),
            "data_movement_policy": str(evidence.get("data_movement_policy", "")),
            "synchronization_mode": str(evidence.get("synchronization_mode", "")),
            "memory_descriptors": [
                descriptor
                for descriptor in memory_descriptors
                if isinstance(descriptor, dict)
            ],
            "argument_descriptors": [
                descriptor
                for descriptor in argument_descriptors
                if isinstance(descriptor, dict)
            ],
            "required_data_movement_policies": [
                str(policy)
                for policy in required_data_movement_policies
                if isinstance(policy, str)
            ],
            "required_synchronization_policies": [
                str(policy)
                for policy in required_synchronization_policies
                if isinstance(policy, str)
            ],
            "simulator_report_identities": [
                str(identity)
                for identity in simulator_reports
                if isinstance(identity, str)
            ],
            "input_artifact_fingerprints": {
                str(identity): str(fingerprint)
                for identity, fingerprint in input_fingerprints.items()
                if isinstance(identity, str) and isinstance(fingerprint, str)
            },
            "output_buffer_identities": [
                str(identity)
                for identity in output_buffers
                if isinstance(identity, str)
            ],
            "diagnostic_records": [
                record
                for record in diagnostic_records
                if isinstance(record, dict)
            ],
            "report_output_configuration": report_output_configuration,
            "fallback_decision": fallback,
        }
        custom_policy = evidence.get("custom_data_movement_policy_identity")
        if isinstance(custom_policy, str) and custom_policy:
            summary["custom_data_movement_policy_identity"] = custom_policy
        summaries.append(summary)
    return summaries


def build_bundle(paths: list[Path]) -> dict[str, object]:
    grouped = group_paths(paths)
    selected = selected_candidate_row(grouped.get("dse_candidate", []))
    candidate_ids, candidate_fingerprints = input_artifact_references(grouped.get("dse_candidate", []))
    workload_report_ids, workload_fingerprints, workload_diagnostics = report_bundle_references(
        grouped.get("workload_report_bundle", []),
        "workload_report_bundle",
    )
    hardware_report_ids, hardware_fingerprints, hardware_diagnostics = report_bundle_references(
        grouped.get("hardware_report_bundle", []),
        "hardware_report_bundle",
    )
    input_artifact_fingerprints = {
        **candidate_fingerprints,
        **workload_fingerprints,
        **hardware_fingerprints,
    }
    diagnostics = workload_diagnostics + hardware_diagnostics
    if selected is None:
        diagnostics.append("no selected DSE candidate row was provided")
        return {
            "schema_version": 1,
            "kind": "dse_report_bundle",
            "dse_run_id": "dse::blocked",
            "objective_records": [],
            "candidate_list": [],
            "selected_candidates": [],
            "pareto_set": [],
            "rejected_candidate_summaries": [],
            "referenced_dse_candidate_artifact_identities": candidate_ids,
            "referenced_workload_report_bundle_identities": workload_report_ids,
            "referenced_hardware_candidate_report_bundle_identities": hardware_report_ids,
            "input_artifact_fingerprints": input_artifact_fingerprints,
            "runtime_evidence_summaries": runtime_evidence_summaries(grouped.get("workload_report_bundle", [])),
            "selected_policy_id": "",
            "policy_configuration": {},
            "candidate_ordering_rule": "",
            "report_status": "blocked",
            "diagnostic_records": diagnostic_records(diagnostics),
            "diagnostics": diagnostics,
        }

    _, selected_row = selected
    candidates: list[dict[str, object]] = []
    selected_candidates: list[str] = []
    pareto_set: list[str] = []
    rejected: list[dict[str, object]] = []
    for path in grouped.get("dse_candidate", []):
        for row in read_csv(path):
            status = row.get("selection_status", "blocked")
            record = candidate_record(row)
            candidates.append(record)
            candidate_id = row.get("candidate", "")
            if status == "selected" and candidate_id:
                selected_candidates.append(candidate_id)
            elif status == "pareto" and candidate_id:
                pareto_set.append(candidate_id)
            elif status == "rejected" and candidate_id:
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "diagnostics": [row.get("diagnostic", "")] if row.get("diagnostic", "") else [],
                    }
                )

    if not workload_report_ids:
        diagnostics.append("no passing workload report bundle was provided")
    if not hardware_report_ids:
        diagnostics.append("no passing hardware report bundle was provided")
    if not selected_candidates:
        diagnostics.append("no selected candidate identity was found")

    policy_id = selected_row.get("policy_id", "")
    status = "pass" if not diagnostics else "blocked"
    return {
        "schema_version": 1,
        "kind": "dse_report_bundle",
        "dse_run_id": f"dse::{policy_id}" if policy_id else "dse::blocked",
        "objective_records": [objective_record(selected_row)],
        "candidate_list": candidates,
        "selected_candidates": selected_candidates,
        "pareto_set": pareto_set,
        "rejected_candidate_summaries": rejected,
        "referenced_dse_candidate_artifact_identities": candidate_ids,
        "referenced_workload_report_bundle_identities": workload_report_ids,
        "referenced_hardware_candidate_report_bundle_identities": hardware_report_ids,
        "input_artifact_fingerprints": input_artifact_fingerprints,
        "runtime_evidence_summaries": runtime_evidence_summaries(grouped.get("workload_report_bundle", [])),
        "selected_policy_id": policy_id,
        "policy_configuration": {
            "policy_kind": "deterministic",
            "random_seed": None,
            "conflict_resolution": "candidate_ordering_rule",
        },
        "candidate_ordering_rule": selected_row.get("ordering_rule", ""),
        "report_status": status,
        "diagnostic_records": diagnostic_records(diagnostics),
        "diagnostics": diagnostics,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    paths = [Path(value) for value in args.artifact]
    bundle = build_bundle(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return 0 if bundle["report_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
