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


def artifact_id(path: Path) -> str:
    for suffix in (".csv", ".json"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


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


def report_bundle_ids(paths: list[Path], expected_kind: str) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    diagnostics: list[str] = []
    for path in paths:
        data = read_json(path)
        if data.get("kind") != expected_kind:
            diagnostics.append(f"{path} is not a {expected_kind}")
            continue
        if data.get("report_status") != "pass":
            diagnostics.append(f"{artifact_id(path)} is not a passing report bundle")
            continue
        ids.append(artifact_id(path))
    return ids, diagnostics


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
        required_data_movement_policies = evidence.get("required_data_movement_policies", [])
        if not isinstance(required_data_movement_policies, list):
            required_data_movement_policies = []
        required_synchronization_policies = evidence.get("required_synchronization_policies", [])
        if not isinstance(required_synchronization_policies, list):
            required_synchronization_policies = []
        summaries.append(
            {
                "workload_report_bundle_identity": artifact_id(path),
                "runtime_package_identity": str(evidence.get("runtime_package_identity", "")),
                "runtime_report_identity": str(evidence.get("runtime_report_identity", "")),
                "launch_status": str(evidence.get("launch_status", "")),
                "target_status": str(evidence.get("target_status", "")),
                "data_movement_policy": str(evidence.get("data_movement_policy", "")),
                "synchronization_mode": str(evidence.get("synchronization_mode", "")),
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
                "input_artifact_fingerprints": {
                    str(identity): str(fingerprint)
                    for identity, fingerprint in input_fingerprints.items()
                    if isinstance(identity, str) and isinstance(fingerprint, str)
                },
                "fallback_decision": fallback,
            }
        )
    return summaries


def build_bundle(paths: list[Path]) -> dict[str, object]:
    grouped = group_paths(paths)
    selected = selected_candidate_row(grouped.get("dse_candidate", []))
    workload_report_ids, workload_diagnostics = report_bundle_ids(
        grouped.get("workload_report_bundle", []),
        "workload_report_bundle",
    )
    hardware_report_ids, hardware_diagnostics = report_bundle_ids(
        grouped.get("hardware_report_bundle", []),
        "hardware_report_bundle",
    )
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
            "referenced_workload_report_bundle_identities": workload_report_ids,
            "referenced_hardware_candidate_report_bundle_identities": hardware_report_ids,
            "runtime_evidence_summaries": runtime_evidence_summaries(grouped.get("workload_report_bundle", [])),
            "selected_policy_id": "",
            "policy_configuration": {},
            "candidate_ordering_rule": "",
            "report_status": "blocked",
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
    return {
        "schema_version": 1,
        "kind": "dse_report_bundle",
        "dse_run_id": f"dse::{policy_id}" if policy_id else "dse::blocked",
        "objective_records": [objective_record(selected_row)],
        "candidate_list": candidates,
        "selected_candidates": selected_candidates,
        "pareto_set": pareto_set,
        "rejected_candidate_summaries": rejected,
        "referenced_workload_report_bundle_identities": workload_report_ids,
        "referenced_hardware_candidate_report_bundle_identities": hardware_report_ids,
        "runtime_evidence_summaries": runtime_evidence_summaries(grouped.get("workload_report_bundle", [])),
        "selected_policy_id": policy_id,
        "policy_configuration": {
            "policy_kind": "deterministic",
            "random_seed": None,
            "conflict_resolution": "candidate_ordering_rule",
        },
        "candidate_ordering_rule": selected_row.get("ordering_rule", ""),
        "report_status": "pass" if not diagnostics else "blocked",
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
