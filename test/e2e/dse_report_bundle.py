#!/usr/bin/env python3
"""Emit DSE report bundles from candidate and report artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402
import artifact_io_helpers  # noqa: E402
import dse_objectives  # noqa: E402
import runtime_evidence_helpers  # noqa: E402


artifact_id = intermediate_artifacts.artifact_id_for_path
artifact_fingerprint = intermediate_artifacts.artifact_fingerprint
input_artifact_fingerprints = intermediate_artifacts.input_artifact_fingerprints
read_csv = artifact_io_helpers.read_csv
read_json = artifact_io_helpers.read_json
group_paths = artifact_io_helpers.group_paths


METRIC_ID_BY_NAME = {
    "cgra_sim_cycles": "metric::{workload}::cgra_sim_cycles",
    "frequency_mhz": "metric::{hardware}::frequency_mhz",
    "area_um2": "metric::{hardware}::area_um2",
    "dynamic_power_mw": "metric::{hardware}::dynamic_power_mw",
    "leakage_power_mw": "metric::{hardware}::leakage_power_mw",
    "energy_nj": "metric::{workload}::energy_nj",
    "unsupported_scope_diagnostics_count": (
        "metric::{workload}::{hardware}::{mapping_id}::unsupported_scope_diagnostics_count"
    ),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    return parser.parse_args(argv)


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
    mapping_id = row.get("mapping_id", "")
    ids: list[str] = []
    for name in parse_metric_names(row.get("metric_records", "")):
        template = METRIC_ID_BY_NAME.get(name)
        if template is None:
            continue
        ids.append(template.format(workload=workload, hardware=hardware, mapping_id=mapping_id))
    return ids


def workload_metric_ids_by_workload(paths: list[Path]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for path in paths:
        data = read_json(path)
        if data.get("kind") != "workload_report_bundle" or data.get("report_status") != "pass":
            continue
        workload = data.get("workload")
        metrics = data.get("metric_records")
        if not isinstance(workload, str) or not workload or not isinstance(metrics, list):
            continue
        ids = grouped.setdefault(workload, [])
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            metric_id = metric.get("metric_id")
            if isinstance(metric_id, str) and metric_id and metric_id not in ids:
                ids.append(metric_id)
    return grouped


def semicolon_list(raw: str) -> list[str]:
    return [entry for entry in raw.split(";") if entry]


def artifact_identity(reference: str) -> str:
    return artifact_id(Path(reference)) if reference else ""


def semicolon_identity_list(raw: str) -> list[str]:
    return [identity for identity in (artifact_identity(entry) for entry in raw.split(";") if entry) if identity]


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


def semicolon_identity_map(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for reference, fingerprint in semicolon_map(raw).items():
        identity = artifact_identity(reference)
        if identity:
            parsed[identity] = fingerprint
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
    spec = dse_objectives.objective_spec(objective)
    if spec is not None:
        metric_id = dse_objectives.metric_id_for_objective(
            objective,
            row.get("workload", ""),
            row.get("hardware", ""),
            row.get("mapping_id", ""),
        )
        metric_inputs = [metric_id] if metric_id is not None else []
        direction = spec.direction
        units = spec.units
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
        "referenced_input_artifacts": semicolon_identity_list(row.get("input_artifacts", "")),
        "input_artifact_fingerprints": semicolon_identity_map(row.get("input_artifact_fingerprints", "")),
        "generated_output_artifacts": semicolon_identity_list(row.get("output_artifacts", "")),
        "objective_records_used": [row.get("objective_record", "")],
        "metric_records_used": metric_ids_for_candidate(row),
        "status": row.get("selection_status", "blocked"),
        "diagnostics": [row.get("diagnostic", "")] if row.get("diagnostic", "") else [],
    }


def report_matches_selected_candidate(
    data: dict[str, object],
    selected_row: dict[str, str] | None,
    expected_kind: str,
) -> bool:
    if selected_row is None:
        return True
    workload = selected_row.get("workload", "")
    hardware = selected_row.get("hardware", "")
    if expected_kind == "workload_report_bundle":
        report_workload = data.get("workload")
        report_hardware = data.get("selected_hardware_candidate_identity")
        if isinstance(report_workload, str) and report_workload != workload:
            return False
        if isinstance(report_hardware, str) and report_hardware:
            return artifact_io_helpers.hardware_matches(report_hardware, hardware)
        return True
    if expected_kind == "hardware_report_bundle":
        for key in ("hardware_candidate_identity", "fabric_adg_identity"):
            report_hardware = data.get(key)
            if isinstance(report_hardware, str) and artifact_io_helpers.hardware_matches(report_hardware, hardware):
                return True
        return False
    return True


def report_bundle_references(
    paths: list[Path],
    expected_kind: str,
    selected_row: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str], list[Path], list[str]]:
    ids: list[str] = []
    fingerprints: dict[str, str] = {}
    selected_paths: list[Path] = []
    diagnostics: list[str] = []
    for path in paths:
        data = read_json(path)
        if data.get("kind") != expected_kind:
            diagnostics.append(f"{artifact_id(path)} is not a {expected_kind}")
            continue
        if data.get("report_status") != "pass":
            diagnostics.append(f"{artifact_id(path)} is not a passing report bundle")
            continue
        if not report_matches_selected_candidate(data, selected_row, expected_kind):
            continue
        identity = artifact_id(path)
        ids.append(identity)
        fingerprints[identity] = artifact_fingerprint(path)
        selected_paths.append(path)
    return ids, fingerprints, selected_paths, diagnostics


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
        summaries.append(
            runtime_evidence_helpers.runtime_evidence_summary(
                evidence,
                artifact_id(path),
            )
        )
    return summaries


def build_bundle(paths: list[Path]) -> dict[str, object]:
    grouped = group_paths(paths)
    selected = selected_candidate_row(grouped.get("dse_candidate", []))
    selected_row = selected[1] if selected is not None else None
    candidate_ids, candidate_fingerprints = input_artifact_references(grouped.get("dse_candidate", []))
    workload_report_ids, workload_fingerprints, workload_report_paths, workload_diagnostics = report_bundle_references(
        grouped.get("workload_report_bundle", []),
        "workload_report_bundle",
        selected_row,
    )
    hardware_report_ids, hardware_fingerprints, _, hardware_diagnostics = report_bundle_references(
        grouped.get("hardware_report_bundle", []),
        "hardware_report_bundle",
        selected_row,
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
            "runtime_evidence_summaries": runtime_evidence_summaries(workload_report_paths),
            "selected_policy_id": "",
            "policy_configuration": {},
            "candidate_ordering_rule": "",
            "report_status": "blocked",
            "diagnostic_records": diagnostic_records(diagnostics),
            "diagnostics": diagnostics,
        }

    workload_metric_ids = workload_metric_ids_by_workload(grouped.get("workload_report_bundle", []))
    candidates: list[dict[str, object]] = []
    selected_candidates: list[str] = []
    pareto_set: list[str] = []
    rejected: list[dict[str, object]] = []
    for path in grouped.get("dse_candidate", []):
        for row in read_csv(path):
            status = row.get("selection_status", "blocked")
            record = candidate_record(row)
            metrics_used = record.get("metric_records_used")
            if isinstance(metrics_used, list):
                for metric_id in workload_metric_ids.get(row.get("workload", ""), []):
                    if metric_id not in metrics_used:
                        metrics_used.append(metric_id)
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
        "runtime_evidence_summaries": runtime_evidence_summaries(workload_report_paths),
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
