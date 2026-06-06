#!/usr/bin/env python3
"""Emit workload report bundles from full-stack intermediate artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


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


def artifact_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_artifact_fingerprints(paths: list[Path | None]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for path in paths:
        if path is None or not path.is_file():
            continue
        fingerprints[artifact_id(path)] = artifact_fingerprint(path)
    return fingerprints


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


def first_path(grouped: dict[str, list[Path]], kind: str) -> Path | None:
    paths = grouped.get(kind, [])
    return paths[0] if paths else None


def selected_dse_row(paths: list[Path]) -> dict[str, str] | None:
    for path in paths:
        for row in read_csv(path):
            if row.get("selection_status") == "selected":
                return row
    return None


def matching_row(paths: list[Path], key: str, value: str) -> dict[str, str] | None:
    for path in paths:
        for row in read_csv(path):
            if row.get(key) == value:
                return row
    return None


def hardware_matches(candidate: str, hardware: str) -> bool:
    return candidate == hardware or candidate.rsplit("::", 1)[-1] == hardware


def matching_rtl_fpa_row(paths: list[Path], workload: str, hardware: str) -> dict[str, str] | None:
    for path in paths:
        for row in read_csv(path):
            if row.get("workload") == workload and hardware_matches(row.get("hardware", ""), hardware):
                return row
    return None


def numeric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def metric_record(
    *,
    metric_id: str,
    metric_class: str,
    value: float | int,
    unit: str,
    fidelity_level: str,
    evidence_source_artifact_id: str,
    producer_component: str,
    derivation_kind: str,
    diagnostics: list[str] | None = None,
    input_metric_ids: list[str] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "metric_id": metric_id,
        "metric_class": metric_class,
        "value": value,
        "unit": unit,
        "fidelity_level": fidelity_level,
        "evidence_source_artifact_id": evidence_source_artifact_id,
        "producer_component": producer_component,
        "derivation_kind": derivation_kind,
        "diagnostics": diagnostics or [],
    }
    if input_metric_ids is not None:
        record["input_metric_ids"] = input_metric_ids
    return record


def diagnostic_class(message: str) -> str:
    if "runtime package is not passing" in message:
        return "runtime_package_failure"
    if "source compatibility row is missing" in message:
        return "source_compatibility_missing"
    if "compiler pipeline row is missing" in message:
        return "compiler_pipeline_missing"
    if "DFG-sim report is missing" in message:
        return "dfg_sim_missing"
    if "CGRA-sim report is missing" in message:
        return "cgra_sim_missing"
    if "simulation comparison report is not passing" in message:
        return "simulation_comparison_failure"
    if "RTL/FPA row is missing" in message:
        return "rtl_fpa_missing"
    if "no selected DSE candidate artifact was provided" in message:
        return "dse_candidate_missing"
    return "report_bundle_failure"


def diagnostic_record(index: int, message: str) -> dict[str, str]:
    return {
        "diagnostic_id": f"workload-report-bundle::{index}",
        "diagnostic_class": diagnostic_class(message),
        "component": "workload_report_bundle",
        "severity": "error",
        "message": message,
    }


def runtime_diagnostic_records(runtime_package: dict[str, object]) -> list[dict[str, object]]:
    records = runtime_package.get("diagnostic_records", [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def runtime_evidence(runtime_package: dict[str, object], runtime_path: Path | None) -> dict[str, object]:
    report = runtime_package.get("runtime_report", {})
    if not isinstance(report, dict):
        report = {}
    fallback = report.get("fallback_decision")
    if not isinstance(fallback, dict):
        fallback = runtime_package.get("fallback_decision", {})
    if not isinstance(fallback, dict):
        fallback = {}
    output_buffers = report.get("output_buffer_identities", [])
    if not isinstance(output_buffers, list):
        output_buffers = []
    input_fingerprints = runtime_package.get("input_artifact_fingerprints", {})
    if not isinstance(input_fingerprints, dict):
        input_fingerprints = {}
    required_data_movement_policies = runtime_package.get("required_data_movement_policies", [])
    if not isinstance(required_data_movement_policies, list):
        required_data_movement_policies = []
    required_synchronization_policies = runtime_package.get("required_synchronization_policies", [])
    if not isinstance(required_synchronization_policies, list):
        required_synchronization_policies = []
    return {
        "runtime_package_identity": artifact_id(runtime_path) if runtime_path is not None else "",
        "runtime_report_identity": str(report.get("report_id", "")),
        "launch_status": str(report.get("launch_status", "")),
        "target_status": str(report.get("target_status", "")),
        "runtime_trace_identity": str(report.get("runtime_trace_identity", "")),
        "profiling_record_identity": str(report.get("profiling_record_identity", "")),
        "data_movement_policy": str(runtime_package.get("data_movement_policy", "")),
        "synchronization_mode": str(runtime_package.get("synchronization_mode", "")),
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
        "output_buffer_identities": [
            str(identity)
            for identity in output_buffers
            if isinstance(identity, str)
        ],
        "fallback_decision": fallback,
    }


def build_bundle(paths: list[Path]) -> dict[str, object]:
    grouped = group_paths(paths)
    dse_path = first_path(grouped, "dse_candidate")
    dse_row = selected_dse_row(grouped.get("dse_candidate", []))
    if dse_row is None or dse_path is None:
        return {
            "schema_version": 1,
            "kind": "workload_report_bundle",
            "bundle_id": "workload::blocked",
            "workload": "unknown",
            "source_artifact_identity": "",
            "compiler_command_identity": "",
            "runtime_input_identity": "",
            "selected_hardware_candidate_identity": "",
            "selected_mapping_artifact_identity": "",
            "runtime_host_interface": {},
            "runtime_fallback_decision": {},
            "runtime_evidence": {},
            "input_artifact_fingerprints": {},
            "report_status": "blocked",
            "diagnostic_records": [
                diagnostic_record(1, "no selected DSE candidate artifact was provided")
            ],
            "diagnostics": ["no selected DSE candidate artifact was provided"],
            "metric_records": [],
        }

    workload = dse_row["workload"]
    hardware = dse_row["hardware"]
    mapping_id = dse_row["mapping_id"]
    source_path = first_path(grouped, "source_compat")
    compiler_path = first_path(grouped, "compiler_pipeline")
    mapping_path = first_path(grouped, "pnr_mapping_artifact")
    dfg_path = first_path(grouped, "dfg_sim_report")
    cgra_path = first_path(grouped, "cgra_sim_report")
    comparison_path = first_path(grouped, "sim_comparison_report")
    runtime_path = first_path(grouped, "runtime_package")
    rtl_path = first_path(grouped, "rtl_fpa")

    dfg_report = read_json(dfg_path) if dfg_path is not None else {}
    cgra_report = read_json(cgra_path) if cgra_path is not None else {}
    comparison_report = read_json(comparison_path) if comparison_path is not None else {}
    runtime_package = read_json(runtime_path) if runtime_path is not None else {}
    runtime_fallback_decision = runtime_package.get("fallback_decision", {})
    if not isinstance(runtime_fallback_decision, dict):
        runtime_fallback_decision = {}
    runtime_host_interface = runtime_package.get("host_interface", {})
    if not isinstance(runtime_host_interface, dict):
        runtime_host_interface = {}
    rtl_row = matching_rtl_fpa_row(grouped.get("rtl_fpa", []), workload, hardware)
    source_row = matching_row(grouped.get("source_compat", []), "case", workload)
    compiler_row = matching_row(grouped.get("compiler_pipeline", []), "case", workload)

    diagnostics: list[str] = []
    if source_row is None:
        diagnostics.append("source compatibility row is missing")
    if compiler_row is None:
        diagnostics.append("compiler pipeline row is missing")
    if not dfg_report:
        diagnostics.append("DFG-sim report is missing")
    if not cgra_report:
        diagnostics.append("CGRA-sim report is missing")
    if comparison_report and comparison_report.get("status") != "pass":
        diagnostics.append("simulation comparison report is not passing")
    if runtime_package and runtime_package.get("status") != "pass":
        diagnostics.append("runtime package is not passing")
    if rtl_row is None or rtl_path is None:
        diagnostics.append("RTL/FPA row is missing")
    diagnostic_records = runtime_diagnostic_records(runtime_package)
    diagnostic_records.extend(
        diagnostic_record(index, message)
        for index, message in enumerate(diagnostics, start=1)
    )

    metric_records: list[dict[str, object]] = []
    if isinstance(dfg_report.get("optimistic_cycles"), int) and dfg_path is not None:
        metric_records.append(
            metric_record(
                metric_id=f"metric::{workload}::dfg_sim_cycles",
                metric_class="optimistic_steps",
                value=int(dfg_report["optimistic_cycles"]),
                unit="cycles",
                fidelity_level="dfg_software",
                evidence_source_artifact_id=artifact_id(dfg_path),
                producer_component="loom-dfg-sim",
                derivation_kind="simulator_report",
            )
        )

    if isinstance(cgra_report.get("hardware_aware_cycles"), int) and cgra_path is not None:
        metric_records.append(
            metric_record(
                metric_id=f"metric::{workload}::cgra_sim_cycles",
                metric_class="hardware_cycles",
                value=int(cgra_report["hardware_aware_cycles"]),
                unit="cycles",
                fidelity_level="cgra_mapped",
                evidence_source_artifact_id=artifact_id(cgra_path),
                producer_component="loom-cgra-sim",
                derivation_kind="simulator_report",
                diagnostics=[str(item) for item in cgra_report.get("diagnostics", [])],
            )
        )

    if rtl_row is not None and rtl_path is not None:
        for key, metric_class, unit in (
            ("frequency_mhz", "frequency", "MHz"),
            ("area_um2", "area", "um2"),
            ("dynamic_power_mw", "dynamic_power", "mW"),
            ("leakage_power_mw", "leakage_power", "mW"),
        ):
            metric_records.append(
                metric_record(
                    metric_id=f"metric::{hardware}::{key}",
                    metric_class=metric_class,
                    value=numeric(rtl_row, key),
                    unit=unit,
                    fidelity_level="custom_calibrated",
                    evidence_source_artifact_id=artifact_id(rtl_path),
                    producer_component="rtl-fpa-summary",
                    derivation_kind="analytic_fpa",
                    diagnostics=[rtl_row.get("diagnostic", "")],
                )
            )

    energy_inputs = [
        f"metric::{workload}::cgra_sim_cycles",
        f"metric::{hardware}::frequency_mhz",
        f"metric::{hardware}::dynamic_power_mw",
        f"metric::{hardware}::leakage_power_mw",
    ]
    metric_records.append(
        metric_record(
            metric_id=f"metric::{workload}::energy_nj",
            metric_class="energy",
            value=numeric(dse_row, "energy_nj"),
            unit="nJ",
            fidelity_level="custom_calibrated",
            evidence_source_artifact_id=artifact_id(dse_path),
            producer_component="dse-candidate-summary",
            derivation_kind="cycle_frequency_power_area",
            diagnostics=[dse_row.get("diagnostic", "")],
            input_metric_ids=energy_inputs,
        )
    )

    return {
        "schema_version": 1,
        "kind": "workload_report_bundle",
        "bundle_id": f"workload::{workload}::{hardware}::{mapping_id}",
        "workload": workload,
        "source_artifact_identity": artifact_id(source_path) if source_path is not None else "",
        "compiler_command_identity": artifact_id(compiler_path) if compiler_path is not None else "",
        "runtime_input_identity": f"test-app-fixture::{workload}::default",
        "selected_hardware_candidate_identity": hardware,
        "selected_mapping_artifact_identity": artifact_id(mapping_path) if mapping_path is not None else "",
        "runtime_host_interface": runtime_host_interface,
        "runtime_evidence": runtime_evidence(runtime_package, runtime_path),
        "runtime_fallback_decision": runtime_fallback_decision,
        "input_artifact_fingerprints": input_artifact_fingerprints(
            [
                source_path,
                compiler_path,
                mapping_path,
                dfg_path,
                cgra_path,
                comparison_path,
                runtime_path,
                rtl_path,
                dse_path,
            ]
        ),
        "optional_artifact_identities": {
            "dfg_sim_report": artifact_id(dfg_path) if dfg_path is not None else "",
            "cgra_sim_report": artifact_id(cgra_path) if cgra_path is not None else "",
            "simulation_comparison_report": artifact_id(comparison_path) if comparison_path is not None else "",
            "runtime_package": artifact_id(runtime_path) if runtime_path is not None else "",
            "fpa_report": artifact_id(rtl_path) if rtl_path is not None else "",
            "dse_feedback_record": artifact_id(dse_path),
        },
        "report_status": "pass" if not diagnostics else "blocked",
        "diagnostic_records": diagnostic_records,
        "diagnostics": diagnostics,
        "metric_records": metric_records,
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
