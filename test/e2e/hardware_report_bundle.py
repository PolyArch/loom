#!/usr/bin/env python3
"""Emit hardware candidate report bundles from ADG and FPA artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402
import artifact_io_helpers  # noqa: E402
import report_metric_helpers  # noqa: E402


artifact_id = intermediate_artifacts.artifact_id_for_path
input_artifact_fingerprints = intermediate_artifacts.input_artifact_fingerprints
read_csv = artifact_io_helpers.read_csv
read_json = artifact_io_helpers.read_json
group_paths = artifact_io_helpers.group_paths
hardware_matches = artifact_io_helpers.hardware_matches
matching_rtl_manifest = artifact_io_helpers.matching_rtl_manifest_path
matching_eda_reports = artifact_io_helpers.matching_eda_report_paths


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    return parser.parse_args(argv)


def selected_hardware_row(paths: list[Path]) -> tuple[Path, dict[str, str]] | None:
    for path in paths:
        for row in read_csv(path):
            if row.get("verify_status") == "pass" and row.get("hardware"):
                return path, row
    return None


def matching_fpa_rows(paths: list[Path], hardware: str) -> list[tuple[Path, dict[str, str]]]:
    rows: list[tuple[Path, dict[str, str]]] = []
    for path in paths:
        for row in read_csv(path):
            if row.get("status") != "pass":
                continue
            if hardware_matches(row.get("hardware", ""), hardware):
                rows.append((path, row))
    return artifact_io_helpers.select_by_artifact_id(rows, lambda entry: entry[0])


def fpa_rows_from_report(path: Path) -> list[dict[str, str]]:
    data = read_json(path)
    if data.get("kind") != "fpa_report" or data.get("status") != "pass":
        return []
    rows_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for metric in data.get("metric_records", []):
        if not isinstance(metric, dict):
            continue
        hardware = metric.get("hardware_candidate_identity")
        workload = metric.get("workload")
        source_column = metric.get("source_column")
        value = metric.get("value")
        if not (
            isinstance(hardware, str)
            and hardware
            and isinstance(workload, str)
            and workload
            and isinstance(source_column, str)
            and isinstance(value, (int, float))
        ):
            continue
        key = (hardware, workload)
        row = rows_by_key.setdefault(
            key,
            {
                "hardware": hardware,
                "workload": workload,
                "status": "pass",
                "fidelity_level": str(metric.get("fidelity_level", "")),
                "frequency_source": "analytic_fpa_model",
                "area_source": "analytic_fpa_model",
                "power_source": "analytic_fpa_model",
                "activity_source": str(metric.get("activity_source", "")),
                "diagnostic": "; ".join(str(item) for item in data.get("diagnostics", []) if item),
            },
        )
        if not row.get("activity_source") and isinstance(metric.get("activity_source"), str):
            row["activity_source"] = str(metric.get("activity_source"))
        row[source_column] = f"{float(value):.3f}"
    required = {"frequency_mhz", "area_um2", "dynamic_power_mw", "leakage_power_mw"}
    return [
        row
        for row in rows_by_key.values()
        if required <= set(row)
    ]


def matching_fpa_report_rows(paths: list[Path], hardware: str) -> list[tuple[Path, dict[str, str]]]:
    rows: list[tuple[Path, dict[str, str]]] = []
    for path in paths:
        for row in fpa_rows_from_report(path):
            if hardware_matches(row.get("hardware", ""), hardware):
                rows.append((path, row))
    if not rows:
        return []
    selected_id = sorted({artifact_id(path) for path, _ in rows})[0]
    return [(path, row) for path, row in rows if artifact_id(path) == selected_id]


def matching_fpa_evidence(
    report_paths: list[Path],
    summary_paths: list[Path],
    hardware: str,
) -> list[tuple[Path, dict[str, str]]]:
    report_rows = matching_fpa_report_rows(report_paths, hardware)
    if report_rows:
        return report_rows
    return matching_fpa_rows(summary_paths, hardware)


def numeric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def diagnostic_class(message: str) -> str:
    if "FPA row" in message:
        return "fpa_report_missing"
    if "RTL manifest" in message:
        return "rtl_manifest_missing"
    if "ADG hardware summary" in message:
        return "hardware_candidate_missing"
    return "hardware_report_bundle_failure"


def diagnostic_records(diagnostics: list[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for index, message in enumerate(diagnostics, start=1):
        records.append(
            {
                "diagnostic_id": f"hardware-report-bundle::{index}",
                "diagnostic_class": diagnostic_class(message),
                "component": "hardware_report_bundle",
                "severity": "error",
                "message": message,
            }
        )
    return records


def fabric_adg_identity(hardware: str) -> str:
    return hardware.rsplit("::", 1)[0] if "::" in hardware else hardware


def build_bundle(paths: list[Path]) -> dict[str, object]:
    grouped = group_paths(paths)
    selected = selected_hardware_row(grouped.get("adg_hardware", []))
    if selected is None:
        return {
            "schema_version": 1,
            "kind": "hardware_report_bundle",
            "bundle_id": "hardware::blocked",
            "hardware_candidate_identity": "unknown",
            "fabric_adg_identity": "",
            "adg_builder_recipe_identity": "",
            "rtl_manifest_identity": "",
            "eda_report_identities": [],
            "fpa_report_identities": [],
            "supported_workload_classes": [],
            "input_artifact_fingerprints": {},
            "report_status": "blocked",
            "diagnostic_records": diagnostic_records(["no passing ADG hardware summary row was provided"]),
            "diagnostics": ["no passing ADG hardware summary row was provided"],
            "metric_records": [],
        }

    hardware_path, hardware_row = selected
    hardware = hardware_row["hardware"]
    rtl_manifest_path = matching_rtl_manifest(grouped.get("rtl_manifest", []), hardware)
    rtl_manifest_identity = artifact_id(rtl_manifest_path) if rtl_manifest_path is not None else ""
    eda_report_paths = matching_eda_reports(grouped.get("eda_report", []), rtl_manifest_identity)
    fpa_rows = matching_fpa_evidence(
        grouped.get("fpa_report", []),
        grouped.get("rtl_fpa", []),
        hardware,
    )
    diagnostics: list[str] = []
    if rtl_manifest_path is None:
        diagnostics.append("no passing RTL manifest matched the hardware candidate")
    if not fpa_rows:
        diagnostics.append("no passing FPA row matched the hardware candidate")

    metric_records: list[dict[str, object]] = []
    metric_records.append(
        report_metric_helpers.metric_record(
            metric_id=f"metric::{hardware}::node_count",
            metric_class="hardware_nodes",
            value=int(hardware_row["node_count"]),
            unit="count",
            fidelity_level="fabric_verified",
            evidence_source_artifact_id=artifact_id(hardware_path),
            producer_component="adg-hardware-summary",
            derivation_kind="fabric_verifier_summary",
            diagnostics=[hardware_row.get("diagnostic", "")],
        )
    )
    metric_records.append(
        report_metric_helpers.metric_record(
            metric_id=f"metric::{hardware}::link_count",
            metric_class="hardware_links",
            value=int(hardware_row["link_count"]),
            unit="count",
            fidelity_level="fabric_verified",
            evidence_source_artifact_id=artifact_id(hardware_path),
            producer_component="adg-hardware-summary",
            derivation_kind="fabric_verifier_summary",
            diagnostics=[hardware_row.get("diagnostic", "")],
        )
    )

    fpa_report_ids = sorted({artifact_id(path) for path, _ in fpa_rows})
    eda_report_ids = [artifact_id(path) for path in eda_report_paths]
    supported_workloads = sorted({row.get("workload", "") for _, row in fpa_rows if row.get("workload")})
    if fpa_rows:
        fpa_path, fpa_row = fpa_rows[0]
        fpa_fidelity = fpa_row.get("fidelity_level", "") or "custom_calibrated"
        for key, metric_class, unit in (
            ("frequency_mhz", "frequency", "MHz"),
            ("area_um2", "area", "um2"),
            ("dynamic_power_mw", "dynamic_power", "mW"),
            ("leakage_power_mw", "leakage_power", "mW"),
        ):
            producer_component = (
                "fpa-report"
                if intermediate_artifacts.artifact_kind_for_path(fpa_path) == "fpa_report"
                else "rtl-fpa-summary"
            )
            metric_records.append(
                report_metric_helpers.metric_record(
                    metric_id=f"metric::{hardware}::{key}",
                    metric_class=metric_class,
                    value=numeric(fpa_row, key),
                    unit=unit,
                    fidelity_level=fpa_fidelity,
                    evidence_source_artifact_id=artifact_id(fpa_path),
                    producer_component=producer_component,
                    derivation_kind="analytic_fpa",
                    diagnostics=[fpa_row.get("diagnostic", "")],
                )
            )

    return {
        "schema_version": 1,
        "kind": "hardware_report_bundle",
        "bundle_id": f"hardware::{hardware}",
        "hardware_candidate_identity": hardware,
        "fabric_adg_identity": fabric_adg_identity(hardware),
        "adg_builder_recipe_identity": hardware_row.get("adg_builder_recipe_identity", ""),
        "rtl_manifest_identity": artifact_id(rtl_manifest_path) if rtl_manifest_path is not None else "",
        "eda_report_identities": eda_report_ids,
        "fpa_report_identities": fpa_report_ids,
        "supported_workload_classes": supported_workloads,
        "input_artifact_fingerprints": input_artifact_fingerprints(
            [hardware_path, rtl_manifest_path, *eda_report_paths, *(path for path, _ in fpa_rows)]
        ),
        "report_status": "pass" if not diagnostics else "blocked",
        "diagnostic_records": diagnostic_records(diagnostics),
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
