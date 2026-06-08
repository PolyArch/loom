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
group_paths = artifact_io_helpers.group_paths
hardware_matches = artifact_io_helpers.hardware_matches
matching_rtl_manifest = artifact_io_helpers.matching_rtl_manifest_path


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
    fpa_rows = matching_fpa_rows(grouped.get("rtl_fpa", []), hardware)
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
            metric_records.append(
                report_metric_helpers.metric_record(
                    metric_id=f"metric::{hardware}::{key}",
                    metric_class=metric_class,
                    value=numeric(fpa_row, key),
                    unit=unit,
                    fidelity_level=fpa_fidelity,
                    evidence_source_artifact_id=artifact_id(fpa_path),
                    producer_component="rtl-fpa-summary",
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
        "adg_builder_recipe_identity": "",
        "rtl_manifest_identity": artifact_id(rtl_manifest_path) if rtl_manifest_path is not None else "",
        "eda_report_identities": [],
        "fpa_report_identities": fpa_report_ids,
        "supported_workload_classes": supported_workloads,
        "input_artifact_fingerprints": input_artifact_fingerprints(
            [hardware_path, rtl_manifest_path, *(path for path, _ in fpa_rows)]
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
