#!/usr/bin/env python3
"""Emit hardware candidate report bundles from ADG and FPA artifacts."""

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
input_artifact_fingerprints = intermediate_artifacts.input_artifact_fingerprints


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


def group_paths(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(intermediate_artifacts.artifact_kind_for_path(path), []).append(path)
    return grouped


def selected_hardware_row(paths: list[Path]) -> tuple[Path, dict[str, str]] | None:
    for path in paths:
        for row in read_csv(path):
            if row.get("verify_status") == "pass" and row.get("hardware"):
                return path, row
    return None


def hardware_matches(row_hardware: str, hardware: str) -> bool:
    return row_hardware == hardware or row_hardware.rsplit("::", 1)[-1] == hardware


def matching_fpa_rows(paths: list[Path], hardware: str) -> list[tuple[Path, dict[str, str]]]:
    rows: list[tuple[Path, dict[str, str]]] = []
    for path in paths:
        for row in read_csv(path):
            if row.get("status") != "pass":
                continue
            if hardware_matches(row.get("hardware", ""), hardware):
                rows.append((path, row))
    return rows


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
) -> dict[str, object]:
    return {
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
            "diagnostics": ["no passing ADG hardware summary row was provided"],
            "metric_records": [],
        }

    hardware_path, hardware_row = selected
    hardware = hardware_row["hardware"]
    fpa_rows = matching_fpa_rows(grouped.get("rtl_fpa", []), hardware)
    diagnostics: list[str] = []
    if not fpa_rows:
        diagnostics.append("no passing FPA row matched the hardware candidate")

    metric_records: list[dict[str, object]] = []
    metric_records.append(
        metric_record(
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
        metric_record(
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
                    value=numeric(fpa_row, key),
                    unit=unit,
                    fidelity_level="custom_calibrated",
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
        "rtl_manifest_identity": "",
        "eda_report_identities": [],
        "fpa_report_identities": fpa_report_ids,
        "supported_workload_classes": supported_workloads,
        "input_artifact_fingerprints": input_artifact_fingerprints(
            [hardware_path, *(path for path, _ in fpa_rows)]
        ),
        "report_status": "pass" if not diagnostics else "blocked",
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
