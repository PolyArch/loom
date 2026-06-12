#!/usr/bin/env python3
"""Emit RTL/FPA summary rows from software and hardware summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402
import candidate_summary_common  # noqa: E402
import report_metric_helpers  # noqa: E402


@dataclass(frozen=True)
class HardwareCandidate:
    name: str
    node_count: int
    link_count: int


@dataclass(frozen=True)
class RtlLintEvidence:
    hardware: str | None
    status: str
    diagnostic: str
    consumed_report: bool
    report_identity: str
    capability_class: str = "rtl_lint"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--primitive-coverage")
    parser.add_argument("--hardware-summary")
    parser.add_argument("--rtl-manifest")
    parser.add_argument("--eda-report")
    parser.add_argument("--rtl-sim-report")
    parser.add_argument("--report-output")
    return parser.parse_args(argv)


def read_hardware_candidates(path: Path) -> list[HardwareCandidate]:
    if not path.is_file():
        return []
    candidates: list[HardwareCandidate] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("verify_status") != "pass" or not row.get("hardware"):
                continue
            try:
                node_count = int(row.get("node_count", ""))
                link_count = int(row.get("link_count", ""))
            except ValueError:
                continue
            if node_count <= 0 or link_count < 0:
                continue
            candidates.append(
                HardwareCandidate(
                    name=row["hardware"],
                    node_count=node_count,
                    link_count=link_count,
                )
            )
    return sorted(candidates, key=lambda candidate: candidate.name)


def format_estimate(value: float) -> str:
    return f"{value:.3f}"


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def default_fpa_report_output(summary_output: Path) -> Path:
    suffix = "rtl-fpa-summary.csv"
    if summary_output.name.endswith(suffix):
        prefix = summary_output.name[: -len(suffix)]
        return summary_output.with_name(f"{prefix}rtl-fpa-report.json")
    return summary_output.with_suffix(".fpa-report.json")


def hardware_matches(candidate: str, hardware: str) -> bool:
    return candidate == hardware or candidate.rsplit("::", 1)[-1] == hardware


def first_diagnostic(data: dict[str, object]) -> str:
    diagnostics = data.get("diagnostics")
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            if isinstance(diagnostic, str) and diagnostic:
                return diagnostic
    records = data.get("diagnostic_records")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict):
                message = record.get("message")
                if isinstance(message, str) and message:
                    return message
    return ""


def blocked_lint_evidence(hardware: str | None, diagnostic: str) -> RtlLintEvidence:
    return RtlLintEvidence(
        hardware=hardware,
        status="blocked",
        diagnostic=diagnostic,
        consumed_report=False,
        report_identity="",
    )


def blocked_backend_evidence(
    hardware: str | None,
    diagnostic: str,
    capability_class: str,
) -> RtlLintEvidence:
    return RtlLintEvidence(
        hardware=hardware,
        status="blocked",
        diagnostic=diagnostic,
        consumed_report=False,
        report_identity="",
        capability_class=capability_class,
    )


def rtl_backend_evidence(
    manifest_path: Path | None,
    eda_path: Path | None,
    *,
    capability_class: str,
    label: str,
) -> RtlLintEvidence | None:
    if manifest_path is None and eda_path is None:
        return None
    if manifest_path is None or eda_path is None:
        return blocked_backend_evidence(
            None,
            f"{label} evidence unavailable: both --rtl-manifest and EDA report are required",
            capability_class,
        )
    manifest = read_json(manifest_path)
    if manifest.get("kind") != "rtl_manifest":
        return blocked_backend_evidence(
            None,
            f"{label} evidence unavailable: input RTL manifest is missing or invalid",
            capability_class,
        )
    raw_hardware = manifest.get("source_fabric_adg_identity")
    hardware = raw_hardware if isinstance(raw_hardware, str) and raw_hardware else None
    eda = read_json(eda_path)
    if eda.get("kind") != "eda_report":
        return blocked_backend_evidence(
            hardware,
            f"{label} evidence unavailable: input EDA report is missing or invalid",
            capability_class,
        )
    if eda.get("capability_class") != capability_class:
        return blocked_backend_evidence(
            hardware,
            f"{label} evidence unavailable: EDA report is not a {capability_class} report",
            capability_class,
        )
    if eda.get("rtl_manifest_identity") != intermediate_artifacts.artifact_id_for_path(manifest_path):
        return blocked_backend_evidence(
            hardware,
            f"{label} evidence unavailable: EDA report does not reference the RTL manifest",
            capability_class,
        )
    status = eda.get("status")
    if not isinstance(status, str) or not status:
        return blocked_backend_evidence(
            hardware,
            f"{label} evidence unavailable: EDA report has no status",
            capability_class,
        )
    if status not in intermediate_artifacts.BASE_STATUSES:
        return blocked_backend_evidence(
            hardware,
            f"{label} evidence unavailable: EDA report has unknown status {status!r}",
            capability_class,
        )
    if hardware is None:
        return blocked_backend_evidence(
            None,
            f"{label} evidence unavailable: RTL manifest has no source fabric ADG identity",
            capability_class,
        )
    return RtlLintEvidence(
        hardware=hardware,
        status=status,
        diagnostic=first_diagnostic(eda),
        consumed_report=True,
        report_identity=intermediate_artifacts.artifact_id_for_path(eda_path),
        capability_class=capability_class,
    )


def rtl_lint_evidence(manifest_path: Path | None, eda_path: Path | None) -> RtlLintEvidence | None:
    return rtl_backend_evidence(
        manifest_path,
        eda_path,
        capability_class="rtl_lint",
        label="RTL lint",
    )


def rtl_sim_evidence(manifest_path: Path | None, eda_path: Path | None) -> RtlLintEvidence | None:
    return rtl_backend_evidence(
        manifest_path,
        eda_path,
        capability_class="rtl_sim",
        label="RTL sim",
    )


def backend_status_and_diagnostic(
    *,
    hardware_name: str,
    evidence: RtlLintEvidence | None,
    label: str,
    default_diagnostic: str,
) -> tuple[str, str]:
    if evidence is None:
        return "skipped", default_diagnostic
    if evidence.hardware is not None and not hardware_matches(evidence.hardware, hardware_name):
        return "skipped", default_diagnostic
    if evidence.consumed_report:
        diagnostic = f"{label} evidence status={evidence.status}"
        if evidence.report_identity:
            diagnostic = f"{diagnostic}; artifact={evidence.report_identity}"
        if evidence.diagnostic:
            diagnostic = f"{diagnostic}; diagnostic={evidence.diagnostic}"
        return evidence.status, diagnostic
    if evidence.diagnostic:
        return evidence.status, evidence.diagnostic
    return evidence.status, default_diagnostic


def analytic_fpa_row(
    workload: str,
    hardware: HardwareCandidate,
    lint_evidence: RtlLintEvidence | None,
    sim_evidence: RtlLintEvidence | None,
    *,
    fpa_report_identity: str,
) -> dict[str, str]:
    node_count = hardware.node_count
    link_count = hardware.link_count
    area_um2 = 1000.0 + 250.0 * node_count + 50.0 * link_count
    frequency_mhz = max(50.0, 500.0 - 10.0 * node_count - 5.0 * link_count)
    dynamic_power_mw = 1.0 + 0.2 * node_count + 0.05 * link_count
    leakage_power_mw = 0.1 + 0.0001 * area_um2
    rtl_lint_status, lint_diagnostic = backend_status_and_diagnostic(
        hardware_name=hardware.name,
        evidence=lint_evidence,
        label="RTL lint",
        default_diagnostic="RTL lint evidence not provided",
    )
    rtl_sim_status, sim_diagnostic = backend_status_and_diagnostic(
        hardware_name=hardware.name,
        evidence=sim_evidence,
        label="RTL sim",
        default_diagnostic="RTL sim evidence not provided",
    )
    return {
        "hardware": hardware.name,
        "workload": workload,
        "rtl_lint_status": rtl_lint_status,
        "rtl_sim_status": rtl_sim_status,
        "synth_status": "skipped",
        "frequency_mhz": format_estimate(frequency_mhz),
        "area_um2": format_estimate(area_um2),
        "dynamic_power_mw": format_estimate(dynamic_power_mw),
        "leakage_power_mw": format_estimate(leakage_power_mw),
        "fidelity_level": "analytic",
        "frequency_source": "analytic_fpa_model",
        "area_source": "analytic_fpa_model",
        "power_source": "analytic_fpa_model",
        "activity_source": "default_toggle",
        "fpa_report_identity": fpa_report_identity,
        "status": "pass",
        "diagnostic": (
            "analytic FPA estimate; fidelity=analytic; "
            f"activity_source=default_toggle; {lint_diagnostic}; "
            f"{sim_diagnostic}; synthesis backend not run"
        ),
    }


def fpa_metric_records(rows: list[dict[str, str]], report_identity: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    metric_specs = (
        ("frequency_mhz", "frequency", "MHz"),
        ("area_um2", "area", "um2"),
        ("dynamic_power_mw", "dynamic_power", "mW"),
        ("leakage_power_mw", "leakage_power", "mW"),
    )
    for row in rows:
        if row.get("status") != "pass":
            continue
        hardware = row.get("hardware", "")
        workload = row.get("workload", "")
        for key, metric_class, unit in metric_specs:
            record = report_metric_helpers.metric_record(
                metric_id=f"metric::{hardware}::{workload}::{key}",
                metric_class=metric_class,
                value=float(row[key]),
                unit=unit,
                fidelity_level=row.get("fidelity_level", "") or "analytic",
                evidence_source_artifact_id=report_identity,
                producer_component="fpa-report",
                derivation_kind="analytic_fpa",
                diagnostics=[row.get("diagnostic", "")],
            )
            record["hardware_candidate_identity"] = hardware
            record["workload"] = workload
            record["source_column"] = key
            if metric_class in {"dynamic_power", "leakage_power"}:
                record["activity_source"] = row.get("activity_source", "")
            if record["fidelity_level"] in {"analytic", "custom_calibrated"}:
                record["confidence"] = "model_default"
            records.append(record)
    return records


def write_fpa_report(
    *,
    output: Path,
    rows: list[dict[str, str]],
    primitive_path: Path,
    hardware_path: Path,
    rtl_manifest_path: Path | None,
    backend_input_paths: list[Path],
    backend_pass_paths: list[Path],
) -> None:
    report_identity = intermediate_artifacts.artifact_id_for_path(output)
    hardware_identities = sorted({row["hardware"] for row in rows if row.get("hardware")})
    workload_identities = sorted({row["workload"] for row in rows if row.get("workload")})
    metric_records = fpa_metric_records(rows, report_identity)
    backend_report_identities = [
        identity
        for identity in [intermediate_artifacts.artifact_id_for_path(path) for path in backend_pass_paths]
        if identity
    ]
    diagnostics = sorted({row.get("diagnostic", "") for row in rows if row.get("status") != "pass"})
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "kind": "fpa_report",
        "report_id": report_identity,
        "hardware_candidate_identity": hardware_identities[0] if len(hardware_identities) == 1 else "multiple",
        "hardware_candidate_identities": hardware_identities,
        "workload_identities": workload_identities,
        "mapping_artifact_identity": "",
        "cgra_sim_report_identity": "",
        "rtl_manifest_identity": intermediate_artifacts.artifact_id_for_path(rtl_manifest_path),
        "tool_profile_id": "analytic_fpa_model",
        "selected_library_profile_id": "",
        "estimation_configuration": {
            "model_id": "analytic_fpa_model",
            "activity_source": "default_toggle",
            "frequency_source": "analytic_fpa_model",
            "area_source": "analytic_fpa_model",
            "power_source": "analytic_fpa_model",
        },
        "calibration_identity": "",
        "frequency_results": [
            record["metric_id"]
            for record in metric_records
            if record.get("metric_class") == "frequency"
        ],
        "area_results": [
            record["metric_id"]
            for record in metric_records
            if record.get("metric_class") == "area"
        ],
        "power_results": [
            record["metric_id"]
            for record in metric_records
            if record.get("metric_class") in {"dynamic_power", "leakage_power"}
        ],
        "combined_metric_records": [],
        "metric_records": metric_records,
        "backend_report_identities": backend_report_identities,
        "input_artifact_fingerprints": intermediate_artifacts.input_artifact_fingerprints(
            [primitive_path, hardware_path, rtl_manifest_path, *backend_input_paths]
        ),
        "diagnostic_records": [],
        "diagnostics": diagnostics,
        "status": "pass" if rows and not diagnostics else "blocked",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    ignore_standard = os.environ.get("LOOM_IGNORE_STANDARD_ARTIFACTS") == "1"
    primitive_path = (
        Path(args.primitive_coverage)
        if args.primitive_coverage
        else Path()
        if ignore_standard
        else ROOT / "temp/dataflow-primitive-coverage.csv"
    )
    hardware_path = (
        Path(args.hardware_summary)
        if args.hardware_summary
        else Path()
        if ignore_standard
        else ROOT / "temp/adg-hardware-summary.csv"
    )
    lint_evidence = rtl_lint_evidence(
        Path(args.rtl_manifest) if args.rtl_manifest else None,
        Path(args.eda_report) if args.eda_report else None,
    )
    sim_evidence = rtl_sim_evidence(
        Path(args.rtl_manifest) if args.rtl_manifest and args.rtl_sim_report else None,
        Path(args.rtl_sim_report) if args.rtl_sim_report else None,
    )
    backend_input_paths: list[Path] = []
    backend_pass_paths: list[Path] = []
    for evidence, raw_path in (
        (lint_evidence, Path(args.eda_report) if args.eda_report else None),
        (sim_evidence, Path(args.rtl_sim_report) if args.rtl_sim_report else None),
    ):
        if evidence is not None and evidence.consumed_report and raw_path is not None:
            backend_input_paths.append(raw_path)
            if evidence.status == "pass":
                backend_pass_paths.append(raw_path)
    report_output = (
        Path(args.report_output)
        if args.report_output
        else default_fpa_report_output(output)
    )
    fpa_report_identity = intermediate_artifacts.artifact_id_for_path(report_output)

    workloads = candidate_summary_common.workloads_from_primitive_coverage(primitive_path)
    hardware = read_hardware_candidates(hardware_path)
    if not workloads or not hardware:
        intermediate_artifacts.write_csv("rtl_fpa", intermediate_artifacts.output_path(args.output))
        intermediate_artifacts.write_json("fpa_report", intermediate_artifacts.output_path(str(report_output)))
        return 0

    rows = [
        analytic_fpa_row(
            workload,
            candidate,
            lint_evidence,
            sim_evidence,
            fpa_report_identity=fpa_report_identity,
        )
        for workload in workloads
        for candidate in hardware
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("rtl_fpa", output, rows)
    write_fpa_report(
        output=report_output,
        rows=rows,
        primitive_path=primitive_path,
        hardware_path=hardware_path,
        rtl_manifest_path=Path(args.rtl_manifest) if args.rtl_manifest else None,
        backend_input_paths=backend_input_paths,
        backend_pass_paths=backend_pass_paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
