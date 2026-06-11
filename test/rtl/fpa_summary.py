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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--primitive-coverage")
    parser.add_argument("--hardware-summary")
    parser.add_argument("--rtl-manifest")
    parser.add_argument("--eda-report")
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


def rtl_lint_evidence(manifest_path: Path | None, eda_path: Path | None) -> RtlLintEvidence | None:
    if manifest_path is None and eda_path is None:
        return None
    if manifest_path is None or eda_path is None:
        return blocked_lint_evidence(
            None,
            "RTL lint evidence unavailable: both --rtl-manifest and --eda-report are required",
        )
    manifest = read_json(manifest_path)
    if manifest.get("kind") != "rtl_manifest":
        return blocked_lint_evidence(
            None,
            "RTL lint evidence unavailable: input RTL manifest is missing or invalid",
        )
    raw_hardware = manifest.get("source_fabric_adg_identity")
    hardware = raw_hardware if isinstance(raw_hardware, str) and raw_hardware else None
    eda = read_json(eda_path)
    if eda.get("kind") != "eda_report":
        return blocked_lint_evidence(
            hardware,
            "RTL lint evidence unavailable: input EDA report is missing or invalid",
        )
    if eda.get("capability_class") != "rtl_lint":
        return blocked_lint_evidence(
            hardware,
            "RTL lint evidence unavailable: EDA report is not an RTL lint report",
        )
    if eda.get("rtl_manifest_identity") != intermediate_artifacts.artifact_id_for_path(manifest_path):
        return blocked_lint_evidence(
            hardware,
            "RTL lint evidence unavailable: EDA report does not reference the RTL manifest",
        )
    status = eda.get("status")
    if not isinstance(status, str) or not status:
        return blocked_lint_evidence(
            hardware,
            "RTL lint evidence unavailable: EDA report has no status",
        )
    if status not in intermediate_artifacts.BASE_STATUSES:
        return blocked_lint_evidence(
            hardware,
            f"RTL lint evidence unavailable: EDA report has unknown status {status!r}",
        )
    if hardware is None:
        return blocked_lint_evidence(
            None,
            "RTL lint evidence unavailable: RTL manifest has no source fabric ADG identity",
        )
    return RtlLintEvidence(
        hardware=hardware,
        status=status,
        diagnostic=first_diagnostic(eda),
        consumed_report=True,
        report_identity=intermediate_artifacts.artifact_id_for_path(eda_path),
    )


def analytic_fpa_row(
    workload: str,
    hardware: HardwareCandidate,
    lint_evidence: RtlLintEvidence | None,
) -> dict[str, str]:
    node_count = hardware.node_count
    link_count = hardware.link_count
    area_um2 = 1000.0 + 250.0 * node_count + 50.0 * link_count
    frequency_mhz = max(50.0, 500.0 - 10.0 * node_count - 5.0 * link_count)
    dynamic_power_mw = 1.0 + 0.2 * node_count + 0.05 * link_count
    leakage_power_mw = 0.1 + 0.0001 * area_um2
    rtl_lint_status = "skipped"
    lint_diagnostic = "RTL lint evidence not provided"
    if (
        lint_evidence is not None
        and (lint_evidence.hardware is None or hardware_matches(lint_evidence.hardware, hardware.name))
    ):
        rtl_lint_status = lint_evidence.status
        if lint_evidence.consumed_report:
            lint_diagnostic = f"RTL lint evidence status={lint_evidence.status}"
            if lint_evidence.report_identity:
                lint_diagnostic = f"{lint_diagnostic}; artifact={lint_evidence.report_identity}"
            if lint_evidence.diagnostic:
                lint_diagnostic = f"{lint_diagnostic}; diagnostic={lint_evidence.diagnostic}"
        elif lint_evidence.diagnostic:
            lint_diagnostic = lint_evidence.diagnostic
    return {
        "hardware": hardware.name,
        "workload": workload,
        "rtl_lint_status": rtl_lint_status,
        "rtl_sim_status": "skipped",
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
        "status": "pass",
        "diagnostic": (
            "analytic FPA estimate; fidelity=analytic; "
            f"activity_source=default_toggle; {lint_diagnostic}; "
            "RTL simulation and synthesis backends not run"
        ),
    }


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

    workloads = candidate_summary_common.workloads_from_primitive_coverage(primitive_path)
    hardware = read_hardware_candidates(hardware_path)
    if not workloads or not hardware:
        intermediate_artifacts.write_csv("rtl_fpa", intermediate_artifacts.output_path(args.output))
        return 0

    rows = [
        analytic_fpa_row(workload, candidate, lint_evidence)
        for workload in workloads
        for candidate in hardware
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("rtl_fpa", output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
