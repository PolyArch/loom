#!/usr/bin/env python3
"""Regression test for ADG Builder system RTL and report consumption."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import artifact_test_common


HARDWARE_HEADER = [
    "hardware",
    "topology_class",
    "node_count",
    "link_count",
    "verify_status",
    "diagnostic",
    "tile_kinds",
    "schedule_kinds",
    "adg_builder_recipe_identity",
    "node_kinds",
]

FPA_HEADER = [
    "hardware",
    "workload",
    "rtl_lint_status",
    "rtl_sim_status",
    "synth_status",
    "frequency_mhz",
    "area_um2",
    "dynamic_power_mw",
    "leakage_power_mw",
    "fidelity_level",
    "frequency_source",
    "area_source",
    "power_source",
    "activity_source",
]

DEMONSTRATOR_HEADER = [
    "demonstrator",
    "compat_status",
    "artifact_status",
    "mapping_status",
    "sim_status",
    "rtl_status",
    "fpa_status",
    "report_status",
]


def read_json_object(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise AssertionError(f"expected JSON object in {path.name}: {data}")
    return data


def single_row(
    rows: list[dict[str, str]],
    *,
    key: str,
    value: str,
    label: str,
) -> dict[str, str]:
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise AssertionError(f"expected one {label} row for {key}={value!r}, got {rows}")
    return matches[0]


def assert_fields(row: dict[str, object] | dict[str, str], expected: dict[str, object], *, label: str) -> None:
    for key, value in expected.items():
        if row.get(key) != value:
            raise AssertionError(f"{label} {key}={row.get(key)!r}, expected {value!r}: {row}")


def write_filtered_hardware_summary(source: Path, output: Path, hardware_identity: str) -> None:
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("hardware") == hardware_identity]
        fieldnames = reader.fieldnames
    if len(rows) != 1 or fieldnames is None:
        raise AssertionError(f"expected one hardware row for {hardware_identity}: {rows}")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_by_id(metrics: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(metric.get("metric_id")): metric for metric in metrics}


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    builder = repo / "build" / "tools" / "loom-adg-builder-test" / "loom-adg-builder-test"
    if not builder.is_file():
        raise AssertionError(f"missing loom-adg-builder-test: {builder}")

    with artifact_test_common.repo_temp_dir(repo, "loom-adg-builder-system-rtl-") as tmp:
        out_dir = Path(tmp)
        system_mlir = out_dir / "adg-builder-heterogeneous-soc.mlir"
        artifact_test_common.require_success(
            repo,
            [str(builder), "--heterogeneous-soc", "--output", str(system_mlir)],
            "ADG Builder generated heterogeneous SoC",
        )

        hardware_summary = out_dir / "adg-hardware-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/fabric/run_adg_hardware_summary.sh",
            hardware_summary,
            HARDWARE_HEADER,
            "--input",
            str(system_mlir),
            "--input-recipe-identity",
            f"{system_mlir}=adg-builder::heterogeneous-soc",
            label="ADG Builder system hardware summary",
        )
        system_identity = (
            f"{system_mlir.resolve().relative_to(repo).as_posix()}::heterogeneous_dual_accel_soc"
        )
        system_row = single_row(
            rows,
            key="hardware",
            value=system_identity,
            label="ADG Builder generated fabric.system",
        )
        assert_fields(
            system_row,
            {
                "topology_class": "fabric_system",
                "node_count": "6",
                "link_count": "25",
                "verify_status": "pass",
                "adg_builder_recipe_identity": "adg-builder::heterogeneous-soc",
                "node_kinds": "acc_core;cache;dma_engine;fixed_accelerator;host_core;memory",
            },
            label="system hardware row",
        )

        system_only_hardware = out_dir / "system-only-adg-hardware-summary.csv"
        write_filtered_hardware_summary(hardware_summary, system_only_hardware, system_identity)

        rtl_manifest = out_dir / "system-rtl-manifest.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_manifest.sh",
                "--hardware-summary",
                str(system_only_hardware),
                "--output",
                str(rtl_manifest),
            ],
            "system RTL manifest",
        )
        manifest = read_json_object(rtl_manifest)
        assert_fields(
            manifest,
            {
                "kind": "rtl_manifest",
                "status": "pass",
                "mode": "architecture_rtl",
                "source_hardware_root": system_identity,
                "source_fabric_adg_identity": system_identity,
                "mapping_artifact_identity": "",
            },
            label="system RTL manifest",
        )
        if manifest.get("top_level_modules") != ["heterogeneous_dual_accel_soc"]:
            raise AssertionError(f"unexpected system top module: {manifest}")
        if manifest.get("generated_interfaces") != []:
            raise AssertionError(f"baseline fabric.system shell should expose no scalar ports: {manifest}")
        if manifest.get("behavioral_models") != ["behavioral_fabric_system_shell"]:
            raise AssertionError(f"system RTL manifest should identify system shell model: {manifest}")
        lowering = manifest.get("lowering_configuration")
        if not isinstance(lowering, dict):
            raise AssertionError(f"system RTL manifest lowering configuration must be object: {manifest}")
        assert_fields(
            lowering,
            {
                "lowering_kind": "architecture_rtl",
                "source_root_kind": "fabric_system",
                "systemverilog_profile": "behavioral_shell_v1",
                "node_count": 6,
                "link_count": 25,
            },
            label="system RTL lowering",
        )
        sources = manifest.get("emitted_source_files")
        if not isinstance(sources, list) or len(sources) != 1:
            raise AssertionError(f"expected one system RTL source: {manifest}")
        source = sources[0]
        if not isinstance(source, dict) or source.get("path") != "rtl/heterogeneous_dual_accel_soc.sv":
            raise AssertionError(f"unexpected system RTL source descriptor: {source}")
        source_path = rtl_manifest.parent / str(source["path"])
        source_text = source_path.read_text()
        for snippet in (
            "module heterogeneous_dual_accel_soc",
            "input logic clk",
            "input logic rst_n",
            "LOOM_NODE_COUNT = 6",
            "LOOM_LINK_COUNT = 25",
        ):
            if snippet not in source_text:
                raise AssertionError(f"system RTL source missed {snippet}: {source_text}")
        if "mgr" in source_text or "ctrl" in source_text:
            raise AssertionError(f"system RTL shell should not invent module-level ports: {source_text}")
        if source.get("fingerprint") != artifact_test_common.fingerprint(source_path):
            raise AssertionError(f"system RTL source fingerprint mismatch: {source}")

        eda_report = out_dir / "system-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(rtl_manifest),
                "--tool",
                "definitely-missing-verilator",
                "--output",
                str(eda_report),
            ],
            "blocked system RTL EDA report",
        )
        eda = read_json_object(eda_report)
        assert_fields(
            eda,
            {
                "kind": "eda_report",
                "status": "blocked",
                "rtl_manifest_identity": "system-rtl-manifest",
            },
            label="system RTL EDA report",
        )

        primitive = out_dir / "dataflow-primitive-coverage.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/dataflow/run_primitive_coverage.sh",
                "--case",
                "vecadd",
                "--output",
                str(primitive),
            ],
            "vecadd primitive coverage",
        )
        fpa = out_dir / "system-rtl-fpa-summary.csv"
        fpa_report = out_dir / "system-rtl-fpa-report.json"
        fpa_rows = artifact_test_common.run_csv_summary(
            repo,
            "test/rtl/run_rtl_fpa_summary.sh",
            fpa,
            FPA_HEADER,
            "--primitive-coverage",
            str(primitive),
            "--hardware-summary",
            str(system_only_hardware),
            "--rtl-manifest",
            str(rtl_manifest),
            "--eda-report",
            str(eda_report),
            label="system RTL/FPA summary",
        )
        fpa_row = single_row(fpa_rows, key="hardware", value=system_identity, label="system FPA")
        assert_fields(
            fpa_row,
            {
                "workload": "vecadd",
                "rtl_lint_status": "blocked",
                "rtl_sim_status": "skipped",
                "synth_status": "skipped",
                "frequency_mhz": "315.000",
                "area_um2": "3750.000",
                "dynamic_power_mw": "3.450",
                "leakage_power_mw": "0.475",
                "fidelity_level": "analytic",
                "frequency_source": "analytic_fpa_model",
                "area_source": "analytic_fpa_model",
                "power_source": "analytic_fpa_model",
                "activity_source": "default_toggle",
                "status": "pass",
            },
            label="system FPA row",
        )
        if "fidelity=analytic" not in fpa_row.get("diagnostic", ""):
            raise AssertionError(f"system FPA row should stay analytic: {fpa_row}")
        if "artifact=system-rtl-eda-report" not in fpa_row.get("diagnostic", ""):
            raise AssertionError(f"system FPA row should cite consumed EDA evidence: {fpa_row}")
        if fpa_row.get("fpa_report_identity") != "system-rtl-fpa-report":
            raise AssertionError(f"system FPA row should cite JSON report: {fpa_row}")

        hardware_bundle = out_dir / "system-hardware-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_hardware_report_bundle.sh",
                "--output",
                str(hardware_bundle),
                "--artifact",
                str(system_only_hardware),
                "--artifact",
                str(rtl_manifest),
                "--artifact",
                str(fpa_report),
                "--artifact",
                str(fpa),
            ],
            "system hardware report bundle",
        )
        bundle = read_json_object(hardware_bundle)
        assert_fields(
            bundle,
            {
                "kind": "hardware_report_bundle",
                "report_status": "pass",
                "hardware_candidate_identity": system_identity,
                "fabric_adg_identity": system_identity.rsplit("::", 1)[0],
                "adg_builder_recipe_identity": "adg-builder::heterogeneous-soc",
                "rtl_manifest_identity": "system-rtl-manifest",
            },
            label="system hardware report bundle",
        )
        if bundle.get("fpa_report_identities") != ["system-rtl-fpa-report"]:
            raise AssertionError(f"system hardware bundle missed FPA report identity: {bundle}")
        if bundle.get("supported_workload_classes") != ["vecadd"]:
            raise AssertionError(f"system hardware bundle should cite vecadd FPA support: {bundle}")
        fingerprints = bundle.get("input_artifact_fingerprints")
        expected_fingerprints = {
            "system-only-adg-hardware-summary": artifact_test_common.fingerprint(system_only_hardware),
            "system-rtl-manifest": artifact_test_common.fingerprint(rtl_manifest),
            "system-rtl-fpa-report": artifact_test_common.fingerprint(fpa_report),
        }
        if fingerprints != expected_fingerprints:
            raise AssertionError(f"system hardware bundle fingerprint drift: {bundle}")
        metrics = metric_by_id(bundle.get("metric_records", []))
        for metric_id, value in (
            (f"metric::{system_identity}::node_count", 6),
            (f"metric::{system_identity}::link_count", 25),
            (f"metric::{system_identity}::frequency_mhz", 315.0),
            (f"metric::{system_identity}::area_um2", 3750.0),
            (f"metric::{system_identity}::dynamic_power_mw", 3.45),
            (f"metric::{system_identity}::leakage_power_mw", 0.475),
        ):
            metric = metrics.get(metric_id)
            if metric is None or metric.get("value") != value:
                raise AssertionError(f"system hardware bundle missed metric {metric_id}: {bundle}")

        artifact_manifest = out_dir / "full-stack-artifact-manifest.json"
        manifest_inputs = [
            system_only_hardware,
            rtl_manifest,
            eda_report,
            fpa,
            fpa_report,
            hardware_bundle,
        ]
        manifest_args = []
        for artifact in manifest_inputs:
            manifest_args.extend(["--artifact", str(artifact)])
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_artifact_manifest.sh",
                *manifest_args,
                "--output",
                str(artifact_manifest),
            ],
            "system artifact manifest",
        )
        manifest_data = read_json_object(artifact_manifest)
        if manifest_data.get("diagnostics"):
            raise AssertionError(f"system artifact manifest should be clean: {manifest_data}")
        manifest_artifacts = {
            artifact.get("logical_path")
            for artifact in manifest_data.get("artifacts", [])
            if isinstance(artifact, dict)
        }
        if "system-hardware-report-bundle.json" not in manifest_artifacts:
            raise AssertionError(f"system hardware bundle should be registered in manifest: {manifest_data}")
        if "system-rtl-fpa-report.json" not in manifest_artifacts:
            raise AssertionError(f"system FPA report should be registered in manifest: {manifest_data}")

        demonstrator = out_dir / "e2e-demonstrator-summary.csv"
        demonstrator_rows = artifact_test_common.run_csv_summary(
            repo,
            "test/e2e/run_demonstrator_summary.sh",
            demonstrator,
            DEMONSTRATOR_HEADER,
            "--artifact",
            str(system_only_hardware),
            "--artifact",
            str(fpa),
            "--artifact",
            str(hardware_bundle),
            "--artifact",
            str(artifact_manifest),
            label="system e2e demonstrator summary",
        )
        system_demo = single_row(
            demonstrator_rows,
            key="demonstrator",
            value=f"hardware::{system_identity}",
            label="system hardware demonstrator",
        )
        assert_fields(
            system_demo,
            {
                "compat_status": "skipped",
                "artifact_status": "pass",
                "mapping_status": "skipped",
                "sim_status": "skipped",
                "rtl_status": "skipped",
                "fpa_status": "skipped",
                "report_status": "pass",
                "diagnostic": "hardware report bundle available",
            },
            label="system hardware demonstrator row",
        )

        final_manifest_args = []
        for artifact in [*manifest_inputs, demonstrator]:
            final_manifest_args.extend(["--artifact", str(artifact)])
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_artifact_manifest.sh",
                *final_manifest_args,
                "--output",
                str(artifact_manifest),
            ],
            "system artifact manifest with demonstrator",
        )
        final_manifest = read_json_object(artifact_manifest)
        if final_manifest.get("diagnostics"):
            raise AssertionError(f"final system artifact manifest should be clean: {final_manifest}")
        final_edges = [
            edge
            for edge in final_manifest.get("edges", [])
            if isinstance(edge, dict)
        ]
        if not any(
            edge.get("producer_artifact_kind") == "hardware_report_bundle"
            and edge.get("consumer_artifact_kind") == "e2e_demonstrator"
            for edge in final_edges
        ):
            raise AssertionError(f"system manifest missed hardware bundle to demonstrator edge: {final_manifest}")
        if not any(
            edge.get("producer_artifact_kind") == "fpa_report"
            and edge.get("consumer_artifact_kind") == "hardware_report_bundle"
            for edge in final_edges
        ):
            raise AssertionError(f"system manifest missed FPA report to hardware bundle edge: {final_manifest}")

        audit = out_dir / "system-artifact-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit),
                str(system_only_hardware),
                str(rtl_manifest),
                str(eda_report),
                str(fpa),
                str(fpa_report),
                str(hardware_bundle),
                str(artifact_manifest),
                str(demonstrator),
            ],
            "system hardware artifact audit",
        )
        audit_data = read_json_object(audit)
        if audit_data.get("verdict") != "pass":
            raise AssertionError(f"system hardware artifact audit should pass: {audit_data}")
        if audit_data.get("cross_artifact_findings"):
            raise AssertionError(f"system hardware audit should have no cross findings: {audit_data}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
