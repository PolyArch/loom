#!/usr/bin/env python3
"""Regression test for end-to-end demonstrator summary rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


HEADER = [
    "demonstrator",
    "compat_status",
    "artifact_status",
    "mapping_status",
    "sim_status",
    "rtl_status",
    "fpa_status",
    "report_status",
]


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-e2e-demonstrator-") as tmp:
        out_dir = Path(tmp)
        source = out_dir / "source-compat-summary.csv"
        cmsis_pipeline = out_dir / "cmsis-compiler-pipeline-summary.csv"
        primitive, hardware = artifact_test_common.prepare_candidate_inputs(repo, out_dir)
        mapping = out_dir / "pnr-mapping-summary.csv"
        sim = out_dir / "sim-cycle-summary.csv"
        rtl_fpa = out_dir / "rtl-fpa-summary.csv"
        manifest = out_dir / "full-stack-artifact-manifest.json"
        demonstrator = out_dir / "e2e-demonstrator-summary.csv"

        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_source_compat_summary.sh",
                "--case",
                "vecadd",
                "--output",
                str(source),
            ],
            "source compatibility summary",
        )
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/cmsis/run_compiler_pipeline_summary.sh",
                "--output",
                str(cmsis_pipeline),
            ],
            "CMSIS compiler pipeline summary",
        )
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/pnr/run_mapping_summary.sh",
                "--primitive-coverage",
                str(primitive),
                "--hardware-summary",
                str(hardware),
                "--output",
                str(mapping),
            ],
            "PnR mapping summary",
        )
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_sim_cycle_summary.sh",
                "--primitive-coverage",
                str(primitive),
                "--output",
                str(sim),
            ],
            "sim cycle summary",
        )
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_fpa_summary.sh",
                "--primitive-coverage",
                str(primitive),
                "--hardware-summary",
                str(hardware),
                "--output",
                str(rtl_fpa),
            ],
            "RTL/FPA summary",
        )
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_artifact_manifest.sh",
                "--artifact",
                str(source),
                "--artifact",
                str(mapping),
                "--artifact",
                str(sim),
                "--artifact",
                str(rtl_fpa),
                "--output",
                str(manifest),
            ],
            "artifact manifest",
        )
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/e2e/run_demonstrator_summary.sh",
            demonstrator,
            HEADER,
            "--artifact",
            str(source),
            "--artifact",
            str(cmsis_pipeline),
            "--artifact",
            str(hardware),
            "--artifact",
            str(mapping),
            "--artifact",
            str(sim),
            "--artifact",
            str(rtl_fpa),
            "--artifact",
            str(manifest),
            label="end-to-end demonstrator summary",
        )

        matches = [row for row in rows if "vecadd" in row["demonstrator"] and row["demonstrator"].endswith("pe_two_pes")]
        if len(matches) != 1:
            raise AssertionError(f"expected one vecadd pe_two_pes demonstrator row, got {rows}")
        row = matches[0]
        expected = {
            "compat_status": "pass",
            "artifact_status": "pass",
            "mapping_status": "blocked",
            "sim_status": "blocked",
            "rtl_status": "skipped",
            "fpa_status": "pass",
            "report_status": "blocked",
        }
        for key, value in expected.items():
            if row[key] != value:
                raise AssertionError(f"{key}={row[key]!r}, expected {value!r}: {row}")
        if "workload report bundle is not available yet" not in row.get("diagnostic", ""):
            raise AssertionError(f"unexpected diagnostic: {row}")

        unregistered_report = out_dir / "unregistered-workload-report-bundle.json"
        unregistered_report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "workload_report_bundle",
                    "bundle_id": "workload::vecadd::test/fabric/unit/pe/valid.mlir::pe_two_pes",
                    "workload": "vecadd",
                    "selected_hardware_candidate_identity": "test/fabric/unit/pe/valid.mlir::pe_two_pes",
                    "report_status": "pass",
                    "diagnostics": [],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        unregistered_output = out_dir / "unregistered-report-e2e-demonstrator-summary.csv"
        unregistered_rows = artifact_test_common.run_csv_summary(
            repo,
            "test/e2e/run_demonstrator_summary.sh",
            unregistered_output,
            HEADER,
            "--artifact",
            str(source),
            "--artifact",
            str(hardware),
            "--artifact",
            str(mapping),
            "--artifact",
            str(sim),
            "--artifact",
            str(rtl_fpa),
            "--artifact",
            str(unregistered_report),
            "--artifact",
            str(manifest),
            label="unregistered report demonstrator summary",
        )
        matches = [
            row
            for row in unregistered_rows
            if "vecadd" in row["demonstrator"] and row["demonstrator"].endswith("pe_two_pes")
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one unregistered report row, got {unregistered_rows}")
        if matches[0]["report_status"] != "blocked":
            raise AssertionError(
                "report bundles that are absent from the artifact manifest must not satisfy a demonstrator: "
                f"{matches[0]}"
            )
        if "artifact manifest" not in matches[0].get("diagnostic", ""):
            raise AssertionError(f"unregistered report diagnostic should name manifest membership: {matches[0]}")

        no_manifest_output = out_dir / "no-manifest-report-e2e-demonstrator-summary.csv"
        no_manifest_rows = artifact_test_common.run_csv_summary(
            repo,
            "test/e2e/run_demonstrator_summary.sh",
            no_manifest_output,
            HEADER,
            "--artifact",
            str(source),
            "--artifact",
            str(hardware),
            "--artifact",
            str(mapping),
            "--artifact",
            str(sim),
            "--artifact",
            str(rtl_fpa),
            "--artifact",
            str(unregistered_report),
            label="no manifest report demonstrator summary",
        )
        matches = [
            row
            for row in no_manifest_rows
            if "vecadd" in row["demonstrator"] and row["demonstrator"].endswith("pe_two_pes")
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one no-manifest report row, got {no_manifest_rows}")
        if matches[0]["report_status"] != "blocked":
            raise AssertionError(
                "report bundles must not satisfy a demonstrator without artifact manifest evidence: "
                f"{matches[0]}"
            )
        if "artifact manifest" not in matches[0].get("diagnostic", ""):
            raise AssertionError(f"no-manifest report diagnostic should name manifest membership: {matches[0]}")

        unregistered_hardware_report = out_dir / "unregistered-hardware-report-bundle.json"
        unregistered_hardware_report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "hardware_report_bundle",
                    "bundle_id": "hardware::test/fabric/unit/pe/valid.mlir::pe_two_pes",
                    "hardware_candidate_identity": "test/fabric/unit/pe/valid.mlir::pe_two_pes",
                    "report_status": "pass",
                    "diagnostics": [],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        unregistered_hardware_output = out_dir / "unregistered-hardware-e2e-demonstrator-summary.csv"
        unregistered_hardware_rows = artifact_test_common.run_csv_summary(
            repo,
            "test/e2e/run_demonstrator_summary.sh",
            unregistered_hardware_output,
            HEADER,
            "--artifact",
            str(source),
            "--artifact",
            str(hardware),
            "--artifact",
            str(mapping),
            "--artifact",
            str(sim),
            "--artifact",
            str(rtl_fpa),
            "--artifact",
            str(unregistered_hardware_report),
            "--artifact",
            str(manifest),
            label="unregistered hardware demonstrator summary",
        )
        hardware_matches = [
            row
            for row in unregistered_hardware_rows
            if row["demonstrator"] == "hardware::test/fabric/unit/pe/valid.mlir::pe_two_pes"
        ]
        if len(hardware_matches) != 1:
            raise AssertionError(f"expected one unregistered hardware report row, got {unregistered_hardware_rows}")
        if hardware_matches[0]["report_status"] != "blocked":
            raise AssertionError(
                "hardware report bundles that are absent from the artifact manifest must not satisfy a demonstrator: "
                f"{hardware_matches[0]}"
            )
        if "artifact manifest" not in hardware_matches[0].get("diagnostic", ""):
            raise AssertionError(
                f"unregistered hardware report diagnostic should name manifest membership: {hardware_matches[0]}"
            )

        hardware_matches = [
            row for row in rows
            if row["demonstrator"] == "hardware::test/fabric/unit/pe/valid.mlir::pe_two_pes"
        ]
        if len(hardware_matches) != 1:
            raise AssertionError(f"expected one hardware pe_two_pes demonstrator row, got {rows}")
        hardware_row = hardware_matches[0]
        expected_hardware = {
            "compat_status": "skipped",
            "artifact_status": "pass",
            "mapping_status": "skipped",
            "sim_status": "skipped",
            "rtl_status": "skipped",
            "fpa_status": "skipped",
            "report_status": "blocked",
        }
        for key, value in expected_hardware.items():
            if hardware_row[key] != value:
                raise AssertionError(f"{key}={hardware_row[key]!r}, expected {value!r}: {hardware_row}")
        if "hardware candidate verified" not in hardware_row.get("diagnostic", ""):
            raise AssertionError(f"unexpected hardware diagnostic: {hardware_row}")

        cmsis_matches = [row for row in rows if row["demonstrator"] == "cmsis::cmsis-dsp"]
        if len(cmsis_matches) != 1:
            raise AssertionError(f"expected one CMSIS-DSP demonstrator row, got {rows}")
        cmsis_row = cmsis_matches[0]
        expected_cmsis = {
            "compat_status": "pass",
            "artifact_status": "pass",
            "mapping_status": "skipped",
            "sim_status": "skipped",
            "rtl_status": "skipped",
            "fpa_status": "skipped",
            "report_status": "skipped",
        }
        for key, value in expected_cmsis.items():
            if cmsis_row[key] != value:
                raise AssertionError(f"{key}={cmsis_row[key]!r}, expected {value!r}: {cmsis_row}")
        if "CMSIS drop-in pipeline reached dataflow" not in cmsis_row.get("diagnostic", ""):
            raise AssertionError(f"unexpected CMSIS diagnostic: {cmsis_row}")

        statusless_sim = out_dir / "statusless-sim-cycle-summary.csv"
        statusless_sim.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "vecadd,0,0,,synthetic statusless numeric row\n"
        )
        statusless_output = out_dir / "statusless-e2e-demonstrator-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/e2e/run_demonstrator_summary.sh",
            statusless_output,
            HEADER,
            "--artifact",
            str(source),
            "--artifact",
            str(mapping),
            "--artifact",
            str(statusless_sim),
            "--artifact",
            str(rtl_fpa),
            "--artifact",
            str(manifest),
            label="statusless sim demonstrator summary",
        )
        matches = [row for row in rows if "vecadd" in row["demonstrator"] and row["demonstrator"].endswith("pe_two_pes")]
        if len(matches) != 1:
            raise AssertionError(f"expected one statusless sim row, got {rows}")
        if matches[0]["sim_status"] != "blocked":
            raise AssertionError(f"statusless sim row must not infer pass from numeric cells: {matches[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
