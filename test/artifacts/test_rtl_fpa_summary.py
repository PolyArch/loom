#!/usr/bin/env python3
"""Regression test for RTL/FPA summary candidate rows."""

from __future__ import annotations

import sys
from pathlib import Path

import artifact_test_common


HEADER = [
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


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-rtl-fpa-") as tmp:
        out_dir = Path(tmp)
        fpa = out_dir / "rtl-fpa-summary.csv"
        primitive, hardware = artifact_test_common.prepare_candidate_inputs(repo, out_dir)
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/rtl/run_rtl_fpa_summary.sh",
            fpa,
            HEADER,
            "--primitive-coverage",
            str(primitive),
            "--hardware-summary",
            str(hardware),
            label="RTL/FPA summary",
        )

        matches = [
            row
            for row in rows
            if row["workload"] == "vecadd" and row["hardware"].endswith("::pe_two_pes")
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one vecadd pe_two_pes row, got {rows}")
        row = matches[0]
        for column in ("rtl_lint_status", "rtl_sim_status", "synth_status"):
            if row[column] != "skipped":
                raise AssertionError(f"{column} should be skipped for analytic FPA: {row}")
        if row["status"] != "pass":
            raise AssertionError(f"analytic FPA row should pass: {row}")
        expected = {
            "frequency_mhz": "480.000",
            "area_um2": "1500.000",
            "dynamic_power_mw": "1.400",
            "leakage_power_mw": "0.250",
            "fidelity_level": "analytic",
            "frequency_source": "analytic_fpa_model",
            "area_source": "analytic_fpa_model",
            "power_source": "analytic_fpa_model",
            "activity_source": "default_toggle",
        }
        for column, value in expected.items():
            if row[column] != value:
                raise AssertionError(f"unexpected {column}: {row}")
        if "analytic FPA estimate" not in row.get("diagnostic", ""):
            raise AssertionError(f"unexpected diagnostic: {row}")
        audit = out_dir / "rtl-fpa-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit),
                str(fpa),
            ],
            "RTL/FPA summary audit",
        )
        bad_activity = out_dir / "bad-activity-rtl-fpa-summary.csv"
        bad_activity.write_text(
            fpa.read_text().replace(",default_toggle,", ",,", 1)
        )
        bad_activity_audit = out_dir / "bad-activity-rtl-fpa-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_activity_audit),
                str(bad_activity),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("RTL/FPA summary with missing activity source unexpectedly passed audit")

        manifest = out_dir / "rtl-manifest.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_manifest.sh",
                "--hardware-summary",
                str(hardware),
                "--output",
                str(manifest),
            ],
            "RTL manifest",
        )
        eda = out_dir / "rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                "definitely-missing-verilator",
                "--output",
                str(eda),
            ],
            "blocked RTL EDA report",
        )
        fpa_with_lint = out_dir / "with-lint-rtl-fpa-summary.csv"
        rows_with_lint = artifact_test_common.run_csv_summary(
            repo,
            "test/rtl/run_rtl_fpa_summary.sh",
            fpa_with_lint,
            HEADER,
            "--primitive-coverage",
            str(primitive),
            "--hardware-summary",
            str(hardware),
            "--rtl-manifest",
            str(manifest),
            "--eda-report",
            str(eda),
            label="RTL/FPA summary with RTL lint evidence",
        )
        lint_matches = [
            row
            for row in rows_with_lint
            if row["workload"] == "vecadd"
            and row["hardware"] == "test/fabric/unit/pe/valid.mlir::pe_2x2"
        ]
        if len(lint_matches) != 1:
            raise AssertionError(f"expected one vecadd pe_2x2 row with lint evidence, got {rows_with_lint}")
        lint_row = lint_matches[0]
        if lint_row["rtl_lint_status"] != "blocked":
            raise AssertionError(f"FPA row should consume blocked RTL lint evidence: {lint_row}")
        for column in ("rtl_sim_status", "synth_status"):
            if lint_row[column] != "skipped":
                raise AssertionError(f"{column} should stay skipped for analytic FPA: {lint_row}")
        for column in ("fidelity_level", "frequency_source", "area_source", "power_source", "activity_source"):
            if lint_row[column] != row[column]:
                raise AssertionError(f"RTL lint evidence must not change analytic FPA {column}: {lint_row}")
        if "RTL lint evidence status=blocked" not in lint_row.get("diagnostic", ""):
            raise AssertionError(f"FPA diagnostic should mention consumed lint evidence: {lint_row}")
        if "artifact=rtl-eda-report" not in lint_row.get("diagnostic", ""):
            raise AssertionError(f"FPA diagnostic should identify consumed lint artifact: {lint_row}")
        fpa_with_lint_audit = out_dir / "rtl-fpa-with-lint-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(fpa_with_lint_audit),
                str(fpa_with_lint),
            ],
            "RTL/FPA summary with lint evidence audit",
        )

        malformed_eda = out_dir / "malformed-rtl-eda-report.json"
        malformed_eda.write_text("{not-json\n")
        fpa_with_bad_lint = out_dir / "bad-lint-input-rtl-fpa-summary.csv"
        rows_with_bad_lint = artifact_test_common.run_csv_summary(
            repo,
            "test/rtl/run_rtl_fpa_summary.sh",
            fpa_with_bad_lint,
            HEADER,
            "--primitive-coverage",
            str(primitive),
            "--hardware-summary",
            str(hardware),
            "--rtl-manifest",
            str(manifest),
            "--eda-report",
            str(malformed_eda),
            label="RTL/FPA summary with malformed RTL lint evidence",
        )
        bad_lint_matches = [
            row
            for row in rows_with_bad_lint
            if row["workload"] == "vecadd"
            and row["hardware"] == "test/fabric/unit/pe/valid.mlir::pe_2x2"
        ]
        if len(bad_lint_matches) != 1:
            raise AssertionError(f"expected one vecadd pe_2x2 row with bad lint evidence, got {rows_with_bad_lint}")
        bad_lint_row = bad_lint_matches[0]
        if bad_lint_row["rtl_lint_status"] != "blocked":
            raise AssertionError(f"explicit malformed RTL lint evidence should block lint status: {bad_lint_row}")
        if "RTL lint evidence unavailable" not in bad_lint_row.get("diagnostic", ""):
            raise AssertionError(f"bad lint diagnostic should explain unavailable evidence: {bad_lint_row}")
        if bad_lint_row["status"] != "pass" or bad_lint_row["fidelity_level"] != "analytic":
            raise AssertionError(f"bad lint evidence must not change analytic FPA status: {bad_lint_row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
