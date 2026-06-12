#!/usr/bin/env python3
"""Regression test for RTL/FPA summary candidate rows."""

from __future__ import annotations

import json
import shlex
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
        fpa_report = out_dir / "rtl-fpa-report.json"
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
            "--report-output",
            str(fpa_report),
            label="RTL/FPA summary",
        )
        report_data = json.loads(fpa_report.read_text())
        expected_report_keys = {
            "schema_version",
            "kind",
            "report_id",
            "hardware_candidate_identity",
            "rtl_manifest_identity",
            "tool_profile_id",
            "metric_records",
            "input_artifact_fingerprints",
            "diagnostic_records",
            "diagnostics",
            "status",
        }
        missing_report_keys = expected_report_keys - set(report_data)
        if missing_report_keys:
            raise AssertionError(f"FPA report missing keys: {sorted(missing_report_keys)}")
        if report_data.get("kind") != "fpa_report" or report_data.get("report_id") != "rtl-fpa-report":
            raise AssertionError(f"unexpected FPA report identity: {report_data}")
        if report_data.get("status") != "pass":
            raise AssertionError(f"FPA report should pass: {report_data}")
        if report_data.get("tool_profile_id") != "analytic_fpa_model":
            raise AssertionError(f"unexpected FPA tool profile: {report_data}")
        report_metrics = report_data.get("metric_records")
        if not isinstance(report_metrics, list) or not report_metrics:
            raise AssertionError(f"FPA report needs metric records: {report_data}")

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
        if row.get("fpa_report_identity") != "rtl-fpa-report":
            raise AssertionError(f"FPA summary row should cite JSON report: {row}")
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
                str(fpa_report),
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

        default_lint_tool = out_dir / "default-lint-tool"
        default_lint_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.test default lint'\n"
            "  exit 0\n"
            "fi\n"
            "exit 0\n"
        )
        default_lint_tool.chmod(default_lint_tool.stat().st_mode | 0o111)
        default_profile = out_dir / "default-rtl-eda-profile.sh"
        default_profile.write_text(
            f"export LOOM_RTL_LINT_TOOL={shlex.quote(str(default_lint_tool))}\n"
        )
        default_fpa = out_dir / "default-rtl-fpa-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "env",
                f"LOOM_RTL_FPA_STANDARD_DIR={out_dir}",
                f"LOOM_RTL_EDA_ENV_FILE={default_profile}",
                "bash",
                "test/rtl/run_rtl_fpa_summary.sh",
                "--output",
                str(default_fpa),
            ],
            "default RTL/FPA summary with auto lint evidence",
        )
        default_rows = artifact_test_common.read_csv_rows(default_fpa, HEADER)
        default_matches = [
            row
            for row in default_rows
            if row["workload"] == "vecadd" and row["hardware"].endswith("::pe_2x2")
        ]
        if len(default_matches) != 1:
            raise AssertionError(f"expected one default vecadd pe_2x2 row, got {default_rows}")
        default_row = default_matches[0]
        if default_row["rtl_lint_status"] != "pass":
            raise AssertionError(f"default FPA row should consume passing RTL lint evidence: {default_row}")
        if default_row["fidelity_level"] != "analytic":
            raise AssertionError(f"default RTL lint must not relabel analytic FPA metrics: {default_row}")
        if "RTL lint evidence status=pass" not in default_row.get("diagnostic", ""):
            raise AssertionError(f"default FPA diagnostic should cite lint evidence: {default_row}")
        default_manifest = out_dir / "rtl-manifest.json"
        default_eda = out_dir / "rtl-eda-report.json"
        default_report = out_dir / "default-rtl-fpa-report.json"
        for artifact in (default_manifest, default_eda, default_report):
            if not artifact.is_file():
                raise AssertionError(f"default FPA summary should produce {artifact}")
        default_eda_data = json.loads(default_eda.read_text())
        if default_eda_data.get("status") != "pass":
            raise AssertionError(f"default RTL EDA report should pass: {default_eda_data}")
        if default_eda_data.get("tool_version") != "Verilator 5.test default lint":
            raise AssertionError(f"default RTL EDA report should record tool version: {default_eda_data}")
        if default_eda_data.get("fidelity_level") != "rtl_structural":
            raise AssertionError(f"default RTL EDA report should record structural fidelity: {default_eda_data}")

        failing_lint_tool = out_dir / "default-failing-lint-tool"
        failing_lint_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.test default fail'\n"
            "  exit 0\n"
            "fi\n"
            "echo 'lint failed for default propagation' >&2\n"
            "exit 9\n"
        )
        failing_lint_tool.chmod(failing_lint_tool.stat().st_mode | 0o111)
        failing_profile = out_dir / "default-failing-rtl-eda-profile.sh"
        failing_profile.write_text(
            f"export LOOM_RTL_LINT_TOOL={shlex.quote(str(failing_lint_tool))}\n"
        )
        failing_fpa = out_dir / "default-failing-rtl-fpa-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "env",
                f"LOOM_RTL_FPA_STANDARD_DIR={out_dir}",
                f"LOOM_RTL_EDA_ENV_FILE={failing_profile}",
                "bash",
                "test/rtl/run_rtl_fpa_summary.sh",
                "--output",
                str(failing_fpa),
            ],
            "default RTL/FPA summary with failing lint evidence",
        )
        failing_rows = artifact_test_common.read_csv_rows(failing_fpa, HEADER)
        failing_matches = [
            row
            for row in failing_rows
            if row["workload"] == "vecadd" and row["hardware"].endswith("::pe_2x2")
        ]
        if len(failing_matches) != 1:
            raise AssertionError(f"expected one failing-lint vecadd pe_2x2 row, got {failing_rows}")
        failing_row = failing_matches[0]
        if failing_row["rtl_lint_status"] != "fail":
            raise AssertionError(f"default FPA row should consume failing RTL lint evidence: {failing_row}")
        if failing_row["status"] != "pass" or failing_row["fidelity_level"] != "analytic":
            raise AssertionError(f"failing lint must not relabel analytic FPA metrics: {failing_row}")
        if "RTL lint evidence status=fail" not in failing_row.get("diagnostic", ""):
            raise AssertionError(f"default FPA diagnostic should cite failing lint evidence: {failing_row}")
        failing_eda_data = json.loads(default_eda.read_text())
        if failing_eda_data.get("status") != "fail":
            raise AssertionError(f"default RTL EDA report should preserve failing lint status: {failing_eda_data}")
        if failing_eda_data.get("tool_version") != "Verilator 5.test default fail":
            raise AssertionError(f"default failing RTL EDA report should record tool version: {failing_eda_data}")

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
        rtl_sim_tool = out_dir / "rtl-sim-tool"
        rtl_sim_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ] || [ \"$1\" = \"-ID\" ]; then\n"
            "  echo 'VCS X.test sim'\n"
            "  exit 0\n"
            "fi\n"
            "out=''\n"
            "while [ \"$#\" -gt 0 ]; do\n"
            "  if [ \"$1\" = \"-o\" ]; then\n"
            "    shift\n"
            "    out=\"$1\"\n"
            "  fi\n"
            "  shift || break\n"
            "done\n"
            "if [ -n \"$out\" ]; then\n"
            "  printf '%s\\n' '#!/bin/sh' 'echo RTL sim smoke passed' 'exit 0' > \"$out\"\n"
            "  chmod +x \"$out\"\n"
            "fi\n"
            "exit 0\n"
        )
        rtl_sim_tool.chmod(rtl_sim_tool.stat().st_mode | 0o111)
        rtl_sim = out_dir / "rtl-sim-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--capability-class",
                "rtl_sim",
                "--tool",
                str(rtl_sim_tool),
                "--output",
                str(rtl_sim),
            ],
            "passing RTL sim EDA report",
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
            "--rtl-sim-report",
            str(rtl_sim),
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
        if lint_row["rtl_sim_status"] != "pass":
            raise AssertionError(f"FPA row should consume passing RTL sim evidence: {lint_row}")
        if lint_row["synth_status"] != "skipped":
            raise AssertionError(f"synth_status should stay skipped for analytic FPA: {lint_row}")
        for column in ("fidelity_level", "frequency_source", "area_source", "power_source", "activity_source"):
            if lint_row[column] != row[column]:
                raise AssertionError(f"RTL backend evidence must not change analytic FPA {column}: {lint_row}")
        if "RTL lint evidence status=blocked" not in lint_row.get("diagnostic", ""):
            raise AssertionError(f"FPA diagnostic should mention consumed lint evidence: {lint_row}")
        if "RTL sim evidence status=pass" not in lint_row.get("diagnostic", ""):
            raise AssertionError(f"FPA diagnostic should mention consumed sim evidence: {lint_row}")
        if "artifact=rtl-eda-report" not in lint_row.get("diagnostic", ""):
            raise AssertionError(f"FPA diagnostic should identify consumed lint artifact: {lint_row}")
        if "artifact=rtl-sim-eda-report" not in lint_row.get("diagnostic", ""):
            raise AssertionError(f"FPA diagnostic should identify consumed sim artifact: {lint_row}")
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
        fpa_with_lint_report = out_dir / "with-lint-rtl-fpa-report.json"
        fpa_with_lint_report_data = json.loads(fpa_with_lint_report.read_text())
        if fpa_with_lint_report_data.get("backend_report_identities") != [
            "rtl-sim-eda-report",
        ]:
            raise AssertionError(
                f"FPA JSON should cite only passing backend reports: {fpa_with_lint_report_data}"
            )
        fingerprints = fpa_with_lint_report_data.get("input_artifact_fingerprints")
        if not isinstance(fingerprints, dict) or "rtl-eda-report" not in fingerprints:
            raise AssertionError(
                f"FPA JSON should still fingerprint consumed blocked lint input: {fpa_with_lint_report_data}"
            )
        bad_backend_reference = out_dir / "bad-backend-reference-rtl-fpa-report.json"
        bad_backend_reference_data = dict(fpa_with_lint_report_data)
        bad_backend_reference_data["backend_report_identities"] = [
            "rtl-eda-report",
            "rtl-sim-eda-report",
        ]
        bad_backend_reference.write_text(
            json.dumps(bad_backend_reference_data, indent=2, sort_keys=True) + "\n"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "bad-backend-reference-audit-summary.json"),
                str(bad_backend_reference),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("FPA report with blocked backend identity unexpectedly passed audit")

        passing_lint_tool = out_dir / "passing-lint-tool"
        passing_lint_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.test lint'\n"
            "  exit 0\n"
            "fi\n"
            "exit 0\n"
        )
        passing_lint_tool.chmod(passing_lint_tool.stat().st_mode | 0o111)
        passing_lint = out_dir / "passing-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                str(passing_lint_tool),
                "--output",
                str(passing_lint),
            ],
            "passing RTL lint EDA report",
        )
        fpa_with_passing_lint = out_dir / "passing-lint-rtl-fpa-summary.csv"
        rows_with_passing_lint = artifact_test_common.run_csv_summary(
            repo,
            "test/rtl/run_rtl_fpa_summary.sh",
            fpa_with_passing_lint,
            HEADER,
            "--primitive-coverage",
            str(primitive),
            "--hardware-summary",
            str(hardware),
            "--rtl-manifest",
            str(manifest),
            "--eda-report",
            str(passing_lint),
            "--rtl-sim-report",
            str(rtl_sim),
            label="RTL/FPA summary with passing RTL lint evidence",
        )
        passing_lint_matches = [
            row
            for row in rows_with_passing_lint
            if row["workload"] == "vecadd"
            and row["hardware"] == "test/fabric/unit/pe/valid.mlir::pe_2x2"
        ]
        if len(passing_lint_matches) != 1:
            raise AssertionError(
                f"expected one vecadd pe_2x2 row with passing lint evidence, got {rows_with_passing_lint}"
            )
        passing_lint_row = passing_lint_matches[0]
        if passing_lint_row["rtl_lint_status"] != "pass":
            raise AssertionError(f"FPA row should consume passing RTL lint evidence: {passing_lint_row}")
        if passing_lint_row["fidelity_level"] != "analytic":
            raise AssertionError(f"passing lint must not relabel analytic FPA metrics: {passing_lint_row}")
        if "RTL lint evidence status=pass" not in passing_lint_row.get("diagnostic", ""):
            raise AssertionError(f"FPA diagnostic should mention passing lint evidence: {passing_lint_row}")
        passing_lint_report = out_dir / "passing-lint-rtl-fpa-report.json"
        passing_lint_report_data = json.loads(passing_lint_report.read_text())
        if passing_lint_report_data.get("backend_report_identities") != [
            "passing-rtl-eda-report",
            "rtl-sim-eda-report",
        ]:
            raise AssertionError(
                f"FPA JSON should cite passing backend reports: {passing_lint_report_data}"
            )
        passing_lint_metrics = passing_lint_report_data.get("metric_records", [])
        if not passing_lint_metrics or any(
            metric.get("fidelity_level") != "analytic"
            for metric in passing_lint_metrics
            if isinstance(metric, dict)
        ):
            raise AssertionError(
                f"lint-backed FPA report must keep analytic metric fidelity: {passing_lint_report_data}"
            )
        bad_fpa_metric_fidelity = out_dir / "bad-fpa-metric-fidelity-rtl-fpa-report.json"
        bad_fpa_metric_fidelity_data = dict(passing_lint_report_data)
        bad_fpa_metric_fidelity_metrics = [
            dict(metric) if isinstance(metric, dict) else metric
            for metric in passing_lint_metrics
        ]
        bad_fpa_metric_fidelity_metrics[0]["fidelity_level"] = "rtl_functional"
        bad_fpa_metric_fidelity_data["metric_records"] = bad_fpa_metric_fidelity_metrics
        bad_fpa_metric_fidelity.write_text(
            json.dumps(bad_fpa_metric_fidelity_data, indent=2, sort_keys=True) + "\n"
        )
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "bad-fpa-metric-fidelity-audit-summary.json"),
                str(bad_fpa_metric_fidelity),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("FPA report with rtl_functional metric fidelity unexpectedly passed audit")

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
        bad_lint_report = out_dir / "bad-lint-input-rtl-fpa-report.json"
        bad_lint_report_data = json.loads(bad_lint_report.read_text())
        if bad_lint_report_data.get("backend_report_identities") != []:
            raise AssertionError(
                f"FPA JSON must not cite malformed backend reports: {bad_lint_report_data}"
            )

        malformed_sim = out_dir / "malformed-rtl-sim-report.json"
        malformed_sim.write_text("{not-json\n")
        fpa_with_bad_sim = out_dir / "bad-sim-input-rtl-fpa-summary.csv"
        rows_with_bad_sim = artifact_test_common.run_csv_summary(
            repo,
            "test/rtl/run_rtl_fpa_summary.sh",
            fpa_with_bad_sim,
            HEADER,
            "--primitive-coverage",
            str(primitive),
            "--hardware-summary",
            str(hardware),
            "--rtl-manifest",
            str(manifest),
            "--eda-report",
            str(eda),
            "--rtl-sim-report",
            str(malformed_sim),
            label="RTL/FPA summary with malformed RTL sim evidence",
        )
        bad_sim_matches = [
            row
            for row in rows_with_bad_sim
            if row["workload"] == "vecadd"
            and row["hardware"] == "test/fabric/unit/pe/valid.mlir::pe_2x2"
        ]
        if len(bad_sim_matches) != 1:
            raise AssertionError(f"expected one vecadd pe_2x2 row with bad sim evidence, got {rows_with_bad_sim}")
        bad_sim_row = bad_sim_matches[0]
        if bad_sim_row["rtl_lint_status"] != "blocked" or bad_sim_row["rtl_sim_status"] != "blocked":
            raise AssertionError(f"bad sim evidence should block only sim status while preserving lint: {bad_sim_row}")
        bad_sim_report = out_dir / "bad-sim-input-rtl-fpa-report.json"
        bad_sim_report_data = json.loads(bad_sim_report.read_text())
        if bad_sim_report_data.get("backend_report_identities") != []:
            raise AssertionError(
                f"FPA JSON must not cite blocked lint or malformed sim backend: {bad_sim_report_data}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
