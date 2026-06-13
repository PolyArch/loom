#!/usr/bin/env python3
"""Regression test for hardware candidate report bundles."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import artifact_test_common


REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "bundle_id",
    "hardware_candidate_identity",
    "fabric_adg_identity",
    "adg_builder_recipe_identity",
    "rtl_manifest_identity",
    "eda_report_identities",
    "fpa_report_identities",
    "supported_workload_classes",
    "input_artifact_fingerprints",
    "report_status",
    "diagnostic_records",
    "diagnostics",
    "metric_records",
}


def metric_by_id(metrics: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(metric.get("metric_id")): metric for metric in metrics}


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-hardware-report-") as tmp:
        out_dir = Path(tmp)
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
            ],
            "intermediate artifact chain",
        )
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_manifest.sh",
                "--hardware-summary",
                str(out_dir / "adg-hardware-summary.csv"),
                "--output",
                str(out_dir / "rtl-manifest.json"),
            ],
            "architecture RTL manifest",
        )

        report = out_dir / "hardware-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_hardware_report_bundle.sh",
                "--output",
                str(report),
                "--artifact",
                str(out_dir / "adg-hardware-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-manifest.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-report.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "hardware candidate report bundle",
        )

        data = json.loads(report.read_text())
        missing = REQUIRED_KEYS - set(data)
        if missing:
            raise AssertionError(f"hardware report bundle missing keys: {sorted(missing)}")
        if data["kind"] != "hardware_report_bundle":
            raise AssertionError(f"unexpected hardware report kind: {data}")
        if data["report_status"] != "pass":
            raise AssertionError(f"hardware report should pass with ADG and FPA evidence: {data}")
        expected_hardware = "test/pnr/shared_reduction_adg.mlir::shared_reduction_adg"
        if data["hardware_candidate_identity"] != expected_hardware:
            raise AssertionError(f"unexpected hardware identity: {data}")
        if data["fabric_adg_identity"] != "test/pnr/shared_reduction_adg.mlir":
            raise AssertionError(f"unexpected Fabric ADG identity: {data}")
        if data["adg_builder_recipe_identity"] != "adg-builder::shared-reduction":
            raise AssertionError(f"unexpected ADG builder recipe identity: {data}")
        if data["rtl_manifest_identity"] != "rtl-manifest":
            raise AssertionError(f"unexpected RTL manifest identity: {data}")
        if data["fpa_report_identities"] != ["rtl-fpa-report"]:
            raise AssertionError(f"unexpected FPA report identities: {data}")
        if data["eda_report_identities"] != []:
            raise AssertionError(f"hardware report should not invent EDA report identities: {data}")
        if data["supported_workload_classes"] != ["vecsum"]:
            raise AssertionError(f"unexpected supported workload classes: {data}")
        expected_input_fingerprints = {
            "adg-hardware-summary": artifact_test_common.fingerprint(out_dir / "adg-hardware-summary.csv"),
            "rtl-manifest": artifact_test_common.fingerprint(out_dir / "rtl-manifest.json"),
            "rtl-fpa-report": artifact_test_common.fingerprint(out_dir / "rtl-fpa-report.json"),
        }
        if data["input_artifact_fingerprints"] != expected_input_fingerprints:
            raise AssertionError(f"unexpected hardware report input fingerprints: {data}")
        if data["diagnostic_records"] != []:
            raise AssertionError(f"passing hardware report should have no diagnostic records: {data}")

        blocked_eda = out_dir / "rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(out_dir / "rtl-manifest.json"),
                "--tool",
                "definitely-missing-verilator",
                "--output",
                str(blocked_eda),
            ],
            "blocked RTL EDA report",
        )
        blocked_eda_report = out_dir / "blocked-eda-hardware-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_hardware_report_bundle.sh",
                "--output",
                str(blocked_eda_report),
                "--artifact",
                str(out_dir / "adg-hardware-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-manifest.json"),
                "--artifact",
                str(blocked_eda),
                "--artifact",
                str(out_dir / "rtl-fpa-report.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "hardware report bundle with blocked EDA report",
        )
        blocked_eda_data = json.loads(blocked_eda_report.read_text())
        if blocked_eda_data["eda_report_identities"] != []:
            raise AssertionError(f"hardware report should ignore blocked EDA report: {blocked_eda_data}")

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
        passing_rtl_sim = out_dir / "passing-rtl-sim-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(out_dir / "rtl-manifest.json"),
                "--capability-class",
                "rtl_sim",
                "--tool",
                str(rtl_sim_tool),
                "--output",
                str(passing_rtl_sim),
            ],
            "passing RTL sim EDA report",
        )
        rtl_sim_report = out_dir / "passing-rtl-sim-hardware-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_hardware_report_bundle.sh",
                "--output",
                str(rtl_sim_report),
                "--artifact",
                str(out_dir / "adg-hardware-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-manifest.json"),
                "--artifact",
                str(passing_rtl_sim),
                "--artifact",
                str(out_dir / "rtl-fpa-report.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "hardware report bundle with passing RTL sim EDA report",
        )
        rtl_sim_data = json.loads(rtl_sim_report.read_text())
        if rtl_sim_data["eda_report_identities"] != ["passing-rtl-sim-report"]:
            raise AssertionError(f"hardware report should consume passing RTL sim report: {rtl_sim_data}")
        if "passing-rtl-sim-report" not in rtl_sim_data["input_artifact_fingerprints"]:
            raise AssertionError(f"hardware report should fingerprint RTL sim report: {rtl_sim_data}")
        rtl_sim_audit = out_dir / "passing-rtl-sim-hardware-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(rtl_sim_audit),
                str(rtl_sim_report),
            ],
            "hardware report bundle audit with RTL sim report",
        )
        passing_rtl_lint = out_dir / "passing-rtl-lint-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(out_dir / "rtl-manifest.json"),
                "--tool",
                str(rtl_sim_tool),
                "--output",
                str(passing_rtl_lint),
            ],
            "passing RTL lint EDA report",
        )
        both_eda_report = out_dir / "passing-lint-sim-hardware-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_hardware_report_bundle.sh",
                "--output",
                str(both_eda_report),
                "--artifact",
                str(out_dir / "adg-hardware-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-manifest.json"),
                "--artifact",
                str(passing_rtl_lint),
                "--artifact",
                str(passing_rtl_sim),
                "--artifact",
                str(out_dir / "rtl-fpa-report.json"),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "hardware report bundle with passing lint and sim EDA reports",
        )
        both_eda_data = json.loads(both_eda_report.read_text())
        if both_eda_data["eda_report_identities"] != [
            "passing-rtl-lint-report",
            "passing-rtl-sim-report",
        ]:
            raise AssertionError(f"hardware report should consume lint and sim EDA reports: {both_eda_data}")
        for identity in ("passing-rtl-lint-report", "passing-rtl-sim-report"):
            if identity not in both_eda_data["input_artifact_fingerprints"]:
                raise AssertionError(f"hardware report should fingerprint {identity}: {both_eda_data}")
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "passing-lint-sim-hardware-report-bundle-audit.json"),
                str(both_eda_report),
            ],
            "hardware report bundle audit with lint and sim EDA reports",
        )

        verilator = shutil.which("verilator")
        if verilator is not None:
            passing_eda = out_dir / "passing-rtl-eda-report.json"
            artifact_test_common.require_success(
                repo,
                [
                    "bash",
                    "test/rtl/run_rtl_eda_report.sh",
                    "--manifest",
                    str(out_dir / "rtl-manifest.json"),
                    "--tool",
                    verilator,
                    "--output",
                    str(passing_eda),
                ],
                "passing RTL EDA report",
            )
            passing_eda_report = out_dir / "passing-eda-hardware-report-bundle.json"
            artifact_test_common.require_success(
                repo,
                [
                    "bash",
                    "test/e2e/run_hardware_report_bundle.sh",
                    "--output",
                    str(passing_eda_report),
                    "--artifact",
                    str(out_dir / "adg-hardware-summary.csv"),
                    "--artifact",
                    str(out_dir / "rtl-manifest.json"),
                    "--artifact",
                    str(passing_eda),
                    "--artifact",
                    str(out_dir / "rtl-fpa-report.json"),
                    "--artifact",
                    str(out_dir / "rtl-fpa-summary.csv"),
                ],
                "hardware report bundle with passing EDA report",
            )
            passing_eda_data = json.loads(passing_eda_report.read_text())
            if passing_eda_data["eda_report_identities"] != ["passing-rtl-eda-report"]:
                raise AssertionError(f"hardware report should consume passing EDA report: {passing_eda_data}")
            if "passing-rtl-eda-report" not in passing_eda_data["input_artifact_fingerprints"]:
                raise AssertionError(f"hardware report should fingerprint passing EDA report: {passing_eda_data}")
            passing_eda_audit = out_dir / "passing-eda-hardware-report-bundle-audit.json"
            artifact_test_common.require_success(
                repo,
                [
                    "python3",
                    "test/e2e/audit_intermediate_artifacts.py",
                    "--output",
                    str(passing_eda_audit),
                    str(passing_eda_report),
                ],
                "hardware report bundle audit with EDA report",
            )
            mismatched_eda = out_dir / "mismatched-rtl-eda-report.json"
            mismatched_eda_data = json.loads(passing_eda.read_text())
            mismatched_eda_data["rtl_manifest_identity"] = "other-rtl-manifest"
            mismatched_eda.write_text(json.dumps(mismatched_eda_data, indent=2, sort_keys=True) + "\n")
            mismatched_eda_report = out_dir / "mismatched-eda-hardware-report-bundle.json"
            mismatched_eda_hardware_data = json.loads(passing_eda_report.read_text())
            mismatched_eda_hardware_data["eda_report_identities"] = ["mismatched-rtl-eda-report"]
            mismatched_eda_hardware_data["input_artifact_fingerprints"].pop(
                "passing-rtl-eda-report",
                None,
            )
            mismatched_eda_hardware_data["input_artifact_fingerprints"][
                "mismatched-rtl-eda-report"
            ] = artifact_test_common.fingerprint(mismatched_eda)
            mismatched_eda_report.write_text(
                json.dumps(mismatched_eda_hardware_data, indent=2, sort_keys=True) + "\n"
            )
            result = artifact_test_common.run_command(
                repo,
                [
                    "python3",
                    "test/e2e/audit_intermediate_artifacts.py",
                    "--output",
                    str(out_dir / "mismatched-eda-hardware-report-bundle-audit.json"),
                    str(mismatched_eda_report),
                ],
            )
            if result.returncode == 0:
                raise AssertionError("hardware report with mismatched EDA report unexpectedly passed audit")

        metrics = data.get("metric_records", [])
        if not isinstance(metrics, list) or not metrics:
            raise AssertionError(f"hardware report should include metric records: {data}")
        metrics_by_id = metric_by_id(metrics)
        expected_metrics = {
            "metric::test/pnr/shared_reduction_adg.mlir::shared_reduction_adg::node_count": (
                "hardware_nodes",
                29,
                "count",
                "fabric_verified",
            ),
            "metric::test/pnr/shared_reduction_adg.mlir::shared_reduction_adg::link_count": (
                "hardware_links",
                0,
                "count",
                "fabric_verified",
            ),
            "metric::test/pnr/shared_reduction_adg.mlir::shared_reduction_adg::frequency_mhz": (
                "frequency",
                210.0,
                "MHz",
                "analytic",
            ),
            "metric::test/pnr/shared_reduction_adg.mlir::shared_reduction_adg::area_um2": (
                "area",
                8250.0,
                "um2",
                "analytic",
            ),
        }
        for metric_id, (metric_class, value, unit, fidelity) in expected_metrics.items():
            metric = metrics_by_id.get(metric_id)
            if metric is None:
                raise AssertionError(f"missing metric {metric_id}: {metrics}")
            if metric.get("metric_class") != metric_class:
                raise AssertionError(f"unexpected metric class for {metric_id}: {metric}")
            if abs(float(metric.get("value")) - float(value)) > 0.001:
                raise AssertionError(f"unexpected metric value for {metric_id}: {metric}")
            if metric.get("unit") != unit or metric.get("fidelity_level") != fidelity:
                raise AssertionError(f"unexpected unit or fidelity for {metric_id}: {metric}")
            if not metric.get("evidence_source_artifact_id"):
                raise AssertionError(f"metric lacks evidence source: {metric}")

        audit = out_dir / "hardware-artifact-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit),
                str(report),
            ],
            "hardware report bundle audit",
        )
        audit_data = json.loads(audit.read_text())
        if audit_data.get("verdict") != "pass":
            raise AssertionError(f"expected hardware report audit pass: {audit_data}")
        mismatched_fpa_summary = out_dir / "mismatched-hardware-rtl-fpa-summary.csv"
        original_fpa_lines = (out_dir / "rtl-fpa-summary.csv").read_text().splitlines()
        header = original_fpa_lines[0].split(",")
        row = original_fpa_lines[1].split(",")
        row[header.index("hardware")] = "other_hardware"
        mismatched_fpa_summary.write_text(",".join(header) + "\n" + ",".join(row) + "\n")
        mismatched_fpa_report = out_dir / "mismatched-fpa-hardware-report-bundle.json"
        mismatched_fpa_data = json.loads(report.read_text())
        mismatched_fpa_data["fpa_report_identities"] = ["mismatched-hardware-rtl-fpa-summary"]
        mismatched_fpa_data["input_artifact_fingerprints"].pop("rtl-fpa-report", None)
        mismatched_fpa_data["input_artifact_fingerprints"][
            "mismatched-hardware-rtl-fpa-summary"
        ] = artifact_test_common.fingerprint(mismatched_fpa_summary)
        for metric in mismatched_fpa_data["metric_records"]:
            if metric.get("producer_component") in {"rtl-fpa-summary", "fpa-report"}:
                metric["evidence_source_artifact_id"] = "mismatched-hardware-rtl-fpa-summary"
        mismatched_fpa_report.write_text(json.dumps(mismatched_fpa_data, indent=2, sort_keys=True) + "\n")
        mismatched_fpa_audit = out_dir / "mismatched-fpa-hardware-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_fpa_audit),
                str(mismatched_fpa_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("hardware report with mismatched FPA hardware unexpectedly passed audit")
        mismatched_rtl_manifest = out_dir / "mismatched-hardware-rtl-manifest.json"
        mismatched_rtl_manifest_data = json.loads((out_dir / "rtl-manifest.json").read_text())
        mismatched_rtl_manifest_data["source_fabric_adg_identity"] = "other_fabric_adg"
        mismatched_rtl_manifest.write_text(
            json.dumps(mismatched_rtl_manifest_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_rtl_report = out_dir / "mismatched-rtl-hardware-report-bundle.json"
        mismatched_rtl_data = json.loads(report.read_text())
        mismatched_rtl_data["rtl_manifest_identity"] = "mismatched-hardware-rtl-manifest"
        mismatched_rtl_data["input_artifact_fingerprints"].pop("rtl-manifest", None)
        mismatched_rtl_data["input_artifact_fingerprints"][
            "mismatched-hardware-rtl-manifest"
        ] = artifact_test_common.fingerprint(mismatched_rtl_manifest)
        mismatched_rtl_report.write_text(json.dumps(mismatched_rtl_data, indent=2, sort_keys=True) + "\n")
        mismatched_rtl_audit = out_dir / "mismatched-rtl-hardware-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_rtl_audit),
                str(mismatched_rtl_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("hardware report with mismatched RTL manifest unexpectedly passed audit")
        stale_input_report = out_dir / "stale-input-hardware-report-bundle.json"
        stale_input_data = json.loads(report.read_text())
        stale_input_data["input_artifact_fingerprints"]["adg-hardware-summary"] = "0" * 64
        stale_input_report.write_text(json.dumps(stale_input_data, indent=2, sort_keys=True) + "\n")
        stale_input_audit = out_dir / "stale-input-hardware-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_input_audit),
                str(stale_input_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("hardware report with stale input fingerprint unexpectedly passed audit")
        stale_rtl_manifest_report = out_dir / "stale-rtl-manifest-hardware-report-bundle.json"
        stale_rtl_manifest_data = json.loads(report.read_text())
        stale_rtl_manifest_data["input_artifact_fingerprints"]["rtl-manifest"] = "0" * 64
        stale_rtl_manifest_report.write_text(
            json.dumps(stale_rtl_manifest_data, indent=2, sort_keys=True) + "\n"
        )
        stale_rtl_manifest_audit = out_dir / "stale-rtl-manifest-hardware-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_rtl_manifest_audit),
                str(stale_rtl_manifest_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("hardware report with stale RTL manifest fingerprint unexpectedly passed audit")

        bad_builder_recipe_report = out_dir / "bad-builder-recipe-hardware-report-bundle.json"
        bad_builder_recipe_data = json.loads(report.read_text())
        bad_builder_recipe_data["adg_builder_recipe_identity"] = []
        bad_builder_recipe_report.write_text(
            json.dumps(bad_builder_recipe_data, indent=2, sort_keys=True) + "\n"
        )
        bad_builder_recipe_audit = out_dir / "bad-builder-recipe-hardware-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_builder_recipe_audit),
                str(bad_builder_recipe_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("hardware report with malformed ADG builder recipe unexpectedly passed audit")

        mismatched_builder_recipe_report = out_dir / "mismatched-builder-recipe-hardware-report-bundle.json"
        mismatched_builder_recipe_data = json.loads(report.read_text())
        mismatched_builder_recipe_data["adg_builder_recipe_identity"] = "adg-builder::other"
        mismatched_builder_recipe_report.write_text(
            json.dumps(mismatched_builder_recipe_data, indent=2, sort_keys=True) + "\n"
        )
        mismatched_builder_recipe_audit = out_dir / "mismatched-builder-recipe-hardware-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mismatched_builder_recipe_audit),
                str(mismatched_builder_recipe_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("hardware report with mismatched ADG builder recipe unexpectedly passed audit")

        bad_rtl_identity_report = out_dir / "bad-rtl-identity-hardware-report-bundle.json"
        bad_rtl_identity_data = json.loads(report.read_text())
        bad_rtl_identity_data["rtl_manifest_identity"] = 7
        bad_rtl_identity_report.write_text(json.dumps(bad_rtl_identity_data, indent=2, sort_keys=True) + "\n")
        bad_rtl_identity_audit = out_dir / "bad-rtl-identity-hardware-report-bundle-audit.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(bad_rtl_identity_audit),
                str(bad_rtl_identity_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("hardware report with malformed RTL manifest identity unexpectedly passed audit")

        missing_fpa_report = out_dir / "missing-fpa-hardware-report-bundle.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/e2e/run_hardware_report_bundle.sh",
                "--output",
                str(missing_fpa_report),
                "--artifact",
                str(out_dir / "adg-hardware-summary.csv"),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("hardware report without FPA row unexpectedly passed")
        missing_fpa_data = json.loads(missing_fpa_report.read_text())
        records = missing_fpa_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "fpa_report_missing"
            and record.get("component") == "hardware_report_bundle"
            for record in records
        ):
            raise AssertionError(f"missing FPA report needs structured diagnostics: {missing_fpa_data}")
        missing_fpa_audit = out_dir / "missing-fpa-hardware-report-bundle-audit.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_fpa_audit),
                str(missing_fpa_report),
            ],
            "blocked hardware report bundle audit",
        )
        reviews = audit_data.get("artifact_reviews", [])
        matching_reviews = [
            review for review in reviews
            if review.get("schema") == "hardware_report_bundle"
        ]
        if len(matching_reviews) != 1:
            raise AssertionError(f"expected one hardware report bundle review: {audit_data}")

        duplicate_fpa_summary = out_dir / "duplicate-rtl-fpa-summary.csv"
        duplicate_fpa_summary.write_text((out_dir / "rtl-fpa-summary.csv").read_text())
        duplicate_fpa_report = out_dir / "duplicate-fpa-hardware-report-bundle.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_hardware_report_bundle.sh",
                "--output",
                str(duplicate_fpa_report),
                "--artifact",
                str(out_dir / "adg-hardware-summary.csv"),
                "--artifact",
                str(out_dir / "rtl-manifest.json"),
                "--artifact",
                str(duplicate_fpa_summary),
                "--artifact",
                str(out_dir / "rtl-fpa-summary.csv"),
            ],
            "hardware report bundle with duplicate FPA rows",
        )
        duplicate_fpa_data = json.loads(duplicate_fpa_report.read_text())
        if duplicate_fpa_data["fpa_report_identities"] != ["duplicate-rtl-fpa-summary"]:
            raise AssertionError(f"hardware report should select one canonical FPA report: {duplicate_fpa_data}")
        if sorted(duplicate_fpa_data["input_artifact_fingerprints"]) != [
            "adg-hardware-summary",
            "duplicate-rtl-fpa-summary",
            "rtl-manifest",
        ]:
            raise AssertionError(f"hardware report should fingerprint only selected FPA input: {duplicate_fpa_data}")
        duplicate_metrics = metric_by_id(duplicate_fpa_data.get("metric_records", []))
        frequency_metric = duplicate_metrics.get(
            "metric::test/pnr/shared_reduction_adg.mlir::shared_reduction_adg::frequency_mhz"
        )
        if frequency_metric is None or frequency_metric.get("evidence_source_artifact_id") != (
            "duplicate-rtl-fpa-summary"
        ):
            raise AssertionError(f"hardware report metrics should cite selected FPA report: {duplicate_fpa_data}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
