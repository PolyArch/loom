#!/usr/bin/env python3
"""Regression test for the ordered intermediate artifact chain."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import artifact_test_common


EXPECTED_FILES = [
    "old-app-corpus-inventory.csv",
    "app-corpus-import-status.csv",
    "source-compat-summary.csv",
    "compiler-pipeline-summary.csv",
    "cmsis-compiler-pipeline-summary.csv",
    "dataflow-primitive-coverage.csv",
    "adg-hardware-summary.csv",
    "pnr-mapping-summary.csv",
    "pnr-mapping.json",
    "vecsum-dfg-sim-report.json",
    "vecsum-dfg-sim-cycle-summary.csv",
    "vecsum-cgra-sim-report.json",
    "sim-comparison-report.json",
    "runtime-package.json",
    "sim-cycle-summary.csv",
    "rtl-manifest.json",
    "rtl-eda-report.json",
    "rtl-fpa-report.json",
    "rtl-fpa-summary.csv",
    "workload-report-bundle.json",
    "hardware-report-bundle.json",
    "dse-report-bundle.json",
    "full-stack-artifact-manifest.json",
    "e2e-demonstrator-summary.csv",
    "dse-candidate-summary.csv",
    "unsupported-scope-ledger.csv",
    "artifact-audit-summary.json",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_legacy_case(root: Path, name: str) -> None:
    case_dir = root / name
    case_dir.mkdir(parents=True)
    (case_dir / "main.cpp").write_text("int main() { return 0; }\n")
    (case_dir / f"{name}.cpp").write_text(f'#include "{name}.h"\n')
    (case_dir / f"{name}.h").write_text("#pragma once\n")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-artifact-chain-") as tmp:
        out_dir = Path(tmp)
        legacy_root = out_dir / "legacy-app"
        write_legacy_case(legacy_root, "legacy_missing")
        write_legacy_case(legacy_root, "vecadd")
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--legacy-app-root",
                str(legacy_root),
            ],
            "intermediate artifact chain",
        )

        missing = [name for name in EXPECTED_FILES if not (out_dir / name).is_file()]
        if missing:
            raise AssertionError(f"missing chain artifacts: {missing}")

        mapping_rows = read_csv_rows(out_dir / "pnr-mapping-summary.csv")
        vecsum_mapping_rows = [row for row in mapping_rows if row["workload"] == "vecsum"]
        if len(vecsum_mapping_rows) != 1:
            raise AssertionError(f"expected one vecsum mapping row, got {mapping_rows}")
        if (
            vecsum_mapping_rows[0]["mapping_id"] != "vecsum__g_t_vecsum_red_0_0__shared_reduction_adg"
            or vecsum_mapping_rows[0]["placed_records"] != "5"
            or vecsum_mapping_rows[0]["routed_edges"] != "0"
            or vecsum_mapping_rows[0]["unrouted_edges"] != "6"
            or vecsum_mapping_rows[0].get("status") != "fail"
        ):
            raise AssertionError(f"expected unrouted vecsum mapping evidence: {vecsum_mapping_rows[0]}")

        sim_rows = read_csv_rows(out_dir / "sim-cycle-summary.csv")
        vecsum_rows = [row for row in sim_rows if row["kernel"] == "vecsum"]
        if len(vecsum_rows) != 1:
            raise AssertionError(f"expected one vecsum sim row, got {sim_rows}")
        if (
            vecsum_rows[0]["dfg_sim_cycles"] == ""
            or vecsum_rows[0]["cgra_sim_cycles"] != ""
            or vecsum_rows[0].get("status") != "blocked"
        ):
            raise AssertionError(f"expected blocked vecsum simulator cycle evidence: {vecsum_rows[0]}")

        comparison = json.loads((out_dir / "sim-comparison-report.json").read_text())
        if comparison.get("kind") != "sim_comparison_report":
            raise AssertionError(f"unexpected simulation comparison report kind: {comparison}")
        if comparison.get("status") != "blocked" or comparison.get("workload") != "vecsum":
            raise AssertionError(f"unexpected simulation comparison report status: {comparison}")
        if comparison.get("performance_comparison_status") != "blocked":
            raise AssertionError(f"comparison should preserve performance blocked evidence: {comparison}")
        if comparison.get("difference_classification") != "unsupported_scope":
            raise AssertionError(f"comparison should classify unsupported route evidence: {comparison}")
        if comparison.get("dfg_sim_cycles") != 579 or comparison.get("cgra_sim_cycles") != 579:
            raise AssertionError(f"comparison should preserve simulator cycle values: {comparison}")

        runtime_package = json.loads((out_dir / "runtime-package.json").read_text())
        if runtime_package.get("kind") != "runtime_package":
            raise AssertionError(f"unexpected runtime package kind: {runtime_package}")
        if runtime_package.get("status") != "blocked" or runtime_package.get("workload") != "vecsum":
            raise AssertionError(f"unexpected runtime package status: {runtime_package}")
        if runtime_package.get("work_package_identity") != "work-package::vecsum::vecsum__g_t_vecsum_red_0_0__shared_reduction_adg":
            raise AssertionError(f"unexpected runtime work package identity: {runtime_package}")
        expected_launch = "launch::vecsum::vecsum__g_t_vecsum_red_0_0__shared_reduction_adg::test-app-fixture::vecsum::default"
        if runtime_package.get("launch_descriptor_identity") != expected_launch:
            raise AssertionError(f"unexpected runtime launch descriptor identity: {runtime_package}")
        if runtime_package.get("selected_mapping_artifact_identity") != "pnr-mapping":
            raise AssertionError(f"unexpected runtime mapping identity: {runtime_package}")
        if runtime_package.get("fabric_adg_identity") != "shared_reduction_adg":
            raise AssertionError(f"unexpected runtime fabric ADG identity: {runtime_package}")
        if runtime_package.get("data_movement_policy") != "simulated":
            raise AssertionError(f"unexpected runtime data movement policy: {runtime_package}")

        dse_rows = read_csv_rows(out_dir / "dse-candidate-summary.csv")
        vecsum_dse_rows = [row for row in dse_rows if row["workload"] == "vecsum"]
        if len(vecsum_dse_rows) != 1:
            raise AssertionError(f"expected one vecsum DSE row, got {dse_rows}")
        vecsum_dse = vecsum_dse_rows[0]
        expected_dse = {
            "mapping_id": "vecsum__g_t_vecsum_red_0_0__shared_reduction_adg",
            "cgra_sim_cycles": "",
            "frequency_mhz": "",
            "area_um2": "",
            "dynamic_power_mw": "",
            "energy_nj": "",
            "selection_status": "blocked",
        }
        for key, value in expected_dse.items():
            if vecsum_dse[key] != value:
                raise AssertionError(f"unexpected vecsum DSE {key}: {vecsum_dse}")
        expected_provenance = {
            "candidate_kind": "combined_full_stack_candidate",
            "objective_record": "objective::minimize_runtime",
            "policy_id": "deterministic_minimize_runtime_v1",
            "ordering_rule": "runtime_score_then_candidate_id",
        }
        for key, value in expected_provenance.items():
            if vecsum_dse.get(key) != value:
                raise AssertionError(f"unexpected vecsum DSE provenance {key}: {vecsum_dse}")
        input_artifacts = {entry for entry in vecsum_dse.get("input_artifacts", "").split(";") if entry}
        for artifact_name in (
            "pnr-mapping-summary",
            "pnr-mapping",
            "sim-cycle-summary",
            "vecsum-cgra-sim-report",
            "rtl-fpa-summary",
        ):
            if artifact_name not in input_artifacts:
                raise AssertionError(f"vecsum DSE input artifacts missed {artifact_name}: {vecsum_dse}")
        if vecsum_dse.get("metric_records", "") != "":
            raise AssertionError(f"blocked vecsum DSE row must not expose objective metrics: {vecsum_dse}")

        report_bundle = json.loads((out_dir / "workload-report-bundle.json").read_text())
        if report_bundle.get("kind") != "workload_report_bundle":
            raise AssertionError(f"unexpected workload report bundle kind: {report_bundle}")
        if report_bundle.get("workload") != "vecsum" or report_bundle.get("report_status") != "blocked":
            raise AssertionError(f"unexpected workload report bundle status: {report_bundle}")
        if report_bundle.get("selected_hardware_candidate_identity") != "shared_reduction_adg":
            raise AssertionError(f"unexpected workload report bundle hardware: {report_bundle}")
        if report_bundle.get("selected_mapping_artifact_identity") != "pnr-mapping":
            raise AssertionError(f"unexpected workload report bundle mapping identity: {report_bundle}")
        optional_identities = report_bundle.get("optional_artifact_identities", {})
        if not isinstance(optional_identities, dict):
            raise AssertionError(f"workload report bundle should include optional artifact identities: {report_bundle}")
        if optional_identities.get("simulation_comparison_report") != "sim-comparison-report":
            raise AssertionError(f"workload report bundle missed simulation comparison identity: {report_bundle}")
        if optional_identities.get("runtime_package") != "runtime-package":
            raise AssertionError(f"workload report bundle missed runtime package identity: {report_bundle}")
        if optional_identities.get("rtl_manifest") != "":
            raise AssertionError(f"workload report bundle must not consume blocked RTL manifest: {report_bundle}")
        bundle_metrics = {
            metric["metric_id"]: metric
            for metric in report_bundle.get("metric_records", [])
            if isinstance(metric, dict)
        }
        required_metric_ids = {
            "metric::vecsum::dfg_sim_cycles",
            "metric::vecsum::workload_size_items",
            "metric::shared_reduction_adg::frequency_mhz",
            "metric::shared_reduction_adg::area_um2",
            "metric::shared_reduction_adg::dynamic_power_mw",
            "metric::shared_reduction_adg::leakage_power_mw",
        }
        if not required_metric_ids.issubset(bundle_metrics):
            raise AssertionError(f"workload report bundle missed blocked evidence metrics: {report_bundle}")
        forbidden_metric_ids = {
            "metric::vecsum::cgra_sim_cycles",
            "metric::vecsum::estimated_runtime_us",
            "metric::vecsum::energy_nj",
        }
        if forbidden_metric_ids & set(bundle_metrics):
            raise AssertionError(f"blocked workload report bundle must not expose derived performance metrics: {report_bundle}")

        hardware_bundle = json.loads((out_dir / "hardware-report-bundle.json").read_text())
        if hardware_bundle.get("kind") != "hardware_report_bundle":
            raise AssertionError(f"unexpected hardware report bundle kind: {hardware_bundle}")
        expected_hardware = "test/pnr/shared_reduction_adg.mlir::shared_reduction_adg"
        if hardware_bundle.get("hardware_candidate_identity") != expected_hardware:
            raise AssertionError(f"unexpected hardware report bundle identity: {hardware_bundle}")
        if hardware_bundle.get("rtl_manifest_identity") != "":
            raise AssertionError(f"hardware report bundle must not consume blocked RTL manifest: {hardware_bundle}")
        if hardware_bundle.get("report_status") != "blocked":
            raise AssertionError(f"hardware report bundle should block without passing RTL evidence: {hardware_bundle}")
        if hardware_bundle.get("supported_workload_classes") != ["vecsum"]:
            raise AssertionError(f"unexpected hardware report supported workloads: {hardware_bundle}")
        eda_report = json.loads((out_dir / "rtl-eda-report.json").read_text())
        if eda_report.get("kind") != "eda_report" or eda_report.get("capability_class") != "rtl_lint":
            raise AssertionError(f"unexpected RTL EDA report: {eda_report}")
        if eda_report.get("status") == "pass":
            if hardware_bundle.get("eda_report_identities") != ["rtl-eda-report"]:
                raise AssertionError(f"hardware report missed passing EDA report: {hardware_bundle}")
        elif eda_report.get("status") == "blocked":
            if hardware_bundle.get("eda_report_identities") != []:
                raise AssertionError(f"hardware report should not consume blocked EDA report: {hardware_bundle}")
        else:
            raise AssertionError(f"unexpected RTL EDA report status: {eda_report}")

        dse_bundle = json.loads((out_dir / "dse-report-bundle.json").read_text())
        if dse_bundle.get("kind") != "dse_report_bundle":
            raise AssertionError(f"unexpected DSE report bundle kind: {dse_bundle}")
        if dse_bundle.get("report_status") != "blocked":
            raise AssertionError(f"DSE report bundle should block without selected candidate evidence: {dse_bundle}")
        if dse_bundle.get("selected_candidates") != []:
            raise AssertionError(f"blocked DSE bundle must not select candidates: {dse_bundle}")

        demonstrator_rows = read_csv_rows(out_dir / "e2e-demonstrator-summary.csv")
        vecsum_demo_rows = [row for row in demonstrator_rows if row["demonstrator"] == "app::vecsum::shared_reduction_adg"]
        if len(vecsum_demo_rows) != 1:
            raise AssertionError(f"expected one vecsum demonstrator row, got {demonstrator_rows}")
        vecsum_demo = vecsum_demo_rows[0]
        eda_status = eda_report.get("status")
        expected_rtl_status = "skipped" if eda_status == "pass" else str(eda_status or "blocked")
        if vecsum_demo["rtl_status"] != expected_rtl_status or vecsum_demo["fpa_status"] != "pass":
            raise AssertionError(
                f"demonstrator should expose RTL lint and passing analytic FPA evidence: {vecsum_demo}"
            )
        if vecsum_demo["report_status"] != "blocked":
            raise AssertionError(f"demonstrator should preserve blocked workload report evidence: {vecsum_demo}")
        hardware_demo_rows = [row for row in demonstrator_rows if row["demonstrator"] == "hardware::test/pnr/shared_reduction_adg.mlir::shared_reduction_adg"]
        if len(hardware_demo_rows) != 1:
            raise AssertionError(f"expected one shared_reduction_adg hardware row, got {demonstrator_rows}")
        hardware_demo = hardware_demo_rows[0]
        if hardware_demo["artifact_status"] != "pass" or hardware_demo["report_status"] != "blocked":
            raise AssertionError(f"unexpected shared_reduction_adg hardware demonstrator row: {hardware_demo}")
        cmsis_demo_rows = [row for row in demonstrator_rows if row["demonstrator"] == "cmsis::cmsis-dsp"]
        if len(cmsis_demo_rows) != 1:
            raise AssertionError(f"expected one CMSIS-DSP demonstrator row, got {demonstrator_rows}")
        cmsis_demo = cmsis_demo_rows[0]
        expected_cmsis_statuses = {
            "compat_status": "pass",
            "artifact_status": "pass",
            "mapping_status": "skipped",
            "sim_status": "skipped",
            "rtl_status": "skipped",
            "fpa_status": "skipped",
            "report_status": "skipped",
        }
        for key, value in expected_cmsis_statuses.items():
            if cmsis_demo[key] != value:
                raise AssertionError(f"unexpected CMSIS demonstrator {key}: {cmsis_demo}")

        import_rows = read_csv_rows(out_dir / "app-corpus-import-status.csv")
        states = {row["case"]: row["import_state"] for row in import_rows}
        if states != {"legacy_missing": "deferred", "vecadd": "accepted"}:
            raise AssertionError(f"unexpected app import states: {import_rows}")

        audit = json.loads((out_dir / "artifact-audit-summary.json").read_text())
        if audit.get("verdict") != "pass":
            raise AssertionError(f"expected chain audit pass, got {audit}")
        reviewed = {Path(review["artifact"]).name for review in audit.get("artifact_reviews", [])}
        expected_reviewed = set(EXPECTED_FILES) - {"artifact-audit-summary.json"}
        if reviewed != expected_reviewed:
            raise AssertionError(f"audit reviewed {reviewed}, expected {expected_reviewed}")
        if audit.get("cross_artifact_findings"):
            raise AssertionError(f"chain should not have cross-artifact findings: {audit}")

        manifest = json.loads((out_dir / "full-stack-artifact-manifest.json").read_text())
        manifest_artifacts = {
            artifact.get("logical_path")
            for artifact in manifest.get("artifacts", [])
            if isinstance(artifact, dict)
        }
        expected_manifest = set(EXPECTED_FILES) - {
            "artifact-audit-summary.json",
            "full-stack-artifact-manifest.json",
        }
        if manifest_artifacts != expected_manifest:
            raise AssertionError(f"manifest artifacts {manifest_artifacts} do not match {expected_manifest}")
        edges = {(edge["from"], edge["to"]) for edge in manifest.get("edges", [])}
        if ("e2e-demonstrator-summary", "dse-candidate-summary") in edges:
            raise AssertionError(f"manifest should not imply demonstrator feeds DSE: {edges}")
        required_edges = {
            ("old-app-corpus-inventory", "app-corpus-import-status"),
            ("app-corpus-import-status", "source-compat-summary"),
            ("source-compat-summary", "compiler-pipeline-summary"),
            ("pnr-mapping-summary", "e2e-demonstrator-summary"),
            ("pnr-mapping-summary", "dse-candidate-summary"),
            ("vecsum-dfg-sim-report", "sim-comparison-report"),
            ("vecsum-cgra-sim-report", "sim-comparison-report"),
            ("pnr-mapping", "sim-comparison-report"),
            ("pnr-mapping", "runtime-package"),
            ("vecsum-cgra-sim-report", "runtime-package"),
            ("sim-comparison-report", "runtime-package"),
            ("pnr-mapping", "rtl-manifest"),
            ("adg-hardware-summary", "rtl-manifest"),
            ("rtl-manifest", "rtl-eda-report"),
            ("rtl-manifest", "rtl-fpa-summary"),
            ("rtl-eda-report", "rtl-fpa-summary"),
            ("dataflow-primitive-coverage", "rtl-fpa-report"),
            ("adg-hardware-summary", "rtl-fpa-report"),
            ("rtl-manifest", "rtl-fpa-report"),
            ("rtl-eda-report", "rtl-fpa-report"),
            ("runtime-package", "workload-report-bundle"),
            ("pnr-mapping", "workload-report-bundle"),
            ("sim-comparison-report", "workload-report-bundle"),
            ("vecsum-cgra-sim-report", "workload-report-bundle"),
            ("rtl-fpa-summary", "workload-report-bundle"),
            ("dse-candidate-summary", "workload-report-bundle"),
            ("adg-hardware-summary", "hardware-report-bundle"),
            ("rtl-fpa-report", "hardware-report-bundle"),
            ("hardware-report-bundle", "e2e-demonstrator-summary"),
            ("dse-candidate-summary", "dse-report-bundle"),
            ("workload-report-bundle", "e2e-demonstrator-summary"),
            ("dse-candidate-summary", "unsupported-scope-ledger"),
        }
        if not required_edges.issubset(edges):
            raise AssertionError(f"manifest edges {edges} missing {required_edges - edges}")
        eda_to_hardware_edge = ("rtl-eda-report", "hardware-report-bundle")
        if hardware_bundle.get("eda_report_identities") == ["rtl-eda-report"]:
            if eda_to_hardware_edge not in edges:
                raise AssertionError(f"manifest missed consumed EDA report edge: {edges}")
        elif eda_to_hardware_edge in edges:
            raise AssertionError(f"manifest must not feed blocked EDA report to hardware bundle: {edges}")
        if ("cmsis-compiler-pipeline-summary", "dataflow-primitive-coverage") in edges:
            raise AssertionError(f"CMSIS pipeline summary must not feed app primitive coverage: {edges}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
