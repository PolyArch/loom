#!/usr/bin/env python3
"""Regression test for an ADG Builder generated full-stack artifact chain."""

from __future__ import annotations

import sys
from pathlib import Path

import artifact_test_common
from test_intermediate_artifact_chain_xor_block import (
    assert_fields,
    read_csv_rows,
    read_json_object,
    single_row,
)


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-adg-builder-chain-") as tmp:
        out_dir = Path(tmp)
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--case",
                "xor_block",
                "--hardware-source",
                "adg-builder",
            ],
            "ADG Builder generated intermediate artifact chain",
        )

        generated_adg = out_dir / "adg-builder-shared-reduction-adg.mlir"
        if not generated_adg.is_file():
            raise AssertionError(f"ADG Builder chain missed generated hardware MLIR: {generated_adg}")
        checked_in_adg = repo / "test/pnr/shared_reduction_adg.mlir"

        def canonical_adg_text(path: Path) -> str:
            return "\n".join(
                line for line in path.read_text().splitlines()
                if not line.lstrip().startswith("//")
            ).strip()

        if canonical_adg_text(generated_adg) != canonical_adg_text(checked_in_adg):
            raise AssertionError(
                "ADG Builder shared-reduction output diverged from checked-in PnR fixture"
            )

        generated_hardware_identity = (
            f"{generated_adg.resolve().relative_to(repo).as_posix()}::shared_reduction_adg"
        )
        hardware_row = single_row(
            read_csv_rows(out_dir / "adg-hardware-summary.csv"),
            key="hardware",
            value=generated_hardware_identity,
            label="ADG Builder generated hardware",
        )
        assert_fields(
            hardware_row,
            {
                "topology_class": "fabric_module_template",
                "node_count": "237",
                "link_count": "0",
                "verify_status": "pass",
                "tile_kinds": "mem;pe;switch",
                "schedule_kinds": "spatial",
                "adg_builder_recipe_identity": "adg-builder::shared-reduction",
            },
            label="ADG Builder hardware summary",
        )
        fixture_summary = out_dir / "checked-in-shared-reduction-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/fabric/run_adg_hardware_summary.sh",
                "--input",
                "test/pnr/shared_reduction_adg.mlir",
                "--input-recipe-identity",
                "test/pnr/shared_reduction_adg.mlir=adg-builder::shared-reduction",
                "--output",
                str(fixture_summary),
            ],
            "checked-in shared reduction hardware summary",
        )
        fixture_row = single_row(
            read_csv_rows(fixture_summary),
            key="hardware",
            value="test/pnr/shared_reduction_adg.mlir::shared_reduction_adg",
            label="checked-in shared reduction hardware",
        )
        for key in (
            "topology_class",
            "node_count",
            "link_count",
            "verify_status",
            "tile_kinds",
            "schedule_kinds",
            "adg_builder_recipe_identity",
        ):
            if hardware_row[key] != fixture_row[key]:
                raise AssertionError(
                    f"ADG Builder shared reduction summary diverged from checked-in fabric {key}: "
                    f"builder={hardware_row} fixture={fixture_row}"
                )

        dotproduct_dir = out_dir / "dotproduct-dfg"
        artifact_test_common.require_success(
            repo,
            [
                "env",
                f"BUILD_DIR={dotproduct_dir}",
                "LOOM_CC=build/bin/loom-cc",
                "LOOM_RAISE=build/bin/loom-raise",
                "LOOM_LOWER=build/bin/loom-lower",
                "LOOM_RAISE_OPT=build/bin/loom-raise-opt",
                "bash",
                "test/app/dotproduct/dfg_check.sh",
            ],
            "dotproduct DFG for ADG Builder generated hardware",
        )
        dotproduct_mapping = out_dir / "dotproduct-adg-builder-mapping.csv"
        dotproduct_mapping_artifact = out_dir / "dotproduct-adg-builder-mapping.json"
        artifact_test_common.require_success(
            repo,
            [
                "build/tools/loom-pnr-map/loom-pnr-map",
                "--dfg-mlir",
                str(dotproduct_dir / "main_func.dfg.mlir"),
                "--graph",
                "g_t_dotproduct_red_0_0",
                "--hardware-mlir",
                str(generated_adg),
                "--hardware",
                "shared_reduction_adg",
                "--workload",
                "dotproduct",
                "--output",
                str(dotproduct_mapping),
                "--artifact",
                str(dotproduct_mapping_artifact),
            ],
            "dotproduct PnR on ADG Builder generated shared reduction hardware",
        )
        dotproduct_mapping_row = single_row(
            read_csv_rows(dotproduct_mapping),
            key="workload",
            value="dotproduct",
            label="dotproduct ADG Builder mapping",
        )
        assert_fields(
            dotproduct_mapping_row,
            {
                "hardware": "shared_reduction_adg",
                "mapping_id": "dotproduct__g_t_dotproduct_red_0_0__shared_reduction_adg",
                "status": "pass",
                "unrouted_edges": "0",
                "unplaced_records": "0",
            },
            label="dotproduct ADG Builder mapping",
        )

        mapping_row = single_row(
            read_csv_rows(out_dir / "pnr-mapping-summary.csv"),
            key="workload",
            value="xor_block",
            label="xor_block mapping",
        )
        assert_fields(
            mapping_row,
            {
                "hardware": "shared_reduction_adg",
                "mapping_id": "xor_block__g_t_xor_block_0_0__shared_reduction_adg",
                "status": "fail",
            },
            label="xor_block mapping",
        )

        cgra_report = read_json_object(out_dir / "xor_block-cgra-sim-report.json")
        assert_fields(
            cgra_report,
            {
                "workload": "xor_block",
                "hardware": "shared_reduction_adg",
                "mapping_id": "xor_block__g_t_xor_block_0_0__shared_reduction_adg",
                "status": "blocked",
                "dfg_cycles": 452,
                "hardware_aware_cycles": 452,
                "hardware_artifact": str(generated_adg),
            },
            label="xor_block CGRA-sim report",
        )
        if cgra_report["hardware_aware_cycles"] < cgra_report["dfg_cycles"]:
            raise AssertionError(f"CGRA-sim must not be more optimistic than DFG-sim: {cgra_report}")

        rtl_manifest = read_json_object(out_dir / "rtl-manifest.json")
        assert_fields(
            rtl_manifest,
            {
                "kind": "rtl_manifest",
                "mode": "mapped_workload_rtl",
                "source_fabric_adg_identity": generated_hardware_identity,
                "mapping_artifact_identity": "pnr-mapping",
                "status": "blocked",
            },
            label="ADG Builder RTL manifest",
        )

        hardware_bundle = read_json_object(out_dir / "hardware-report-bundle.json")
        assert_fields(
            hardware_bundle,
            {
                "hardware_candidate_identity": generated_hardware_identity,
                "fabric_adg_identity": generated_hardware_identity.rsplit("::", 1)[0],
                "adg_builder_recipe_identity": "adg-builder::shared-reduction",
                "rtl_manifest_identity": "",
                "report_status": "blocked",
            },
            label="ADG Builder hardware report bundle",
        )
        if hardware_bundle.get("supported_workload_classes") != ["xor_block"]:
            raise AssertionError(f"hardware report should cite xor_block FPA support: {hardware_bundle}")

        audit = read_json_object(out_dir / "artifact-audit-summary.json")
        if audit.get("verdict") != "pass":
            raise AssertionError(f"expected ADG Builder chain audit pass, got {audit}")
        if audit.get("cross_artifact_findings"):
            raise AssertionError(f"ADG Builder chain should not have cross-artifact findings: {audit}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
