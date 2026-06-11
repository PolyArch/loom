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
                "node_count": "4",
                "link_count": "0",
                "verify_status": "pass",
                "tile_kinds": "mem;pe",
                "schedule_kinds": "spatial",
                "adg_builder_recipe_identity": "adg-builder::shared-reduction",
            },
            label="ADG Builder hardware summary",
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
                "status": "pass",
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
                "status": "pass",
                "dfg_cycles": 448,
                "hardware_aware_cycles": 466,
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
                "source_fabric_adg_identity": generated_hardware_identity,
                "status": "pass",
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
                "report_status": "pass",
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
