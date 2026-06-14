#!/usr/bin/env python3
"""Regression test for ADG hardware summary evidence rows."""

from __future__ import annotations

import sys
from pathlib import Path

import artifact_test_common


HEADER = [
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


def assert_pe_two_pes(rows: list[dict[str, str]]) -> None:
    matches = [row for row in rows if row["hardware"].endswith("::pe_two_pes")]
    if len(matches) != 1:
        raise AssertionError(f"expected one pe_two_pes row, got {rows}")
    row = matches[0]
    expected = {
        "topology_class": "fabric_module_template",
        "node_count": "2",
        "link_count": "0",
        "verify_status": "pass",
        "tile_kinds": "pe",
        "schedule_kinds": "spatial",
        "adg_builder_recipe_identity": "",
    }
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"pe_two_pes {key}={row[key]!r}, expected {value!r}")
    if "fabric.module template verified" not in row["diagnostic"]:
        raise AssertionError(f"unexpected diagnostic: {row}")


def assert_shared_reduction_adg(rows: list[dict[str, str]]) -> None:
    matches = [row for row in rows if row["hardware"].endswith("::shared_reduction_adg")]
    if len(matches) != 1:
        raise AssertionError(f"expected one shared_reduction_adg row, got {rows}")
    row = matches[0]
    expected = {
        "topology_class": "fabric_module_template",
        "node_count": "44",
        "link_count": "0",
        "verify_status": "pass",
        "tile_kinds": "mem;pe;switch",
        "schedule_kinds": "spatial",
        "adg_builder_recipe_identity": "adg-builder::shared-reduction",
    }
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"shared_reduction_adg {key}={row[key]!r}, expected {value!r}")
    if "fabric.module template verified" not in row["diagnostic"]:
        raise AssertionError(f"unexpected diagnostic: {row}")


def assert_dotproduct_fmuladd_adg(rows: list[dict[str, str]]) -> None:
    matches = [row for row in rows if row["hardware"].endswith("::dotproduct_fmuladd_adg")]
    if len(matches) != 1:
        raise AssertionError(f"expected one dotproduct_fmuladd_adg row, got {rows}")
    row = matches[0]
    expected = {
        "topology_class": "fabric_module_template",
        "node_count": "3",
        "link_count": "0",
        "verify_status": "pass",
        "tile_kinds": "mem;pe",
        "schedule_kinds": "spatial",
        "adg_builder_recipe_identity": "",
    }
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"dotproduct_fmuladd_adg {key}={row[key]!r}, expected {value!r}")
    if "fabric.module template verified" not in row["diagnostic"]:
        raise AssertionError(f"unexpected diagnostic: {row}")


def assert_byte_swap_store_adg(rows: list[dict[str, str]]) -> None:
    matches = [row for row in rows if row["hardware"].endswith("::byte_swap_store_adg")]
    if len(matches) != 1:
        raise AssertionError(f"expected one byte_swap_store_adg row, got {rows}")
    row = matches[0]
    expected = {
        "topology_class": "fabric_module_template",
        "node_count": "4",
        "link_count": "0",
        "verify_status": "pass",
        "tile_kinds": "mem;pe",
        "schedule_kinds": "spatial",
        "adg_builder_recipe_identity": "",
    }
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"byte_swap_store_adg {key}={row[key]!r}, expected {value!r}")
    if "fabric.module template verified" not in row["diagnostic"]:
        raise AssertionError(f"unexpected diagnostic: {row}")


def assert_shared_vector_alu_adg(rows: list[dict[str, str]]) -> None:
    matches = [row for row in rows if row["hardware"].endswith("::shared_vector_alu_adg")]
    if len(matches) != 1:
        raise AssertionError(f"expected one shared_vector_alu_adg row, got {rows}")
    row = matches[0]
    expected = {
        "topology_class": "fabric_module_template",
        "node_count": "8",
        "link_count": "0",
        "verify_status": "pass",
        "tile_kinds": "mem;pe;switch",
        "schedule_kinds": "spatial",
        "adg_builder_recipe_identity": "",
    }
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"shared_vector_alu_adg {key}={row[key]!r}, expected {value!r}")
    if "fabric.module template verified" not in row["diagnostic"]:
        raise AssertionError(f"unexpected diagnostic: {row}")


def assert_minimal_spatial_adg(rows: list[dict[str, str]]) -> None:
    matches = [row for row in rows if row["hardware"].endswith("::minimal_spatial_adg")]
    if len(matches) != 1:
        raise AssertionError(f"expected one minimal_spatial_adg row, got {rows}")
    row = matches[0]
    expected = {
        "topology_class": "fabric_module_template",
        "node_count": "3",
        "link_count": "0",
        "verify_status": "pass",
        "tile_kinds": "mem;pe;switch",
        "schedule_kinds": "spatial",
        "adg_builder_recipe_identity": "adg-builder::minimal-spatial",
    }
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"minimal_spatial_adg {key}={row[key]!r}, expected {value!r}")
    if "fabric.module template verified" not in row["diagnostic"]:
        raise AssertionError(f"unexpected diagnostic: {row}")


def assert_minimal_temporal_adg(rows: list[dict[str, str]]) -> None:
    matches = [row for row in rows if row["hardware"].endswith("::minimal_temporal_adg")]
    if len(matches) != 1:
        raise AssertionError(f"expected one minimal_temporal_adg row, got {rows}")
    row = matches[0]
    expected = {
        "topology_class": "fabric_module_template",
        "node_count": "3",
        "link_count": "0",
        "verify_status": "pass",
        "tile_kinds": "mem;pe;switch",
        "schedule_kinds": "temporal",
        "adg_builder_recipe_identity": "adg-builder::minimal-temporal",
    }
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"minimal_temporal_adg {key}={row[key]!r}, expected {value!r}")
    if "fabric.module template verified" not in row["diagnostic"]:
        raise AssertionError(f"unexpected diagnostic: {row}")


def assert_quoted_named_pe(rows: list[dict[str, str]]) -> None:
    matches = [row for row in rows if "quoted module" in row["hardware"]]
    if len(matches) != 1:
        raise AssertionError(f"expected one quoted module row, got {rows}")
    row = matches[0]
    expected = {
        "topology_class": "fabric_module_template",
        "node_count": "1",
        "link_count": "0",
        "verify_status": "pass",
        "tile_kinds": "pe",
        "schedule_kinds": "spatial",
        "adg_builder_recipe_identity": "",
    }
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"quoted module {key}={row[key]!r}, expected {value!r}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-adg-hardware-") as tmp:
        output = Path(tmp) / "adg-hardware-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/fabric/run_adg_hardware_summary.sh",
            output,
            HEADER,
            "--input",
            "test/fabric/unit/pe/valid.mlir",
            label="ADG hardware summary",
        )
        assert_pe_two_pes(rows)

        default_output = Path(tmp) / "adg-hardware-summary-default.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/fabric/run_adg_hardware_summary.sh",
            default_output,
            HEADER,
            label="default ADG hardware summary",
        )
        assert_pe_two_pes(rows)
        assert_shared_reduction_adg(rows)
        assert_dotproduct_fmuladd_adg(rows)
        assert_byte_swap_store_adg(rows)
        assert_shared_vector_alu_adg(rows)
        assert_minimal_spatial_adg(rows)
        assert_minimal_temporal_adg(rows)

        quoted_input = Path(tmp) / "quoted-named-pe.mlir"
        quoted_input.write_text(
            """fabric.module @"quoted module"(%a : !fabric.bits<32>) {
  fabric.pe @\"ALU 0\" [spatial] (!fabric.bits<32>) -> (!fabric.bits<32>) {
  ^bb0(%pa: !fabric.bits<32>):
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
    fabric.yield %pa : !fabric.bits<32>
  }
  fabric.yield
}
"""
        )
        quoted_output = Path(tmp) / "adg-hardware-summary-quoted.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/fabric/run_adg_hardware_summary.sh",
            quoted_output,
            HEADER,
            "--input",
            str(quoted_input),
            label="quoted ADG hardware summary",
        )
        assert_quoted_named_pe(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
