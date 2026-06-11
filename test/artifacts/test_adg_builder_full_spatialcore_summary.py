#!/usr/bin/env python3
"""Regression test for an ADG Builder generated full SpatialCore row."""

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


def assert_fields(row: dict[str, str], expected: dict[str, str], *, label: str) -> None:
    for key, value in expected.items():
        if row.get(key) != value:
            raise AssertionError(f"{label} {key}={row.get(key)!r}, expected {value!r}: {row}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    builder = repo / "build" / "tools" / "loom-adg-builder-test" / "loom-adg-builder-test"
    if not builder.is_file():
        raise AssertionError(f"missing loom-adg-builder-test: {builder}")

    with artifact_test_common.repo_temp_dir(repo, "loom-adg-builder-full-spatialcore-") as tmp:
        out_dir = Path(tmp)
        hardware_mlir = out_dir / "adg-builder-full-spatialcore.mlir"
        artifact_test_common.require_success(
            repo,
            [
                str(builder),
                "--full-spatialcore",
                "--output",
                str(hardware_mlir),
            ],
            "ADG Builder generated full SpatialCore",
        )
        generated = hardware_mlir.read_text()
        required_fragments = [
            "fabric.module @full_spatialcore_adg",
            "fabric.pe [spatial]",
            "fabric.pe [temporal]",
            "fabric.switch [spatial]",
            "fabric.switch [temporal]",
            "fabric.mem [spatial]",
            "fabric.mem [temporal]",
            "fabric.boundary [s2t]",
            "fabric.fifo",
            "fabric.pe @ALU",
            "fabric.instantiate @ALU",
        ]
        for fragment in required_fragments:
            if fragment not in generated:
                raise AssertionError(f"generated full SpatialCore MLIR missed {fragment!r}:\n{generated}")

        output = out_dir / "adg-hardware-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/fabric/run_adg_hardware_summary.sh",
            output,
            HEADER,
            "--input",
            str(hardware_mlir),
            "--input-recipe-identity",
            f"{hardware_mlir}=adg-builder::full-spatialcore",
            label="ADG Builder full SpatialCore hardware summary",
        )

        generated_identity = (
            f"{hardware_mlir.resolve().relative_to(repo).as_posix()}::full_spatialcore_adg"
        )
        matches = [row for row in rows if row.get("hardware") == generated_identity]
        if len(matches) != 1:
            raise AssertionError(f"expected one full SpatialCore row for {generated_identity!r}, got {rows}")
        row = matches[0]
        assert_fields(
            row,
            {
                "topology_class": "fabric_module_template",
                "node_count": "10",
                "link_count": "0",
                "verify_status": "pass",
                "tile_kinds": "mem;pe;switch",
                "schedule_kinds": "spatial;temporal",
                "adg_builder_recipe_identity": "adg-builder::full-spatialcore",
                "node_kinds": "",
            },
            label="ADG Builder full SpatialCore",
        )
        if "fabric.module template verified" not in row["diagnostic"]:
            raise AssertionError(f"unexpected full SpatialCore diagnostic: {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
