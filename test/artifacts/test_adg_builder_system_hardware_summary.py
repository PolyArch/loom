#!/usr/bin/env python3
"""Regression test for an ADG Builder generated system-level hardware row."""

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


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    builder = repo / "build" / "tools" / "loom-adg-builder-test" / "loom-adg-builder-test"
    if not builder.is_file():
        raise AssertionError(f"missing loom-adg-builder-test: {builder}")

    with artifact_test_common.repo_temp_dir(repo, "loom-adg-builder-system-") as tmp:
        out_dir = Path(tmp)
        system_mlir = out_dir / "adg-builder-heterogeneous-soc.mlir"
        artifact_test_common.require_success(
            repo,
            [
                str(builder),
                "--heterogeneous-soc",
                "--output",
                str(system_mlir),
            ],
            "ADG Builder generated heterogeneous SoC",
        )
        if not system_mlir.is_file():
            raise AssertionError(f"ADG Builder did not create heterogeneous SoC MLIR: {system_mlir}")

        generated = system_mlir.read_text()
        required_fragments = [
            "fabric.module @shared_reduction_adg",
            "fabric.system @heterogeneous_dual_accel_soc",
            "fabric.node @host0",
            "fabric.node @acc0",
            "fabric.node @fft0",
            "fabric.node @dram0",
            "fabric.link src = @acc0",
        ]
        for fragment in required_fragments:
            if fragment not in generated:
                raise AssertionError(f"generated system MLIR missed {fragment!r}:\n{generated}")

        output = out_dir / "adg-hardware-summary.csv"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/fabric/run_adg_hardware_summary.sh",
            output,
            HEADER,
            "--input",
            str(system_mlir),
            "--input-recipe-identity",
            f"{system_mlir}=adg-builder::heterogeneous-soc",
            label="ADG Builder system hardware summary",
        )

        generated_system_identity = (
            f"{system_mlir.resolve().relative_to(repo).as_posix()}::heterogeneous_dual_accel_soc"
        )
        system_row = single_row(
            rows,
            key="hardware",
            value=generated_system_identity,
            label="ADG Builder generated fabric.system",
        )
        assert_fields(
            system_row,
            {
                "topology_class": "fabric_system",
                "node_count": "5",
                "link_count": "20",
                "verify_status": "pass",
                "tile_kinds": "",
                "schedule_kinds": "",
                "adg_builder_recipe_identity": "adg-builder::heterogeneous-soc",
                "node_kinds": "acc_core;cache;fixed_accelerator;host_core;memory",
            },
            label="ADG Builder generated fabric.system",
        )
        if "fabric.system verified" not in system_row["diagnostic"]:
            raise AssertionError(f"unexpected system diagnostic: {system_row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
