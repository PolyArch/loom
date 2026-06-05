#!/usr/bin/env python3
"""Regression test for PnR mapping summary candidate rows."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import artifact_test_common


HEADER = [
    "workload",
    "hardware",
    "mapping_id",
    "placed_records",
    "routed_edges",
    "unrouted_edges",
    "status",
]


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="loom-pnr-mapping-") as tmp:
        out_dir = Path(tmp)
        primitive = out_dir / "dataflow-primitive-coverage.csv"
        hardware = out_dir / "adg-hardware-summary.csv"
        mapping = out_dir / "pnr-mapping-summary.csv"

        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/dataflow/run_primitive_coverage.sh",
                "--case",
                "vecadd",
                "--output",
                str(primitive),
            ],
            "primitive coverage summary",
        )
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/fabric/run_adg_hardware_summary.sh",
                "--input",
                "test/fabric/unit/pe/valid.mlir",
                "--output",
                str(hardware),
            ],
            "ADG hardware summary",
        )
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/pnr/run_mapping_summary.sh",
            mapping,
            HEADER,
            "--primitive-coverage",
            str(primitive),
            "--hardware-summary",
            str(hardware),
            label="PnR mapping summary",
        )

        matches = [
            row
            for row in rows
            if row["workload"] == "vecadd" and row["hardware"].endswith("::pe_two_pes")
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one vecadd to pe_two_pes row, got {rows}")
        row = matches[0]
        for column in ("mapping_id", "placed_records", "routed_edges", "unrouted_edges"):
            if row[column] != "":
                raise AssertionError(f"blocked row must not fake {column}: {row}")
        if row["status"] != "blocked":
            raise AssertionError(f"mapping row should be blocked: {row}")
        if "PnR mapping artifact producer is not implemented yet" not in row.get("diagnostic", ""):
            raise AssertionError(f"unexpected diagnostic: {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
