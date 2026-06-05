#!/usr/bin/env python3
"""Regression test for PnR mapping summary candidate rows."""

from __future__ import annotations

import sys
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
    with artifact_test_common.repo_temp_dir(repo, "loom-pnr-mapping-") as tmp:
        out_dir = Path(tmp)
        mapping = out_dir / "pnr-mapping-summary.csv"
        primitive, hardware = artifact_test_common.prepare_candidate_inputs(repo, out_dir)
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

        dfg_dir = out_dir / "vecsum-dfg"
        result = artifact_test_common.run_command(
            repo,
            [
                "env",
                f"BUILD_DIR={dfg_dir}",
                "bash",
                "test/app/vecsum/dfg_check.sh",
            ],
        )
        if result.returncode != 0:
            raise AssertionError(
                "vecsum DFG check with explicit build dir failed\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        mapped = out_dir / "pnr-mapping-summary-pass.csv"
        artifact = out_dir / "pnr-mapping.json"
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/pnr/run_mapping_summary.sh",
            mapped,
            HEADER,
            "--dfg-mlir",
            str(dfg_dir / "main_func.dfg.mlir"),
            "--graph",
            "g_t_vecsum_red_0_0",
            "--hardware-mlir",
            "test/pnr/shared_reduction_adg.mlir",
            "--hardware",
            "shared_reduction_adg",
            "--workload",
            "vecsum",
            "--artifact",
            str(artifact),
            label="PnR mapping summary explicit mapper",
        )
        if len(rows) != 1:
            raise AssertionError(f"expected one explicit mapping row, got {rows}")
        row = rows[0]
        expected = {
            "workload": "vecsum",
            "hardware": "shared_reduction_adg",
            "mapping_id": "vecsum__shared_reduction_adg",
            "placed_records": "6",
            "routed_edges": "8",
            "unrouted_edges": "0",
            "status": "pass",
        }
        for key, value in expected.items():
            if row[key] != value:
                raise AssertionError(f"explicit mapping {key}={row[key]!r}, expected {value!r}")
        if not artifact.is_file():
            raise AssertionError("explicit mapping did not emit JSON artifact")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
