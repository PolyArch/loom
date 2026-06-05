#!/usr/bin/env python3
"""Regression test for RTL/FPA summary candidate rows."""

from __future__ import annotations

import sys
import tempfile
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
]


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="loom-rtl-fpa-") as tmp:
        out_dir = Path(tmp)
        primitive = out_dir / "dataflow-primitive-coverage.csv"
        hardware = out_dir / "adg-hardware-summary.csv"
        fpa = out_dir / "rtl-fpa-summary.csv"

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
            "test/rtl/run_rtl_fpa_summary.sh",
            fpa,
            HEADER,
            "--primitive-coverage",
            str(primitive),
            "--hardware-summary",
            str(hardware),
            label="RTL/FPA summary",
        )

        matches = [
            row
            for row in rows
            if row["workload"] == "vecadd" and row["hardware"].endswith("::pe_two_pes")
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one vecadd pe_two_pes row, got {rows}")
        row = matches[0]
        for column in ("rtl_lint_status", "rtl_sim_status", "synth_status", "status"):
            if row[column] != "blocked":
                raise AssertionError(f"{column} should be blocked: {row}")
        for column in ("frequency_mhz", "area_um2", "dynamic_power_mw", "leakage_power_mw"):
            if row[column] != "":
                raise AssertionError(f"blocked row must not fake {column}: {row}")
        if "RTL/FPA backend evidence is not available yet" not in row.get("diagnostic", ""):
            raise AssertionError(f"unexpected diagnostic: {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
