#!/usr/bin/env python3
"""Regression test for DSE candidate summary rows."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import artifact_test_common


HEADER = [
    "candidate",
    "workload",
    "hardware",
    "mapping_id",
    "objective",
    "cgra_sim_cycles",
    "frequency_mhz",
    "area_um2",
    "dynamic_power_mw",
    "energy_nj",
    "selection_status",
]


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="loom-dse-candidate-") as tmp:
        out_dir = Path(tmp)
        primitive, hardware = artifact_test_common.prepare_candidate_inputs(repo, out_dir)
        mapping = out_dir / "pnr-mapping-summary.csv"
        sim = out_dir / "sim-cycle-summary.csv"
        rtl_fpa = out_dir / "rtl-fpa-summary.csv"
        dse = out_dir / "dse-candidate-summary.csv"

        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/pnr/run_mapping_summary.sh",
                "--primitive-coverage",
                str(primitive),
                "--hardware-summary",
                str(hardware),
                "--output",
                str(mapping),
            ],
            "PnR mapping summary",
        )
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_sim_cycle_summary.sh",
                "--primitive-coverage",
                str(primitive),
                "--output",
                str(sim),
            ],
            "sim cycle summary",
        )
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_fpa_summary.sh",
                "--primitive-coverage",
                str(primitive),
                "--hardware-summary",
                str(hardware),
                "--output",
                str(rtl_fpa),
            ],
            "RTL/FPA summary",
        )
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/dse/run_candidate_summary.sh",
            dse,
            HEADER,
            "--artifact",
            str(mapping),
            "--artifact",
            str(sim),
            "--artifact",
            str(rtl_fpa),
            label="DSE candidate summary",
        )

        matches = [
            row
            for row in rows
            if row["workload"] == "vecadd" and row["hardware"].endswith("::pe_two_pes")
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one vecadd pe_two_pes candidate row, got {rows}")
        row = matches[0]
        if not row["candidate"].startswith("candidate::vecadd::"):
            raise AssertionError(f"unexpected candidate id: {row}")
        for column in ("mapping_id", "cgra_sim_cycles", "frequency_mhz", "area_um2", "dynamic_power_mw", "energy_nj"):
            if row[column] != "":
                raise AssertionError(f"blocked candidate must not fake {column}: {row}")
        if row["objective"] != "minimize_runtime":
            raise AssertionError(f"unexpected objective: {row}")
        if row["selection_status"] != "blocked":
            raise AssertionError(f"candidate should be blocked: {row}")
        if "missing mapping, simulator, or FPA evidence" not in row.get("diagnostic", ""):
            raise AssertionError(f"unexpected diagnostic: {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
