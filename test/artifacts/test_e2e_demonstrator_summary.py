#!/usr/bin/env python3
"""Regression test for end-to-end demonstrator summary rows."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import artifact_test_common


HEADER = [
    "demonstrator",
    "compat_status",
    "artifact_status",
    "mapping_status",
    "sim_status",
    "rtl_status",
    "fpa_status",
    "report_status",
]


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="loom-e2e-demonstrator-") as tmp:
        out_dir = Path(tmp)
        source = out_dir / "source-compat-summary.csv"
        primitive, hardware = artifact_test_common.prepare_candidate_inputs(repo, out_dir)
        mapping = out_dir / "pnr-mapping-summary.csv"
        sim = out_dir / "sim-cycle-summary.csv"
        rtl_fpa = out_dir / "rtl-fpa-summary.csv"
        manifest = out_dir / "full-stack-artifact-manifest.json"
        demonstrator = out_dir / "e2e-demonstrator-summary.csv"

        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/app/run_source_compat_summary.sh",
                "--case",
                "vecadd",
                "--output",
                str(source),
            ],
            "source compatibility summary",
        )
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
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_artifact_manifest.sh",
                "--artifact",
                str(source),
                "--artifact",
                str(mapping),
                "--artifact",
                str(sim),
                "--artifact",
                str(rtl_fpa),
                "--output",
                str(manifest),
            ],
            "artifact manifest",
        )
        rows = artifact_test_common.run_csv_summary(
            repo,
            "test/e2e/run_demonstrator_summary.sh",
            demonstrator,
            HEADER,
            "--artifact",
            str(source),
            "--artifact",
            str(mapping),
            "--artifact",
            str(sim),
            "--artifact",
            str(rtl_fpa),
            "--artifact",
            str(manifest),
            label="end-to-end demonstrator summary",
        )

        matches = [row for row in rows if "vecadd" in row["demonstrator"] and row["demonstrator"].endswith("pe_two_pes")]
        if len(matches) != 1:
            raise AssertionError(f"expected one vecadd pe_two_pes demonstrator row, got {rows}")
        row = matches[0]
        expected = {
            "compat_status": "pass",
            "artifact_status": "pass",
            "mapping_status": "blocked",
            "sim_status": "blocked",
            "rtl_status": "blocked",
            "fpa_status": "blocked",
            "report_status": "blocked",
        }
        for key, value in expected.items():
            if row[key] != value:
                raise AssertionError(f"{key}={row[key]!r}, expected {value!r}: {row}")
        if "workload report bundle is not available yet" not in row.get("diagnostic", ""):
            raise AssertionError(f"unexpected diagnostic: {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
