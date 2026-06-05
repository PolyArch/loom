#!/usr/bin/env python3
"""Regression test for ADG hardware summary evidence rows."""

from __future__ import annotations

import sys
from pathlib import Path

import artifact_test_common


HEADER = ["hardware", "topology_class", "node_count", "link_count", "verify_status", "diagnostic"]


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
    }
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"pe_two_pes {key}={row[key]!r}, expected {value!r}")
    if "fabric.module template verified" not in row["diagnostic"]:
        raise AssertionError(f"unexpected diagnostic: {row}")


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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
