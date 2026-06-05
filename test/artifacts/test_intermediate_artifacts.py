#!/usr/bin/env python3
"""Regression tests for the intermediate artifact gate scaffold."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


CSV_COMMANDS = [
    (
        "test/app/run_source_compat_summary.sh",
        "source-compat-summary.csv",
        ["case", "suite", "native_status", "loom_status", "mode", "diagnostic"],
    ),
    (
        "test/app/run_compiler_pipeline_summary.sh",
        "compiler-pipeline-summary.csv",
        [
            "case",
            "suite",
            "llvm_ir_status",
            "raised_mlir_status",
            "dataflow_status",
            "diagnostic",
        ],
    ),
    (
        "test/dataflow/run_primitive_coverage.sh",
        "dataflow-primitive-coverage.csv",
        ["workload", "primitive", "op_count", "dfg_sim_status", "diagnostic"],
    ),
    (
        "test/fabric/run_adg_hardware_summary.sh",
        "adg-hardware-summary.csv",
        ["hardware", "topology_class", "node_count", "link_count", "verify_status", "diagnostic"],
    ),
    (
        "test/pnr/run_mapping_summary.sh",
        "pnr-mapping-summary.csv",
        [
            "workload",
            "hardware",
            "mapping_id",
            "placed_records",
            "routed_edges",
            "unrouted_edges",
            "status",
        ],
    ),
    (
        "test/app/run_sim_cycle_summary.sh",
        "sim-cycle-summary.csv",
        ["kernel", "dfg_sim_cycles", "cgra_sim_cycles"],
    ),
    (
        "test/rtl/run_rtl_fpa_summary.sh",
        "rtl-fpa-summary.csv",
        [
            "hardware",
            "workload",
            "rtl_lint_status",
            "rtl_sim_status",
            "synth_status",
            "frequency_mhz",
            "area_um2",
            "dynamic_power_mw",
            "leakage_power_mw",
        ],
    ),
    (
        "test/e2e/run_demonstrator_summary.sh",
        "e2e-demonstrator-summary.csv",
        [
            "demonstrator",
            "compat_status",
            "artifact_status",
            "mapping_status",
            "sim_status",
            "rtl_status",
            "fpa_status",
            "report_status",
        ],
    ),
    (
        "test/dse/run_candidate_summary.sh",
        "dse-candidate-summary.csv",
        [
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
        ],
    ),
    (
        "test/e2e/run_unsupported_scope_ledger.sh",
        "unsupported-scope-ledger.csv",
        ["stage", "case", "artifact", "reason", "owner", "blocking_input"],
    ),
]


JSON_COMMANDS = [
    (
        "test/e2e/run_artifact_manifest.sh",
        "full-stack-artifact-manifest.json",
        {"schema_version", "run_id", "artifacts", "edges", "diagnostics"},
    ),
]


def run_command(repo: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames or [], rows


def assert_csv_artifact(
    path: Path,
    required_first_columns: list[str],
    *,
    allow_pass_rows: bool = False,
) -> None:
    header, rows = read_csv(path)
    if header[: len(required_first_columns)] != required_first_columns:
        raise AssertionError(
            f"{path.name}: header {header[:len(required_first_columns)]} "
            f"does not match {required_first_columns}"
        )
    if not rows:
        raise AssertionError(f"{path.name}: expected at least one diagnostic row")
    for row in rows:
        if None in row:
            raise AssertionError(f"{path.name}: row has extra unnamed cells: {row}")
        missing = [key for key, value in row.items() if value is None]
        if missing:
            raise AssertionError(f"{path.name}: row is missing values for {missing}")
    statuses = []
    for row in rows:
        statuses.extend(
            value
            for key, value in row.items()
            if key.endswith("_status") or key in {"status", "selection_status"}
        )
    if "pass" in statuses and not allow_pass_rows:
        raise AssertionError(f"{path.name}: scaffold rows must not claim pass evidence")


def assert_json_artifact(path: Path, required_keys: set[str]) -> None:
    data = json.loads(path.read_text())
    missing = sorted(required_keys - set(data))
    if missing:
        raise AssertionError(f"{path.name}: missing keys {missing}")
    if data.get("schema_version") != 1:
        raise AssertionError(f"{path.name}: schema_version must be 1")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="loom-artifacts-") as tmp:
        out_dir = Path(tmp)
        produced: list[Path] = []

        for script, filename, required_columns in CSV_COMMANDS:
            output = out_dir / filename
            result = run_command(repo, ["bash", script, "--output", str(output)])
            if result.returncode != 0:
                raise AssertionError(
                    f"{script} failed with {result.returncode}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            assert_csv_artifact(
                output,
                required_columns,
                allow_pass_rows=filename
                in {"source-compat-summary.csv", "compiler-pipeline-summary.csv"},
            )
            produced.append(output)

        for script, filename, required_keys in JSON_COMMANDS:
            output = out_dir / filename
            result = run_command(repo, ["bash", script, "--output", str(output)])
            if result.returncode != 0:
                raise AssertionError(
                    f"{script} failed with {result.returncode}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            assert_json_artifact(output, required_keys)
            produced.append(output)

        audit_pass = out_dir / "artifact-audit-summary.json"
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit_pass),
                *[str(path) for path in produced],
            ],
        )
        if result.returncode != 0:
            raise AssertionError(
                f"audit failed with {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        audit_data = json.loads(audit_pass.read_text())
        if audit_data.get("verdict") != "pass":
            raise AssertionError(f"expected pass audit, got {audit_data}")

        invalid = out_dir / "invalid-sim-cycle-summary.csv"
        invalid.write_text(
            "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n"
            "bad,0,0,pass,\n"
        )
        audit_fail = out_dir / "artifact-audit-summary-fail.json"
        result = run_command(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit_fail),
                str(invalid),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("invalid artifact audit unexpectedly exited zero")
        audit_data = json.loads(audit_fail.read_text())
        if audit_data.get("verdict") != "fail":
            raise AssertionError(f"expected fail audit, got {audit_data}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
