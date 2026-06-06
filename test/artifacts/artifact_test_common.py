#!/usr/bin/env python3
"""Shared helpers for artifact regression tests."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import tempfile
from pathlib import Path


def run_command(repo: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require_success(repo: Path, argv: list[str], label: str) -> subprocess.CompletedProcess[str]:
    result = run_command(repo, argv)
    if result.returncode != 0:
        raise AssertionError(
            f"{label} failed with {result.returncode}\n"
            f"command: {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def repo_temp_dir(repo: Path, prefix: str) -> tempfile.TemporaryDirectory[str]:
    temp_root = repo / "temp" / "test-runs"
    temp_root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=temp_root)


def read_csv_rows(path: Path, expected_header: list[str]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames[: len(expected_header)] != expected_header:
            raise AssertionError(f"unexpected header: {reader.fieldnames}")
        return rows


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semicolon_map(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in raw.split(";"):
        if not entry:
            continue
        key, value = entry.rsplit("=", 1)
        parsed[key] = value
    return parsed


def run_csv_summary(
    repo: Path,
    script: str,
    output: Path,
    expected_header: list[str],
    *args: str,
    label: str,
) -> list[dict[str, str]]:
    require_success(
        repo,
        [
            "bash",
            script,
            *args,
            "--output",
            str(output),
        ],
        label,
    )
    return read_csv_rows(output, expected_header)


def prepare_candidate_inputs(repo: Path, out_dir: Path) -> tuple[Path, Path]:
    primitive = out_dir / "dataflow-primitive-coverage.csv"
    hardware = out_dir / "adg-hardware-summary.csv"
    require_success(
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
    require_success(
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
    return primitive, hardware
