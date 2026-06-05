#!/usr/bin/env python3
"""Shared helpers for artifact regression tests."""

from __future__ import annotations

import csv
import subprocess
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


def read_csv_rows(path: Path, expected_header: list[str]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames[: len(expected_header)] != expected_header:
            raise AssertionError(f"unexpected header: {reader.fieldnames}")
        return rows


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
