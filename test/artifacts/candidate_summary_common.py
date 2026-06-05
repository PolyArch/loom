#!/usr/bin/env python3
"""Shared candidate discovery for downstream artifact summaries."""

from __future__ import annotations

import csv
from pathlib import Path


IGNORED_WORKLOADS = {"", "scaffold", "none", None}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def workloads_from_primitive_coverage(path: Path) -> list[str]:
    workloads = {
        row["workload"]
        for row in read_rows(path)
        if row.get("workload") not in IGNORED_WORKLOADS
    }
    return sorted(workloads)


def hardware_from_summary(path: Path) -> list[str]:
    hardware = {
        row["hardware"]
        for row in read_rows(path)
        if row.get("verify_status") == "pass" and row.get("hardware")
    }
    return sorted(hardware)


def has_candidate_inputs(primitive_path: Path, hardware_path: Path) -> bool:
    return bool(workloads_from_primitive_coverage(primitive_path)) and bool(hardware_from_summary(hardware_path))
