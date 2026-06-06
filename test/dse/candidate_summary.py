#!/usr/bin/env python3
"""Emit DSE candidate summary rows from mapping, sim, and FPA artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--objective", default="minimize_runtime")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def artifacts_by_kind(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(intermediate_artifacts.artifact_kind_for_path(path), []).append(path)
    return grouped


def mapping_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(
            row
            for row in read_csv(path)
            if row.get("workload") not in {"", "scaffold", None}
            and row.get("hardware") not in {"", "scaffold", None}
        )
    return rows


def sim_for_workload(paths: list[Path], workload: str) -> dict[str, str]:
    for path in paths:
        for row in read_csv(path):
            if row.get("kernel") == workload:
                return row
    return {}


def fpa_for_candidate(paths: list[Path], workload: str, hardware: str) -> dict[str, str]:
    suffix_matches: list[dict[str, str]] = []
    for path in paths:
        for row in read_csv(path):
            if row.get("workload") != workload:
                continue
            row_hardware = row.get("hardware", "")
            if row_hardware == hardware:
                return row
            if row_hardware.rsplit("::", 1)[-1] == hardware:
                suffix_matches.append(row)
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return {}


def parse_float(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "")
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def parse_positive_float(row: dict[str, str], column: str) -> float | None:
    parsed = parse_float(row, column)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def complete_evidence(
    mapping: dict[str, str],
    sim: dict[str, str],
    fpa: dict[str, str],
) -> tuple[float, float] | None:
    if mapping.get("status") != "pass" or not mapping.get("mapping_id"):
        return None
    if sim.get("status") != "pass":
        return None
    if fpa.get("status") != "pass":
        return None

    cycles = parse_positive_float(sim, "cgra_sim_cycles")
    frequency_mhz = parse_positive_float(fpa, "frequency_mhz")
    dynamic_power_mw = parse_float(fpa, "dynamic_power_mw")
    leakage_power_mw = parse_float(fpa, "leakage_power_mw")
    area_um2 = parse_positive_float(fpa, "area_um2")
    if (
        cycles is None
        or frequency_mhz is None
        or dynamic_power_mw is None
        or leakage_power_mw is None
        or area_um2 is None
    ):
        return None
    total_power_mw = dynamic_power_mw + leakage_power_mw
    energy_nj = total_power_mw * cycles / frequency_mhz
    return cycles, energy_nj


def candidate_row(
    mapping: dict[str, str],
    sim: dict[str, str],
    fpa: dict[str, str],
    objective: str,
) -> dict[str, str]:
    workload = mapping["workload"]
    hardware = mapping["hardware"]
    complete = complete_evidence(mapping, sim, fpa)
    if complete is not None:
        _cycles, energy_nj = complete
        return {
            "candidate": f"candidate::{workload}::{hardware}",
            "workload": workload,
            "hardware": hardware,
            "mapping_id": mapping["mapping_id"],
            "objective": objective,
            "cgra_sim_cycles": sim["cgra_sim_cycles"],
            "frequency_mhz": fpa["frequency_mhz"],
            "area_um2": fpa["area_um2"],
            "dynamic_power_mw": fpa["dynamic_power_mw"],
            "energy_nj": f"{energy_nj:.3f}",
            "selection_status": "selected",
            "diagnostic": (
                "cycle-frequency-power-area energy estimate; "
                "energy_nj=(dynamic_power_mw+leakage_power_mw)*"
                "cgra_sim_cycles/frequency_mhz"
            ),
        }
    return {
        "candidate": f"candidate::{workload}::{hardware}",
        "workload": workload,
        "hardware": hardware,
        "mapping_id": "",
        "objective": objective,
        "cgra_sim_cycles": "",
        "frequency_mhz": "",
        "area_um2": "",
        "dynamic_power_mw": "",
        "energy_nj": "",
        "selection_status": "blocked",
        "diagnostic": "missing mapping, simulator, or FPA evidence for DSE selection",
    }


def runtime_score(row: dict[str, str]) -> float:
    cycles = parse_positive_float(row, "cgra_sim_cycles")
    frequency_mhz = parse_positive_float(row, "frequency_mhz")
    if cycles is None or frequency_mhz is None:
        return float("inf")
    return cycles / frequency_mhz


def energy_score(row: dict[str, str]) -> float:
    energy = parse_positive_float(row, "energy_nj")
    return energy if energy is not None else float("inf")


def select_candidates(rows: list[dict[str, str]], objective: str) -> None:
    complete = [row for row in rows if row.get("selection_status") == "selected"]
    if len(complete) <= 1:
        return
    score: Callable[[dict[str, str]], float]
    if objective in {"minimize_energy", "minimize_power"}:
        score = energy_score
    else:
        score = runtime_score
    selected = min(complete, key=lambda row: (score(row), row["candidate"]))
    for row in complete:
        if row is selected:
            continue
        row["selection_status"] = "rejected"
        row["diagnostic"] = (
            "complete cycle-frequency-power-area evidence; rejected by "
            f"{objective} deterministic ordering"
        )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    paths = intermediate_artifacts.discover_artifact_paths(
        ROOT,
        args.artifact,
        include_unsupported_scope=False,
    )
    grouped = artifacts_by_kind(paths)
    rows = [
        candidate_row(
            row,
            sim_for_workload(grouped.get("sim_cycle", []), row["workload"]),
            fpa_for_candidate(grouped.get("rtl_fpa", []), row["workload"], row["hardware"]),
            args.objective,
        )
        for row in mapping_rows(grouped.get("pnr_mapping", []))
    ]
    select_candidates(rows, args.objective)

    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        intermediate_artifacts.write_csv_rows("dse_candidate", output, rows)
    else:
        intermediate_artifacts.write_csv("dse_candidate", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
