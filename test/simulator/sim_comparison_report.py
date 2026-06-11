#!/usr/bin/env python3
"""Emit DFG/CGRA simulation comparison reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfg-report", required=True)
    parser.add_argument("--cgra-report", required=True)
    parser.add_argument("--mapping-artifact")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def string_field(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    return value if isinstance(value, str) else ""


def int_field(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if isinstance(value, int) and value >= 0:
        return value
    return None


def list_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def cycle_breakdown_categories(cgra: dict[str, object]) -> list[str]:
    breakdown = cgra.get("cycle_breakdown")
    if not isinstance(breakdown, list):
        return []
    categories: list[str] = []
    for item in breakdown:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        if isinstance(category, str) and category:
            categories.append(category)
    return categories


def compare_final_outputs(dfg: dict[str, object], cgra: dict[str, object]) -> tuple[str, list[str]]:
    dfg_outputs = dfg.get("final_outputs")
    cgra_outputs = cgra.get("final_outputs")
    if not isinstance(dfg_outputs, list) or not isinstance(cgra_outputs, list):
        return "blocked", ["functional output comparison blocked because one report lacks final_outputs"]
    if [str(item) for item in dfg_outputs] == [str(item) for item in cgra_outputs]:
        return "pass", []
    return "fail", ["functional output mismatch between DFG-sim and CGRA-sim reports"]


def compare_memory_state(dfg: dict[str, object], cgra: dict[str, object]) -> tuple[str, list[str]]:
    dfg_memory = dfg.get("final_memory_state")
    cgra_memory = cgra.get("final_memory_state")
    if not isinstance(dfg_memory, dict) or not isinstance(cgra_memory, dict):
        return "blocked", ["visible memory-state comparison blocked because one report lacks final_memory_state"]
    if dfg_memory == cgra_memory:
        return "pass", []
    return "fail", ["visible memory-state mismatch between DFG-sim and CGRA-sim reports"]


def build_report(
    dfg_path: Path,
    cgra_path: Path,
    mapping_path: Path | None,
) -> dict[str, object]:
    dfg = read_json(dfg_path)
    cgra = read_json(cgra_path)
    mapping = read_json(mapping_path) if mapping_path is not None else {}

    dfg_workload = string_field(dfg, "workload")
    cgra_workload = string_field(cgra, "workload")
    workload = dfg_workload or cgra_workload or "unknown"
    diagnostics: list[str] = []
    difference_classification = "match"
    status = "pass"
    performance_status = "pass"
    input_status_blocked = False
    input_status_classification = "unsupported_scope"

    if string_field(dfg, "kind") != "dfg_sim_report":
        diagnostics.append("DFG input is not a dfg_sim_report")
        difference_classification = "report_mismatch"
    if string_field(cgra, "kind") != "cgra_sim_report":
        diagnostics.append("CGRA input is not a cgra_sim_report")
        difference_classification = "report_mismatch"
    if dfg_workload != cgra_workload:
        diagnostics.append(
            f"workload identity mismatch: DFG-sim={dfg_workload!r}, CGRA-sim={cgra_workload!r}"
        )
        difference_classification = "report_mismatch"
    if mapping:
        mapping_workload = string_field(mapping, "workload")
        mapping_id = string_field(mapping, "mapping_id")
        cgra_mapping_id = string_field(cgra, "mapping_id")
        if mapping_workload and mapping_workload != cgra_workload:
            diagnostics.append(
                f"mapping workload identity mismatch: mapping={mapping_workload!r}, CGRA-sim={cgra_workload!r}"
            )
            difference_classification = "report_mismatch"
        if mapping_id and cgra_mapping_id and mapping_id != cgra_mapping_id:
            diagnostics.append(
                f"mapping artifact identity mismatch: mapping={mapping_id!r}, CGRA-sim={cgra_mapping_id!r}"
            )
            difference_classification = "mapping_invalid"

    dfg_status = string_field(dfg, "status")
    cgra_status = string_field(cgra, "status")
    if dfg_status and dfg_status != "pass":
        diagnostics.append(f"DFG-sim report status {dfg_status} blocks simulation comparison")
        input_status_blocked = True
    if cgra_status and cgra_status != "pass":
        diagnostics.append(f"CGRA-sim report status {cgra_status} blocks performance comparison")
        input_status_blocked = True
        input_status_classification = string_field(cgra, "difference_classification") or "unsupported_scope"

    dfg_cycles = int_field(dfg, "optimistic_cycles")
    cgra_cycles = int_field(cgra, "hardware_aware_cycles")
    performance_delta = int_field(cgra, "performance_delta_cycles")
    if input_status_blocked:
        performance_status = "blocked"
        if difference_classification == "match":
            difference_classification = input_status_classification
    elif dfg_cycles is None or cgra_cycles is None:
        diagnostics.append("missing comparable simulator cycle metric")
        performance_status = "blocked"
        if difference_classification == "match":
            difference_classification = "metric_not_comparable"
    elif cgra_cycles < dfg_cycles:
        diagnostics.append("CGRA-sim cycles are more optimistic than DFG-sim cycles")
        performance_status = "fail"
        if difference_classification == "match":
            difference_classification = "metric_not_comparable"
    else:
        cgra_classification = string_field(cgra, "difference_classification")
        if difference_classification != "match":
            pass
        elif cgra_cycles == dfg_cycles:
            difference_classification = "match"
        elif cgra_classification:
            difference_classification = cgra_classification
        else:
            difference_classification = "expected_hardware_constraint"

    identity_or_mapping_failure = difference_classification in {"report_mismatch", "mapping_invalid"}
    if identity_or_mapping_failure:
        status = "fail"
        performance_status = "blocked"
    elif input_status_blocked:
        status = "blocked"
    elif performance_status == "fail":
        status = "fail"
    elif performance_status == "blocked":
        status = "blocked"

    functional_status, functional_diagnostics = compare_final_outputs(dfg, cgra)
    memory_status, memory_diagnostics = compare_memory_state(dfg, cgra)
    diagnostics.extend(functional_diagnostics)
    diagnostics.extend(memory_diagnostics)
    if functional_status == "fail" and not identity_or_mapping_failure:
        status = "fail"
        difference_classification = "functional_mismatch"
    if functional_status == "blocked" and status == "pass":
        status = "blocked"
        if not identity_or_mapping_failure:
            difference_classification = "unsupported_scope"
    if memory_status == "fail" and not identity_or_mapping_failure:
        status = "fail"
        difference_classification = "functional_mismatch"
    if memory_status == "blocked" and status == "pass":
        status = "blocked"
        if not identity_or_mapping_failure:
            difference_classification = "unsupported_scope"

    explanation_categories = cycle_breakdown_categories(cgra)
    explanation_categories.extend(list_strings(cgra.get("unmodeled_constraints")))
    explanation_categories = sorted(set(explanation_categories))
    runtime_input_identity = string_field(dfg, "runtime_input_identity")
    if not runtime_input_identity:
        runtime_input_identity = string_field(cgra, "runtime_input_identity")
    if not runtime_input_identity:
        runtime_input_identity = f"test-app-fixture::{workload}::default"

    return {
        "schema_version": 1,
        "kind": "sim_comparison_report",
        "comparison_id": f"sim-comparison::{workload}::{intermediate_artifacts.artifact_id_for_path(cgra_path)}",
        "workload": workload,
        "runtime_input_identity": runtime_input_identity,
        "dfg_sim_report_identity": intermediate_artifacts.artifact_id_for_path(dfg_path),
        "cgra_sim_report_identity": intermediate_artifacts.artifact_id_for_path(cgra_path),
        "mapping_artifact_identity": intermediate_artifacts.artifact_id_for_path(mapping_path),
        "functional_comparison_status": functional_status,
        "memory_comparison_status": memory_status,
        "performance_comparison_status": performance_status,
        "performance_metric_definitions": {
            "dfg": string_field(dfg, "metric_definition"),
            "cgra": string_field(cgra, "metric_definition"),
        },
        "dfg_sim_cycles": dfg_cycles,
        "cgra_sim_cycles": cgra_cycles,
        "performance_delta_cycles": performance_delta,
        "difference_classification": difference_classification,
        "explanation_categories": explanation_categories,
        "diagnostics": diagnostics,
        "status": status,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dfg_path = Path(args.dfg_report)
    cgra_path = Path(args.cgra_report)
    mapping_path = Path(args.mapping_artifact) if args.mapping_artifact else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(dfg_path, cgra_path, mapping_path)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
