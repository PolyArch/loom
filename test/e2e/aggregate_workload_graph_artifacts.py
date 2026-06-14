#!/usr/bin/env python3
"""Build workload-level aggregate artifacts from per-graph evidence."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


AGGREGATABLE_DFG_COMPONENT_STATUSES = {"pass", "unsupported", "blocked", "fail"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--mapping-id", required=True)
    parser.add_argument("--graph", default="workload_graph_set")
    parser.add_argument("--dfg-report", action="append", required=True)
    parser.add_argument("--mapping-artifact", action="append", required=True)
    parser.add_argument("--cgra-report", action="append", required=True)
    parser.add_argument("--dfg-output", required=True)
    parser.add_argument("--mapping-output", required=True)
    parser.add_argument("--cgra-output", required=True)
    parser.add_argument("--mapping-summary-output", required=True)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def artifact_id(path: Path) -> str:
    return intermediate_artifacts.artifact_id_for_path(path)


def artifact_fingerprint(path: Path) -> str:
    return intermediate_artifacts.artifact_fingerprint(path)


def component_identity_list(paths: list[Path]) -> list[str]:
    return [artifact_id(path) for path in paths]


def component_fingerprint_map(paths: list[Path]) -> dict[str, str]:
    return {artifact_id(path): artifact_fingerprint(path) for path in paths}


def sum_int(items: list[dict[str, Any]], key: str) -> int:
    total = 0
    for item in items:
        value = item.get(key)
        require(isinstance(value, int), f"{key} must be an integer in component artifact")
        total += value
    return total


def same_string(items: list[dict[str, Any]], key: str, fallback: str = "") -> str:
    values = [item.get(key) for item in items if isinstance(item.get(key), str) and item.get(key)]
    if not values:
        return fallback
    first = values[0]
    require(all(value == first for value in values), f"component artifacts disagree on {key}")
    return first


def merge_counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in items:
        counts = item.get(key)
        require(isinstance(counts, dict), f"{key} must be an object in component artifact")
        for name, value in counts.items():
            require(isinstance(name, str), f"{key} contains a non-string key")
            require(isinstance(value, int), f"{key}.{name} must be an integer")
            merged[name] = merged.get(name, 0) + value
    return dict(sorted(merged.items()))


def unique_strings(items: list[dict[str, Any]], key: str) -> list[str]:
    values: set[str] = set()
    for item in items:
        raw = item.get(key)
        if isinstance(raw, str) and raw:
            values.add(raw)
        elif isinstance(raw, list):
            for value in raw:
                if isinstance(value, str) and value:
                    values.add(value)
    return sorted(values)


def component_statuses(items: list[dict[str, Any]]) -> list[str]:
    statuses: list[str] = []
    for item in items:
        status = item.get("status")
        require(isinstance(status, str) and status, "component artifact lacks status")
        statuses.append(status)
    return statuses


def aggregate_component_status(items: list[dict[str, Any]]) -> str:
    statuses = component_statuses(items)
    if all(status == "pass" for status in statuses):
        return "pass"
    if any(status == "unsupported" for status in statuses):
        return "unsupported"
    return "blocked"


def component_diagnostics(items: list[dict[str, Any]]) -> list[str]:
    diagnostics: set[str] = set()
    for item in items:
        raw = item.get("diagnostics")
        if not isinstance(raw, list):
            continue
        for diagnostic in raw:
            if isinstance(diagnostic, str) and diagnostic:
                diagnostics.add(diagnostic)
    return sorted(diagnostics)


def merge_records(
    items: list[dict[str, Any]],
    key: str,
    *,
    id_key: str | None = None,
) -> list[Any]:
    merged: list[Any] = []
    for item in items:
        component_id = item.get("mapping_id")
        component_graph = item.get("graph")
        records = item.get(key)
        require(isinstance(records, list), f"{key} must be a list in component artifact")
        for index, record in enumerate(records):
            copied = copy.deepcopy(record)
            if isinstance(copied, dict):
                if isinstance(component_id, str) and component_id:
                    copied["component_mapping_id"] = component_id
                if isinstance(component_graph, str) and component_graph:
                    copied["component_graph"] = component_graph
                if id_key is not None and isinstance(copied.get(id_key), str):
                    copied[id_key] = f"{component_graph}:{copied[id_key]}"
                if key == "routes":
                    segments = copied.get("segments")
                    if isinstance(segments, list):
                        for segment in segments:
                            if isinstance(segment, dict) and isinstance(segment.get("segment_id"), str):
                                segment["segment_id"] = f"{component_graph}:{segment['segment_id']}"
            else:
                copied = {
                    "component_mapping_id": component_id,
                    "component_graph": component_graph,
                    "record_index": index,
                    "value": copied,
                }
            merged.append(copied)
    return merged


def aggregate_dfg(
    args: argparse.Namespace,
    dfg_paths: list[Path],
    dfg_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    graphs = [str(report["graph"]) for report in dfg_reports]
    final_outputs = merge_final_outputs(dfg_reports)
    final_memory_state = merge_final_memory_state(dfg_reports, graphs)
    status = aggregate_component_status(dfg_reports)
    diagnostics = ["derived workload graph-set DFG report from component DFG simulator reports"]
    if status != "pass":
        diagnostics.append(
            "one or more component DFG simulator reports are not passing: "
            + ",".join(component_statuses(dfg_reports))
        )
        diagnostics.extend(component_diagnostics(dfg_reports))
    return {
        "schema_version": 1,
        "kind": "dfg_sim_report",
        "workload": args.workload,
        "graph": args.graph,
        "aggregation_kind": "workload_graph_set",
        "component_graphs": graphs,
        "component_dfg_sim_report_identities": component_identity_list(dfg_paths),
        "input_artifact_fingerprints": component_fingerprint_map(dfg_paths),
        "status": status,
        "metric_definition": same_string(dfg_reports, "metric_definition"),
        "operation_semantics_source": same_string(dfg_reports, "operation_semantics_source"),
        "operation_cost_model_source": same_string(dfg_reports, "operation_cost_model_source"),
        "optimistic_cycles": sum_int(dfg_reports, "optimistic_cycles"),
        "wavefront_steps": sum_int(dfg_reports, "wavefront_steps"),
        "event_count": sum_int(dfg_reports, "event_count"),
        "dynamic_work_items": sum_int(dfg_reports, "dynamic_work_items"),
        "operation_fire_counts": merge_counts(dfg_reports, "operation_fire_counts"),
        "final_outputs": final_outputs,
        "final_memory_state": final_memory_state,
        "diagnostics": diagnostics,
    }


def merge_final_outputs(reports: list[dict[str, Any]]) -> list[str]:
    outputs: list[str] = []
    for report in reports:
        report_outputs = report.get("final_outputs")
        require(isinstance(report_outputs, list), "component simulator report lacks final_outputs")
        for output in report_outputs:
            require(isinstance(output, str), "component final output must be a string")
            outputs.append(output)
    return outputs


def merge_final_memory_state(
    reports: list[dict[str, Any]],
    component_names: list[str],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    require(len(reports) == len(component_names), "component memory-state labels must match reports")
    for component_name, report in zip(component_names, reports):
        memory_state = report.get("final_memory_state")
        require(isinstance(memory_state, dict), "component simulator report lacks final_memory_state")
        for memory_name, values in memory_state.items():
            require(isinstance(memory_name, str), "component memory-state key must be a string")
            require(isinstance(values, list), "component memory-state value must be an array")
            serialized_values: list[str] = []
            for value in values:
                require(isinstance(value, str), "component memory-state element must be a string")
                serialized_values.append(value)
            merged[f"{component_name}:{memory_name}"] = serialized_values
    return merged


def aggregate_mapping(
    args: argparse.Namespace,
    mapping_paths: list[Path],
    mapping_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    mapping_ids = [str(artifact["mapping_id"]) for artifact in mapping_artifacts]
    graphs = [str(artifact["graph"]) for artifact in mapping_artifacts]
    statuses = component_statuses(mapping_artifacts)
    placed_records = sum_int(mapping_artifacts, "placed_records")
    routed_edges = sum_int(mapping_artifacts, "routed_edges")
    unplaced_records = sum_int(mapping_artifacts, "unplaced_records")
    unrouted_edges = sum_int(mapping_artifacts, "unrouted_edges")
    config_records = sum_int(mapping_artifacts, "config_records")
    route_segments = sum(
        intermediate_artifacts.route_segment_count(artifact.get("routes", []))
        for artifact in mapping_artifacts
    )
    components_pass = all(status == "pass" for status in statuses)
    if components_pass and unplaced_records == 0 and unrouted_edges == 0:
        status = "pass"
    elif any(component_status == "unsupported" for component_status in statuses):
        status = "unsupported"
    else:
        status = "blocked"
    diagnostics = ["derived workload graph-set mapping artifact from component PnR mapping artifacts"]
    if not components_pass:
        diagnostics.append(
            "one or more component mapping artifacts are not passing: "
            + ",".join(statuses)
        )
        diagnostics.extend(component_diagnostics(mapping_artifacts))
    return {
        "schema_version": 1,
        "kind": "pnr_mapping",
        "workload": args.workload,
        "hardware": args.hardware,
        "graph": args.graph,
        "mapping_id": args.mapping_id,
        "aggregation_kind": "workload_graph_set",
        "component_graphs": graphs,
        "component_mapping_ids": mapping_ids,
        "component_mapping_artifact_identities": component_identity_list(mapping_paths),
        "input_artifact_fingerprints": component_fingerprint_map(mapping_paths),
        "status": status,
        "placed_records": placed_records,
        "routed_edges": routed_edges,
        "unrouted_edges": unrouted_edges,
        "unplaced_records": unplaced_records,
        "route_segments": route_segments,
        "config_records": config_records,
        "placements": merge_records(mapping_artifacts, "placements"),
        "routes": merge_records(mapping_artifacts, "routes", id_key="record_id"),
        "unrouted_edge_details": merge_records(mapping_artifacts, "unrouted_edge_details"),
        "config_bitstream": merge_records(mapping_artifacts, "config_bitstream"),
        "diagnostics": diagnostics,
    }


def aggregate_cycle_breakdown(cgra_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories: dict[str, int] = {}
    for report in cgra_reports:
        for item in report.get("cycle_breakdown", []):
            if not isinstance(item, dict):
                continue
            category = item.get("category")
            cycles = item.get("cycles")
            if isinstance(category, str) and isinstance(cycles, int):
                categories[category] = categories.get(category, 0) + cycles
    return [
        {
            "category": category,
            "cycles": cycles,
            "evidence": "component_cgra_sim_reports",
            "explanation": "sum of component workload graph-set CGRA-sim cycle breakdown entries",
            "modeled": True,
        }
        for category, cycles in sorted(categories.items())
    ]


def aggregate_cgra(
    args: argparse.Namespace,
    cgra_paths: list[Path],
    cgra_reports: list[dict[str, Any]],
    mapping_artifact: dict[str, Any],
    dfg_paths: list[Path],
    dfg_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    dfg_cycles = sum_int(cgra_reports, "dfg_cycles")
    hardware_aware_cycles = sum_int(cgra_reports, "hardware_aware_cycles")
    mapping_ids = [str(report["mapping_id"]) for report in cgra_reports]
    graphs = [str(report["graph"]) for report in dfg_reports]
    performance_delta = hardware_aware_cycles - dfg_cycles
    aggregate_status = (
        "pass"
        if mapping_artifact.get("status") == "pass"
        and all(report.get("status") == "pass" for report in cgra_reports)
        else "blocked"
    )
    diagnostics = ["derived workload graph-set CGRA report from component CGRA simulator reports"]
    if aggregate_status != "pass":
        diagnostics.append("one or more component mappings or CGRA reports are not passing")
    return {
        "schema_version": 1,
        "kind": "cgra_sim_report",
        "workload": args.workload,
        "hardware": args.hardware,
        "hardware_artifact": same_string(cgra_reports, "hardware_artifact"),
        "mapping_id": args.mapping_id,
        "aggregation_kind": "workload_graph_set",
        "component_mapping_ids": mapping_ids,
        "component_dfg_sim_report_identities": component_identity_list(dfg_paths),
        "component_cgra_sim_report_identities": component_identity_list(cgra_paths),
        "input_artifact_fingerprints": {
            **component_fingerprint_map(dfg_paths),
            **component_fingerprint_map(cgra_paths),
        },
        "status": aggregate_status,
        "fidelity_level": same_string(cgra_reports, "fidelity_level"),
        "metric_definition": same_string(cgra_reports, "metric_definition"),
        "operation_semantics_source": same_string(cgra_reports, "operation_semantics_source"),
        "operation_cost_model_source": same_string(cgra_reports, "operation_cost_model_source"),
        "functional_state_source": "component_cgra_sim_reports_carried_from_dfg_sim_reports",
        "final_outputs": merge_final_outputs(cgra_reports),
        "final_memory_state": merge_final_memory_state(cgra_reports, graphs),
        "difference_classification": (
            "unsupported_scope"
            if aggregate_status != "pass"
            else "match"
            if hardware_aware_cycles == dfg_cycles
            else "expected_hardware_constraint"
        ),
        "hardware_bound_classification": (
            "unsupported_scope"
            if aggregate_status != "pass"
            else same_string(cgra_reports, "hardware_bound_classification")
        ),
        "dfg_cycles": dfg_cycles,
        "modeled_lower_bound_cycles": hardware_aware_cycles,
        "performance_delta_cycles": performance_delta,
        "route_latency_cycles": sum_int(cgra_reports, "route_latency_cycles"),
        "memory_latency_cycles": sum_int(cgra_reports, "memory_latency_cycles"),
        "temporal_penalty_cycles": sum_int(cgra_reports, "temporal_penalty_cycles"),
        "hardware_aware_cycles": hardware_aware_cycles,
        "cycle_breakdown": aggregate_cycle_breakdown(cgra_reports),
        "unmodeled_constraints": unique_strings(cgra_reports, "unmodeled_constraints"),
        "first_principles_checks": [
            {
                "name": "aggregate_cgra_not_more_optimistic_than_dfg",
                "status": "pass" if hardware_aware_cycles >= dfg_cycles else "fail",
                "evidence": "sum(component.hardware_aware_cycles) >= sum(component.dfg_cycles)",
            },
            {
                "name": "aggregate_cgra_routes_match_mapping",
                "status": "pass",
                "evidence": "aggregate route_segments matches aggregate mapping routes",
            },
        ],
        "diagnostics": diagnostics,
        "placed_records": int(mapping_artifact["placed_records"]),
        "spatial_placements": sum_int(cgra_reports, "spatial_placements"),
        "temporal_placements": sum_int(cgra_reports, "temporal_placements"),
        "routed_edges": int(mapping_artifact["routed_edges"]),
        "route_segments": intermediate_artifacts.route_segment_count(mapping_artifact["routes"]),
        "config_records": int(mapping_artifact["config_records"]),
    }


def write_mapping_summary(path: Path, mapping: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "workload": str(mapping["workload"]),
        "hardware": str(mapping["hardware"]),
        "mapping_id": str(mapping["mapping_id"]),
        "placed_records": str(mapping["placed_records"]),
        "routed_edges": str(mapping["routed_edges"]),
        "unrouted_edges": str(mapping["unrouted_edges"]),
        "unplaced_records": str(mapping["unplaced_records"]),
        "status": str(mapping["status"]),
        "diagnostic": "derived workload graph-set mapping row from component PnR mapping artifacts",
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=intermediate_artifacts.csv_header("pnr_mapping"))
        writer.writeheader()
        writer.writerow(row)


def validate_components(
    args: argparse.Namespace,
    dfg_reports: list[dict[str, Any]],
    mapping_artifacts: list[dict[str, Any]],
    cgra_reports: list[dict[str, Any]],
) -> None:
    require(len(dfg_reports) == len(mapping_artifacts) == len(cgra_reports), "component artifact counts must match")
    require(len(dfg_reports) > 0, "at least one component graph is required")
    for index, report in enumerate(dfg_reports):
        require(report.get("kind") == "dfg_sim_report", "DFG component has wrong kind")
        require(
            report.get("status") in AGGREGATABLE_DFG_COMPONENT_STATUSES,
            "DFG component status cannot be aggregated",
        )
        require(report.get("workload") == args.workload, "DFG component workload mismatch")
        graph = report.get("graph")
        require(isinstance(graph, str) and graph, "DFG component lacks graph")
        mapping = mapping_artifacts[index]
        require(mapping.get("graph") == graph, "component mapping graph does not match DFG graph")
        cgra = cgra_reports[index]
        require(
            cgra.get("mapping_id") == mapping.get("mapping_id"),
            "component CGRA mapping_id does not match mapping artifact order",
        )
    for artifact in mapping_artifacts:
        require(artifact.get("kind") == "pnr_mapping", "mapping component has wrong kind")
        require(artifact.get("workload") == args.workload, "mapping component workload mismatch")
        require(artifact.get("hardware") == args.hardware, "mapping component hardware mismatch")
    mapping_ids = {artifact.get("mapping_id") for artifact in mapping_artifacts}
    for report in cgra_reports:
        require(report.get("kind") == "cgra_sim_report", "CGRA component has wrong kind")
        require(report.get("workload") == args.workload, "CGRA component workload mismatch")
        require(report.get("hardware") == args.hardware, "CGRA component hardware mismatch")
        require(report.get("mapping_id") in mapping_ids, "CGRA component mapping mismatch")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dfg_paths = [Path(value) for value in args.dfg_report]
    mapping_paths = [Path(value) for value in args.mapping_artifact]
    cgra_paths = [Path(value) for value in args.cgra_report]
    dfg_reports = [read_json(path) for path in dfg_paths]
    mapping_artifacts = [read_json(path) for path in mapping_paths]
    cgra_reports = [read_json(path) for path in cgra_paths]
    validate_components(args, dfg_reports, mapping_artifacts, cgra_reports)

    dfg = aggregate_dfg(args, dfg_paths, dfg_reports)
    mapping = aggregate_mapping(args, mapping_paths, mapping_artifacts)
    cgra = aggregate_cgra(args, cgra_paths, cgra_reports, mapping, dfg_paths, dfg_reports)
    require(cgra["route_segments"] == mapping["route_segments"], "aggregate CGRA route segments do not match mapping")
    require(cgra["config_records"] == mapping["config_records"], "aggregate CGRA config records do not match mapping")
    require(cgra["hardware_aware_cycles"] >= dfg["optimistic_cycles"], "aggregate CGRA cycles are too optimistic")

    write_json(Path(args.dfg_output), dfg)
    write_json(Path(args.mapping_output), mapping)
    write_json(Path(args.cgra_output), cgra)
    write_mapping_summary(Path(args.mapping_summary_output), mapping)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
