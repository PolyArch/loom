#!/usr/bin/env python3
"""Emit structured unsupported evidence when a complete primary graph is unavailable."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))
sys.path.insert(0, str(ROOT / "test" / "pnr"))

import intermediate_artifacts  # noqa: E402
import mapping_summary  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--dfg-mlir", required=True)
    parser.add_argument("--expected-graph-token", required=True)
    parser.add_argument("--required-discovered-graph", action="append", default=[])
    parser.add_argument("--required-residual-call", action="append", default=[])
    parser.add_argument(
        "--expected-graph-presence",
        choices=("absent", "present"),
        default="absent",
    )
    parser.add_argument("--diagnostic")
    parser.add_argument("--evidence", default="primary workload graph unavailable")
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--graph", default="missing_primary_graph")
    parser.add_argument("--dfg-output", required=True)
    parser.add_argument("--dfg-cycle-output", required=True)
    parser.add_argument("--mapping-output", required=True)
    parser.add_argument("--mapping-summary-output", required=True)
    return parser.parse_args(argv)


def ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def graph_ids_from_text(text: str) -> list[str]:
    definitions = re.findall(
        r"\bdataflow\.graph\.func\s+(?:private\s+)?@([A-Za-z_.$][\w.$-]*)",
        text,
    )
    launches = re.findall(r"\bdataflow\.graph\.launch\s+@([A-Za-z_.$][\w.$-]*)", text)
    return ordered_unique([*definitions, *launches])


def residual_call_targets_from_text(text: str) -> list[str]:
    calls = re.findall(r"\b(?:func\.call|call)\s+@([A-Za-z_.$][\w.$-]*)", text)
    return ordered_unique(calls)


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def build_diagnostic(expected_token: str, explicit_diagnostic: str | None) -> str:
    if explicit_diagnostic:
        return explicit_diagnostic
    return f"primary workload graph absent: expected token {expected_token}"


def emit_artifacts(args: argparse.Namespace) -> None:
    dfg_mlir = Path(args.dfg_mlir)
    text = dfg_mlir.read_text(errors="replace")
    graph_ids = graph_ids_from_text(text)
    residual_calls = residual_call_targets_from_text(text)
    token_present = any(args.expected_graph_token in graph_id for graph_id in graph_ids)
    if args.expected_graph_presence == "absent" and token_present:
        raise SystemExit(
            "primary workload graph is present; use the real DFG-sim and PnR chain instead: "
            f"{args.expected_graph_token}"
        )
    if args.expected_graph_presence == "present" and not token_present:
        raise SystemExit(
            "expected workload graph is absent; use the primary-graph absence path instead: "
            f"{args.expected_graph_token}"
        )
    missing_required_graphs = [
        graph for graph in args.required_discovered_graph if graph not in graph_ids
    ]
    if missing_required_graphs:
        raise SystemExit(
            "required discovered graph is absent: "
            + ",".join(missing_required_graphs)
        )
    missing_required_calls = [
        call for call in args.required_residual_call if call not in residual_calls
    ]
    if missing_required_calls:
        raise SystemExit(
            "required residual call is absent: "
            + ",".join(missing_required_calls)
        )

    map_id = mapping_summary.mapping_id(args.workload, args.graph, args.hardware)
    diagnostic = build_diagnostic(args.expected_graph_token, args.diagnostic)
    dfg_report = {
        "schema_version": 1,
        "kind": "dfg_sim_report",
        "workload": args.workload,
        "graph": args.graph,
        "status": "unsupported",
        "metric_definition": "optimistic_pipeline_latency_throughput_sum",
        "operation_semantics_source": "loom.sim.operation_semantics.v1",
        "operation_cost_model_source": "loom.sim.operation_cost.v1",
        "optimistic_cycles": 0,
        "pipeline_latency_throughput_cycles": 0,
        "operation_mix_cycles": 0,
        "memory_address_setup_cycles": 0,
        "cycle_breakdown": [
            {
                "category": "pipeline_latency_throughput",
                "cycles": 0,
                "evidence": args.evidence,
                "modeled": True,
            },
            {
                "category": "operation_mix",
                "cycles": 0,
                "evidence": args.evidence,
                "modeled": True,
            },
            {
                "category": "memory_address_setup",
                "cycles": 0,
                "evidence": args.evidence,
                "modeled": True,
            },
        ],
        "wavefront_steps": 0,
        "event_count": 0,
        "dynamic_work_items": 0,
        "operation_fire_counts": {},
        "final_outputs": [],
        "final_memory_state": {},
        "dfg_mlir_identity": intermediate_artifacts.artifact_id_for_path(dfg_mlir),
        "dfg_mlir_fingerprint": intermediate_artifacts.artifact_fingerprint(dfg_mlir),
        "discovered_graph_ids": graph_ids,
        "residual_call_targets": residual_calls,
        "diagnostics": [diagnostic],
    }
    mapping_artifact = {
        "schema_version": 1,
        "kind": "pnr_mapping",
        "workload": args.workload,
        "hardware": args.hardware,
        "graph": args.graph,
        "mapping_id": map_id,
        "config_id": "loom.default",
        "config_fingerprint": mapping_summary.run_config_tool("--resolved-fingerprint"),
        "component_config_view": "pnr.mapping.v1",
        "component_config_fingerprint": mapping_summary.run_config_tool(
            "--component-fingerprint",
            "--component-view",
            "pnr.mapping.v1",
        ),
        "status": "unsupported",
        "placed_records": 0,
        "routed_edges": 0,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 0,
        "placements": [],
        "routes": [],
        "unrouted_edge_details": [],
        "config_bitstream": [],
        "diagnostics": [diagnostic],
    }
    row = {
        "workload": args.workload,
        "hardware": args.hardware,
        "mapping_id": map_id,
        "placed_records": "",
        "routed_edges": "",
        "unrouted_edges": "",
        "unplaced_records": "",
        "status": "unsupported",
        "diagnostic": diagnostic,
    }
    cycle_row = {
        "kernel": args.workload,
        "dfg_sim_cycles": "",
        "cgra_sim_cycles": "",
        "status": "unsupported",
        "diagnostic": diagnostic,
    }

    write_json(Path(args.dfg_output), dfg_report)
    write_json(Path(args.mapping_output), mapping_artifact)
    intermediate_artifacts.write_csv_rows(
        "sim_cycle",
        intermediate_artifacts.output_path(args.dfg_cycle_output),
        [cycle_row],
    )
    intermediate_artifacts.write_csv_rows(
        "pnr_mapping",
        intermediate_artifacts.output_path(args.mapping_summary_output),
        [row],
    )


def main(argv: list[str]) -> int:
    emit_artifacts(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
