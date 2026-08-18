#!/usr/bin/env python3
"""Validate one source-to-Deployment product execution contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def read_diagnostics(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        value = json.loads(line)
        if value.get("schema") != "loom.invocation.diagnostic.1":
            continue
        require(isinstance(value.get("payload"), dict),
                "diagnostic payload must be an object")
        events.append(value)
    require(events, "product build emitted no structured diagnostics")
    return events


def matching_payloads(
    events: list[dict[str, Any]], *, stage: str, event: str
) -> list[dict[str, Any]]:
    return [
        row["payload"]
        for row in events
        if row.get("stage") == stage and row.get("event") == event
    ]


def validate_context(
    events: list[dict[str, Any]],
    stage: str,
    context_kind: str,
    expected_contexts: int | None = 1,
) -> None:
    rows = [
        payload
        for payload in matching_payloads(
            events, stage=stage, event="derived_context"
        )
        if payload.get("context_kind") == context_kind
    ]
    contexts: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get("context_key")
        require(
            isinstance(key, str)
            and len(key) == 64
            and key == key.lower()
            and all(character in "0123456789abcdef" for character in key),
            f"{context_kind} lacks a complete immutable context key",
        )
        contexts.setdefault(key, []).append(row)
    require(contexts, f"{context_kind} emitted no context identities")
    if expected_contexts is not None:
        require(
            len(contexts) == expected_contexts,
            f"{context_kind} has the wrong distinct context count",
        )
    for key, context_rows in contexts.items():
        require(
            sum(row.get("cache_misses", -1) for row in context_rows) == 1,
            f"{context_kind} {key} has the wrong construction count",
        )
        require(
            sum(row.get("cache_hits", -1) for row in context_rows) >= 1,
            f"{context_kind} {key} was not reused",
        )
        require(
            all(row.get("construction_count") == 1 for row in context_rows),
            f"{context_kind} {key} construction count is inconsistent",
        )
        for field in (
            "construction_time_ns",
            "retained_bytes",
            "deterministic_work",
        ):
            values = {row.get(field) for row in context_rows}
            require(
                len(values) == 1
                and all(isinstance(value, int) and value > 0 for value in values),
                f"{context_kind} {key} lacks stable positive {field}",
            )


def search_invocations(
    events: list[dict[str, Any]], stage: str, statistics_kind: str
) -> list[dict[str, Any]]:
    rows = [
        payload
        for payload in matching_payloads(events, stage=stage, event="statistics")
        if payload.get("statistics_kind") == statistics_kind
    ]
    require(rows, f"expected at least one {statistics_kind} row")
    return rows


def validate_mapping_work(
    events: list[dict[str, Any]], expected_system_active_contexts: int | None,
    spatial_search_frontier: bool,
) -> None:
    tech_rows = [
        payload
        for payload in matching_payloads(
            events, stage="tech_mapping", event="statistics"
        )
        if payload.get("statistics_kind")
        == "application_tech_root_supply_frontier"
    ]
    nonempty_tech = [
        row for row in tech_rows if row.get("candidate_publications", 0) > 0
    ]
    require(nonempty_tech,
            "no software alternative published a TechMapping frontier")
    selected_tech = nonempty_tech[-1]
    require(selected_tech.get("candidate_publications", 0) >= 2,
            "product gate did not exercise multiple TechMapping candidates")

    validate_context(events, "spatial_pnr", "fabric_static")
    validate_context(events, "spatial_pnr", "fabric_timing")
    validate_context(events, "system_pnr", "system_static")
    validate_context(
        events,
        "system_pnr",
        "system_active",
        expected_system_active_contexts,
    )

    spatial = search_invocations(
        events, "spatial_pnr", "spatial_pnr_invocation"
    )
    if spatial_search_frontier:
        require(len(spatial) > 1,
                "Spatial search frontier did not exercise multiple searches")
    system = search_invocations(
        events, "system_pnr", "system_pnr_invocation"
    )
    require(len(system) == 1,
            "expected one verified System search result")
    for name, rows in (("Spatial", spatial), ("System", system)):
        for row in rows:
            require(row.get("closure_status") == "semantic_limit_reached",
                    f"{name} search did not stop at its verified product result")
            require(row.get("candidate_publications") == 1,
                    f"{name} search published more than one candidate")
            require(row.get("seed_attempt_slots") == 1 and
                    row.get("prepared_seeds") == 1,
                    f"{name} search prepared unexpected restart work")
            for field in (
                "calibration_proposal_slots",
                "annealing_base_proposal_slots",
                "annealing_movable_proposal_slots",
                "annealing_accepted_actions",
                "exact_repair_invocations",
            ):
                require(row.get(field) == 0,
                        f"{name} search performed {field}")
            require(row.get("final_closure_attempts") == 1,
                    f"{name} search skipped or repeated final closure")
            require(row.get("finalized_restarts") == 1 and
                    row.get("publication_slots") == 1,
                    f"{name} search did not finalize exactly one result")
    require(system[0].get("final_verification_attempts") == 1,
            "System search skipped independent candidate verification")
    require(system[0].get("mutation_oracle_verification_attempts") == 0,
            "first-verified System search entered mutation verification")
    require(all(row.get("endpoint_expansion_slots", 0) > 0 for row in spatial),
            "a Spatial search did not exercise endpoint routing")


def validate_spatial_unconditional_handshake(
    events: list[dict[str, Any]],
) -> None:
    static_rows = [
        payload
        for payload in matching_payloads(
            events, stage="spatial_pnr", event="derived_context"
        )
        if payload.get("context_kind") == "fabric_static"
    ]
    unconditional_counts = {
        row.get("handshake_unconditional_arc_count") for row in static_rows
    }
    require(len(unconditional_counts) == 1,
            "Fabric static handshake arc count is inconsistent")
    unconditional_count = next(iter(unconditional_counts))
    require(isinstance(unconditional_count, int) and unconditional_count > 0,
            "Fabric static context has no unconditional handshake arcs")

    active_rows = [
        payload
        for payload in matching_payloads(
            events, stage="spatial_pnr", event="derived_context"
        )
        if payload.get("context_kind") == "spatial_active_handshake"
    ]
    require(active_rows, "Spatial search emitted no active handshake context")
    require(all(
        row.get("fabric_unconditional_arc_count") == unconditional_count and
        row.get("materialized_arc_count", 0) > unconditional_count
        for row in active_rows
    ), "active handshake graph omitted Fabric unconditional dependencies")


def validate_reference(value: Any, context: str) -> None:
    require(isinstance(value, dict), f"{context} must be an artifact reference")
    identity = value.get("artifact")
    require(isinstance(identity, str) and len(identity) == 64 and
            all(character in "0123456789abcdef" for character in identity),
            f"{context} has an invalid artifact identity")


def validate_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    spatial_invocations: int,
    required_dataflow_text: list[str],
    mapping_inspector: str | None,
    require_actor_multicast: bool,
    require_operand_queue_atomic_fanout: bool,
    require_memory_engine: bool,
    require_memory_internal_edge: bool,
    require_temporal_memory: bool,
    require_register_fifo: bool,
    require_packed_switch_row: bool,
    require_temporal_dispatch: bool,
    dense_coordinate_rank: int | None,
    require_unique_dense_coordinates: bool,
    expected_unique_dispatch_targets: int | None,
    minimum_unique_acc_cores: int,
) -> None:
    require(manifest.get("schema") == "loom.execution_matrix_workspace.1.2",
            "execution workspace has the wrong schema")
    for field in ("deployment", "workload", "runtime_input", "gem5_binding"):
        validate_reference(manifest.get(field), field)
    require(manifest.get("value_results") == [["0"]],
            "execution cells did not agree with the independent product oracle")

    runs = manifest.get("runs")
    require(isinstance(runs, list), "execution workspace runs must be an array")
    system_runs = [run for run in runs if run.get("scope") == "system"]
    spatial_runs = [run for run in runs if run.get("scope") == "spatial"]
    require(
        len(system_runs) == 2
        and {run.get("engine") for run in system_runs} == {"dfg", "cgra"},
        "execution workspace does not contain both System cells",
    )
    require(
        len(spatial_runs) == spatial_invocations * 2,
        "execution workspace has the wrong Spatial invocation count",
    )
    for ordinal in range(spatial_invocations):
        invocation_runs = [
            run
            for run in spatial_runs
            if run.get("invocation_ordinal") == ordinal
        ]
        require(
            len(invocation_runs) == 2
            and {run.get("engine") for run in invocation_runs}
            == {"dfg", "cgra"},
            f"Spatial invocation {ordinal} lacks both execution cells",
        )
        coordinates = [run.get("dense_coordinates") for run in invocation_runs]
        require(
            len(coordinates) == 2 and coordinates[0] == coordinates[1],
            f"Spatial invocation {ordinal} engines disagree on coordinates",
        )
        require(
            isinstance(coordinates[0], list)
            and all(
                isinstance(coordinate, int) and coordinate >= 0
                for coordinate in coordinates[0]
            ),
            f"Spatial invocation {ordinal} has invalid coordinates",
        )
        if dense_coordinate_rank is not None:
            require(
                len(coordinates[0]) == dense_coordinate_rank,
                f"Spatial invocation {ordinal} has the wrong coordinate rank",
            )
        target_ordinals = [
            run.get("dispatch_target_ordinal") for run in invocation_runs
        ]
        acc_core_references = [
            run.get("acc_core_ref") for run in invocation_runs
        ]
        context_keys = [
            run.get("execution_context_key") for run in invocation_runs
        ]
        require(
            len(target_ordinals) == 2
            and target_ordinals[0] == target_ordinals[1]
            and isinstance(target_ordinals[0], int)
            and target_ordinals[0] >= 0,
            f"Spatial invocation {ordinal} engines disagree on dispatch target",
        )
        for values, name in (
            (acc_core_references, "AccCore reference"),
            (context_keys, "execution-context key"),
        ):
            require(
                len(values) == 2
                and values[0] == values[1]
                and isinstance(values[0], str)
                and len(values[0]) > 0
                and len(values[0]) % 2 == 0
                and all(
                    character in "0123456789abcdef" for character in values[0]
                ),
                f"Spatial invocation {ordinal} has an invalid {name}",
            )
    if require_unique_dense_coordinates:
        coordinate_points = {
            tuple(run["dense_coordinates"])
            for run in spatial_runs
            if run.get("engine") == "dfg"
        }
        require(
            len(coordinate_points) == spatial_invocations,
            "Spatial invocations do not have unique dense coordinates",
        )
    unique_dispatch_targets = {
        run["dispatch_target_ordinal"]
        for run in spatial_runs
        if run.get("engine") == "dfg"
    }
    if expected_unique_dispatch_targets is not None:
        require(
            len(unique_dispatch_targets) == expected_unique_dispatch_targets,
            "Spatial invocations use the wrong number of dispatch targets",
        )
    unique_acc_cores = {
        run["acc_core_ref"]
        for run in spatial_runs
        if run.get("engine") == "dfg"
    }
    require(
        len(unique_acc_cores) >= minimum_unique_acc_cores,
        "Spatial invocations use fewer AccCores than required",
    )
    for run in runs:
        label = f"{run.get('scope')}/{run.get('engine')}"
        for field in ("request", "evidence", "execution"):
            validate_reference(run.get(field), f"{label} {field}")
        if run.get("scope") == "spatial":
            for field in ("dataflow", "spatial_mapping", "hardware_implementation"):
                validate_reference(run.get(field), f"{label} {field}")
            require(isinstance(run.get("terminal_cycle"), dict),
                    f"{label} lacks a terminal cycle")
        else:
            require(all(isinstance(run.get(field), int)
                        for field in ("entry_tick", "exit_tick", "terminal_tick")),
                    f"{label} lacks exact gem5 ticks")

    dataflow_identities = {
        run["dataflow"]["artifact"] for run in spatial_runs
    }
    require(
        len(dataflow_identities) == 1,
        "Spatial invocations do not share one exact Dataflow artifact",
    )
    dataflow_path = manifest_path.parent / "objects" / next(
        iter(dataflow_identities)
    )
    dataflow = dataflow_path.read_bytes()
    for text in required_dataflow_text:
        require(
            text.encode("ascii") in dataflow,
            f"canonical Dataflow does not contain {text}",
        )

    if (
        require_actor_multicast
        or require_operand_queue_atomic_fanout
        or require_memory_engine
        or require_memory_internal_edge
        or require_temporal_memory
        or require_register_fifo
        or require_packed_switch_row
        or require_temporal_dispatch
    ):
        require(mapping_inspector is not None,
                "Mapping feature validation requires a Mapping inspector")
        mapping_identities = {
            run["spatial_mapping"]["artifact"] for run in spatial_runs
        }
        reports: list[dict[str, Any]] = []
        for identity in sorted(mapping_identities):
            completed = subprocess.run(
                [mapping_inspector, str(manifest_path.parent / "objects"),
                 identity],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            require(isinstance(report, dict),
                    "Mapping inspection report must be an object")
            require(
                report.get("schema")
                == "loom.test.product_mapping_inspection.1",
                "Mapping inspection report has the wrong schema",
            )
            reports.append(report)
        if require_actor_multicast:
            require(
                any(report.get("actor_multicast_route_count", 0) > 0 and
                    report.get("maximum_actor_multicast_sinks", 0) >= 2
                    for report in reports),
                "Spatial Mapping contains no complete actor-result multicast",
            )
        if require_operand_queue_atomic_fanout:
            require(
                any(
                    report.get("operand_queue_atomic_fanout_group_count", 0)
                    > 0
                    and report.get("maximum_operand_queue_matches", 0) >= 2
                    for report in reports
                ),
                "Spatial Mapping contains no atomic operand-queue fanout",
            )
        if require_memory_engine:
            require(
                any(
                    report.get("spatial_memory_engine_binding_count", 0) > 0
                    and report.get("configured_memory_occurrence_count", 0) > 0
                    and report.get(
                        "configured_memory_active_operation_row_count", 0
                    ) > 0
                    for report in reports
                ),
                "Spatial Mapping contains no configured Memory Engine",
            )
        if require_memory_internal_edge:
            require(
                any(
                    report.get("memory_internal_edge_count", 0) > 0
                    and report.get(
                        "fabric_memory_template_internal_connection_count", 0
                    ) > 0
                    and report.get(
                        "fabric_memory_template_with_internal_connection_count",
                        0,
                    ) > 0
                    for report in reports
                ),
                "Spatial Mapping contains no Fabric-backed memory internal "
                "edge",
            )
        if require_temporal_memory:
            require(
                any(
                    report.get("temporal_memory_engine_binding_count", 0) > 0
                    and report.get("temporal_memory_operation_count", 0) > 0
                    and report.get("temporal_memory_occurrence_count", 0)
                    == report.get("dense_temporal_memory_occurrence_count", -1)
                    and report.get(
                        "temporal_memory_external_ingress_claim_count", 0
                    ) > 0
                    and report.get(
                        "temporal_memory_external_ingress_claim_count", 0
                    ) == report.get(
                        "unique_temporal_memory_external_ingress_claim_count",
                        -1,
                    )
                    for report in reports
                ),
                "Spatial Mapping contains no closed Temporal Memory Engine",
            )
        if require_register_fifo:
            require(
                any(
                    report.get("register_fifo_transfer_count", 0) > 0
                    for report in reports
                ),
                "Spatial Mapping contains no RegFIFO local transfer",
            )
        if require_packed_switch_row:
            require(
                any(
                    report.get("shared_packed_switch_row_count", 0) > 0
                    and report.get(
                        "maximum_packed_switch_row_signatures", 0
                    ) >= 2
                    for report in reports
                ),
                "Spatial Mapping contains no shared packed switch row",
            )
        if require_temporal_dispatch:
            require(
                any(
                    report.get("temporal_compute_binding_count", 0) > 0
                    and report.get("temporal_dispatch_domain_count", 0) > 0
                    and report.get("temporal_dispatch_candidate_count", 0) > 0
                    for report in reports
                ),
                "Spatial Mapping contains no Temporal PE dispatch domain",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spatial-invocations", type=int, default=1)
    parser.add_argument("--expected-system-active-contexts", type=int)
    parser.add_argument("--spatial-search-frontier", action="store_true")
    parser.add_argument(
        "--require-spatial-unconditional-handshake", action="store_true"
    )
    parser.add_argument("--required-dataflow-text", action="append", default=[])
    parser.add_argument("--mapping-inspector")
    parser.add_argument("--require-actor-multicast", action="store_true")
    parser.add_argument(
        "--require-operand-queue-atomic-fanout", action="store_true"
    )
    parser.add_argument("--require-memory-engine", action="store_true")
    parser.add_argument(
        "--require-memory-internal-edge", action="store_true"
    )
    parser.add_argument("--require-temporal-memory", action="store_true")
    parser.add_argument("--require-register-fifo", action="store_true")
    parser.add_argument("--require-packed-switch-row", action="store_true")
    parser.add_argument("--require-temporal-dispatch", action="store_true")
    parser.add_argument("--dense-coordinate-rank", type=int)
    parser.add_argument(
        "--require-unique-dense-coordinates", action="store_true"
    )
    parser.add_argument("--expected-unique-dispatch-targets", type=int)
    parser.add_argument("--minimum-unique-acc-cores", type=int, default=1)
    arguments = parser.parse_args()
    require(arguments.spatial_invocations > 0,
            "Spatial invocation count must be positive")
    require(
        arguments.expected_system_active_contexts is None
        or arguments.expected_system_active_contexts > 0,
        "expected System active context count must be positive",
    )
    require(arguments.minimum_unique_acc_cores > 0,
            "minimum unique AccCore count must be positive")
    require(
        arguments.expected_unique_dispatch_targets is None
        or arguments.expected_unique_dispatch_targets > 0,
        "expected unique dispatch target count must be positive",
    )
    require(
        arguments.dense_coordinate_rank is None
        or arguments.dense_coordinate_rank >= 0,
        "dense coordinate rank must be nonnegative",
    )
    events = read_diagnostics(arguments.diagnostics)
    validate_mapping_work(
        events, arguments.expected_system_active_contexts,
        arguments.spatial_search_frontier,
    )
    if arguments.require_spatial_unconditional_handshake:
        validate_spatial_unconditional_handshake(events)
    validate_manifest(
        read_json(arguments.manifest),
        arguments.manifest,
        arguments.spatial_invocations,
        arguments.required_dataflow_text,
        arguments.mapping_inspector,
        arguments.require_actor_multicast,
        arguments.require_operand_queue_atomic_fanout,
        arguments.require_memory_engine,
        arguments.require_memory_internal_edge,
        arguments.require_temporal_memory,
        arguments.require_register_fifo,
        arguments.require_packed_switch_row,
        arguments.require_temporal_dispatch,
        arguments.dense_coordinate_rank,
        arguments.require_unique_dense_coordinates,
        arguments.expected_unique_dispatch_targets,
        arguments.minimum_unique_acc_cores,
    )


if __name__ == "__main__":
    main()
