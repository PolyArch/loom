#!/usr/bin/env python3
"""Validate one source-to-Deployment product execution contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    events: list[dict[str, Any]], stage: str, context_kind: str
) -> None:
    rows = [
        payload
        for payload in matching_payloads(
            events, stage=stage, event="derived_context"
        )
        if payload.get("context_kind") == context_kind
    ]
    require(len(rows) >= 2, f"{context_kind} did not expose miss and hit rows")
    require(sum(row.get("cache_misses", -1) for row in rows) == 1,
            f"{context_kind} was constructed more than once")
    require(sum(row.get("cache_hits", -1) for row in rows) >= 1,
            f"{context_kind} was not reused")
    require(all(row.get("construction_count") == 1 for row in rows),
            f"{context_kind} construction count is inconsistent")
    for field in ("construction_time_ns", "retained_bytes", "deterministic_work"):
        require(all(isinstance(row.get(field), int) and row[field] > 0
                    for row in rows),
                f"{context_kind} lacks positive {field}")


def one_invocation(
    events: list[dict[str, Any]], stage: str, statistics_kind: str
) -> dict[str, Any]:
    rows = [
        payload
        for payload in matching_payloads(events, stage=stage, event="statistics")
        if payload.get("statistics_kind") == statistics_kind
    ]
    require(len(rows) == 1, f"expected one {statistics_kind} row")
    return rows[0]


def validate_mapping_work(
    events: list[dict[str, Any]], minimum_spatial_negotiations: int
) -> None:
    tech_ends = matching_payloads(
        events, stage="tech_mapping", event="invocation_end"
    )
    require(len(tech_ends) == 1, "expected one TechMapping invocation")
    require(tech_ends[0].get("candidate_publications", 0) >= 2,
            "scalar gate did not exercise multiple TechMapping candidates")

    validate_context(events, "spatial_pnr", "fabric_static")
    validate_context(events, "spatial_pnr", "fabric_timing")
    validate_context(events, "system_pnr", "system_static")
    validate_context(events, "system_pnr", "system_active")

    spatial = one_invocation(
        events, "spatial_pnr", "spatial_pnr_invocation"
    )
    system = one_invocation(events, "system_pnr", "system_pnr_invocation")
    for name, row in (("Spatial", spatial), ("System", system)):
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
            require(row.get(field) == 0, f"{name} search performed {field}")
        require(row.get("final_closure_attempts") == 1,
                f"{name} search skipped or repeated final closure")
        require(row.get("finalized_restarts") == 1 and
                row.get("publication_slots") == 1,
                f"{name} search did not finalize exactly one result")
    require(system.get("final_verification_attempts") == 1,
            "System search skipped independent candidate verification")
    require(
        spatial.get("negotiation_iteration_slots", 0)
        >= minimum_spatial_negotiations,
        "Spatial search did not exercise the required routing work",
    )


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
) -> None:
    require(manifest.get("schema") == "loom.execution_matrix_workspace.1.0",
            "execution workspace has the wrong schema")
    for field in ("deployment", "workload", "runtime_input", "gem5_binding"):
        validate_reference(manifest.get(field), field)
    require(manifest.get("value_results") == [["0"]],
            "execution cells did not agree with the independent scalar oracle")

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spatial-invocations", type=int, default=1)
    parser.add_argument("--minimum-spatial-negotiations", type=int, default=0)
    parser.add_argument("--required-dataflow-text", action="append", default=[])
    arguments = parser.parse_args()
    require(arguments.spatial_invocations > 0,
            "Spatial invocation count must be positive")
    require(arguments.minimum_spatial_negotiations >= 0,
            "minimum Spatial negotiation count cannot be negative")
    validate_mapping_work(
        read_diagnostics(arguments.diagnostics),
        arguments.minimum_spatial_negotiations,
    )
    validate_manifest(
        read_json(arguments.manifest),
        arguments.manifest,
        arguments.spatial_invocations,
        arguments.required_dataflow_text,
    )


if __name__ == "__main__":
    main()
