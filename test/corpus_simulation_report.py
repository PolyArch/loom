#!/usr/bin/env python3
"""Strict typed projection for corpus DFG simulation results."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DseExecutionMetrics:
    plan_executions: int
    generate_invocations: int
    incomplete_generate_invocations: int
    input_bindings: int
    input_artifacts: int
    output_bindings: int
    output_artifacts: int
    generate_lineage_edges: int

    @staticmethod
    def zero() -> "DseExecutionMetrics":
        return DseExecutionMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    def combine(self, other: "DseExecutionMetrics") -> "DseExecutionMetrics":
        return DseExecutionMetrics(
            plan_executions=self.plan_executions + other.plan_executions,
            generate_invocations=(
                self.generate_invocations + other.generate_invocations
            ),
            incomplete_generate_invocations=(
                self.incomplete_generate_invocations
                + other.incomplete_generate_invocations
            ),
            input_bindings=self.input_bindings + other.input_bindings,
            input_artifacts=self.input_artifacts + other.input_artifacts,
            output_bindings=self.output_bindings + other.output_bindings,
            output_artifacts=self.output_artifacts + other.output_artifacts,
            generate_lineage_edges=(
                self.generate_lineage_edges + other.generate_lineage_edges
            ),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "generate_invocations": self.generate_invocations,
            "generate_lineage_edges": self.generate_lineage_edges,
            "incomplete_generate_invocations": (self.incomplete_generate_invocations),
            "input_artifacts": self.input_artifacts,
            "input_bindings": self.input_bindings,
            "output_artifacts": self.output_artifacts,
            "output_bindings": self.output_bindings,
            "plan_executions": self.plan_executions,
        }


def parse_dse_execution_projection(value: object) -> DseExecutionMetrics:
    if not isinstance(value, dict):
        raise ValueError("DSE execution summary is not an object")
    expected_fields = set(DseExecutionMetrics.__dataclass_fields__)
    if set(value) != expected_fields:
        raise ValueError("DSE execution summary field inventory changed")
    counts: dict[str, int] = {}
    for field in expected_fields:
        count = value[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"DSE execution summary has invalid {field}")
        counts[field] = count
    if counts["plan_executions"] == 0:
        raise ValueError("DSE execution summary has no plan execution")
    if counts["generate_invocations"] == 0:
        raise ValueError("DSE execution summary has no Generate invocation")
    if counts["incomplete_generate_invocations"] != 0:
        raise ValueError("DSE execution summary retains incomplete Generate")
    if counts["generate_lineage_edges"] == 0:
        raise ValueError("DSE execution summary has no Generate lineage")
    return DseExecutionMetrics(**counts)


@dataclass(frozen=True)
class DfgSimulationMetrics:
    graphs: int
    actors: int
    dynamic_calls: int
    value_lanes_compared: int
    memory_bytes_compared: int
    floating_variance_bytes: int
    wavefront_steps: int
    event_count: int
    simulation_seconds: float
    operation_firings: dict[str, int]
    selected_source_files: tuple[str, ...]
    dse_execution: DseExecutionMetrics | None = None
    canonical_dataflow_identity: str | None = None
    simulation_workload_identity: str | None = None
    simulation_runtime_input_identity: str | None = None
    execution_terminal: str | None = None

    @staticmethod
    def zero() -> "DfgSimulationMetrics":
        return DfgSimulationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0.0, {}, ())

    @property
    def wavefront_steps_per_second(self) -> float:
        if self.simulation_seconds == 0.0:
            return 0.0
        return self.wavefront_steps / self.simulation_seconds

    def combine(self, other: "DfgSimulationMetrics") -> "DfgSimulationMetrics":
        firings = dict(self.operation_firings)
        for operation, count in other.operation_firings.items():
            firings[operation] = firings.get(operation, 0) + count
        return DfgSimulationMetrics(
            graphs=self.graphs + other.graphs,
            actors=self.actors + other.actors,
            dynamic_calls=self.dynamic_calls + other.dynamic_calls,
            value_lanes_compared=(
                self.value_lanes_compared + other.value_lanes_compared
            ),
            memory_bytes_compared=(
                self.memory_bytes_compared + other.memory_bytes_compared
            ),
            floating_variance_bytes=(
                self.floating_variance_bytes + other.floating_variance_bytes
            ),
            wavefront_steps=self.wavefront_steps + other.wavefront_steps,
            event_count=self.event_count + other.event_count,
            simulation_seconds=self.simulation_seconds + other.simulation_seconds,
            operation_firings=firings,
            selected_source_files=tuple(
                sorted(
                    set(self.selected_source_files) | set(other.selected_source_files)
                )
            ),
            dse_execution=(
                self.dse_execution.combine(other.dse_execution)
                if self.dse_execution is not None and other.dse_execution is not None
                else self.dse_execution or other.dse_execution
            ),
        )

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dynamic_calls": self.dynamic_calls,
            "event_count": self.event_count,
            "floating_variance_bytes": self.floating_variance_bytes,
            "floating_variance_kind": (
                "selected_decision_replay" if self.floating_variance_bytes else "none"
            ),
            "value_lanes_compared": self.value_lanes_compared,
            "memory_bytes_compared": self.memory_bytes_compared,
            "operation_firings": dict(sorted(self.operation_firings.items())),
            "selected_source_files": list(self.selected_source_files),
            "simulation_seconds": self.simulation_seconds,
            "wavefront_steps": self.wavefront_steps,
            "wavefront_steps_per_second": self.wavefront_steps_per_second,
        }
        identities = (
            self.canonical_dataflow_identity,
            self.simulation_workload_identity,
            self.simulation_runtime_input_identity,
        )
        if all(identity is not None for identity in identities):
            payload["artifacts"] = {
                "canonical_dataflow": self.canonical_dataflow_identity,
                "simulation_runtime_input": self.simulation_runtime_input_identity,
                "simulation_workload": self.simulation_workload_identity,
            }
        if self.execution_terminal is not None:
            payload["execution_terminal"] = self.execution_terminal
        if self.dse_execution is not None:
            payload["dse_execution"] = self.dse_execution.as_dict()
        return payload


def parse_dfg_simulation_report(
    path: Path,
) -> tuple[DfgSimulationMetrics | None, str | None]:
    """Parse the exact substantive projection emitted by loom-dfg-run."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        return None, f"cannot read DFG simulation report {path}: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"malformed DFG simulation report {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"DFG simulation report is not a JSON object: {path}"
    expected_fields = {
        "actor_refs",
        "actors",
        "artifacts",
        "compiler_target",
        "dynamic_calls",
        "dse_execution",
        "event_count",
        "execution_terminal",
        "floating_variance_bytes",
        "floating_variance_kind",
        "graphs",
        "kind",
        "memory_bytes_compared",
        "operation_firings",
        "selected_source_files",
        "simulation_seconds",
        "source_oracle",
        "status",
        "transform_lineage",
        "value_lanes_compared",
        "wavefront_steps",
        "wavefront_steps_per_second",
    }
    if set(payload) != expected_fields:
        return None, (
            "DFG simulation report contains missing or unexpected fields "
            f"(expected {sorted(expected_fields)}): {path}"
        )
    if payload["kind"] != "source_backed_dfg_comparison" or payload["status"] != "pass":
        return None, f"DFG simulation report has invalid kind or status: {path}"

    target = payload["compiler_target"]
    target_fields = {
        "data_layout",
        "host_binding",
        "instruction_bindings",
        "instruction_core_count",
        "target_triple",
    }
    if not isinstance(target, dict) or set(target) != target_fields:
        return None, f"DFG simulation report has invalid compiler target: {path}"

    def is_artifact_identity(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    artifacts = payload["artifacts"]
    artifact_fields = {
        "canonical_dataflow",
        "canonical_dataflow_initial",
        "simulation_runtime_input",
        "simulation_workload",
        "structured_initial",
        "structured_selected",
    }
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != artifact_fields
        or any(not is_artifact_identity(artifacts[field]) for field in artifact_fields)
    ):
        return None, f"DFG simulation report has invalid artifact identities: {path}"

    actor_refs = payload["actor_refs"]
    actor_entities: set[str] = set()
    if not isinstance(actor_refs, list) or not actor_refs:
        return None, f"DFG simulation report has no stable ActorRefs: {path}"
    for reference in actor_refs:
        if (
            not isinstance(reference, dict)
            or set(reference) != {"artifact", "entity"}
            or reference["artifact"] != artifacts["canonical_dataflow"]
            or not isinstance(reference["entity"], str)
            or re.fullmatch(r"0|[1-9][0-9]*", reference["entity"]) is None
            or reference["entity"] in actor_entities
        ):
            return None, f"DFG simulation report has an invalid ActorRef: {path}"
        actor_entities.add(reference["entity"])

    source_oracle = payload["source_oracle"]
    if (
        not isinstance(source_oracle, dict)
        or set(source_oracle) != {"comparison", "entry_result"}
        or source_oracle["comparison"] != "equivalent"
        or (
            source_oracle["entry_result"] is not None
            and (
                isinstance(source_oracle["entry_result"], bool)
                or not isinstance(source_oracle["entry_result"], int)
            )
        )
    ):
        return None, f"DFG simulation report has an invalid source oracle: {path}"

    transform_lineage = payload["transform_lineage"]
    lineage_fields = {
        "dataflow_rewrite",
        "execution_shape",
        "memory_communication",
        "ownership",
        "schedule",
        "special_math_accuracy",
    }
    if (
        not isinstance(transform_lineage, dict)
        or set(transform_lineage) != lineage_fields
    ):
        return None, f"DFG simulation report has invalid transform lineage: {path}"
    for field in ("ownership", "execution_shape", "special_math_accuracy", "schedule"):
        count = transform_lineage[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None, f"DFG simulation report has invalid transform lineage: {path}"
    for field in ("memory_communication", "dataflow_rewrite"):
        kinds = transform_lineage[field]
        if not isinstance(kinds, list) or any(
            isinstance(kind, bool) or not isinstance(kind, int) or kind < 0
            for kind in kinds
        ):
            return None, f"DFG simulation report has invalid transform lineage: {path}"
    if payload["execution_terminal"] != "retired":
        return None, f"DFG simulation report has invalid execution terminal: {path}"

    try:
        dse_execution = parse_dse_execution_projection(payload["dse_execution"])
    except ValueError as exc:
        return None, f"DFG simulation report has invalid DSE execution: {exc}: {path}"

    instruction_bindings = target["instruction_bindings"]
    instruction_core_count = target["instruction_core_count"]
    if (
        not is_artifact_identity(target["host_binding"])
        or not isinstance(instruction_bindings, list)
        or not instruction_bindings
        or any(not is_artifact_identity(value) for value in instruction_bindings)
        or len(instruction_bindings) != len(set(instruction_bindings))
        or isinstance(instruction_core_count, bool)
        or not isinstance(instruction_core_count, int)
        or instruction_core_count < len(instruction_bindings)
        or not isinstance(target["target_triple"], str)
        or not target["target_triple"]
        or not isinstance(target["data_layout"], str)
        or not target["data_layout"]
    ):
        return None, f"DFG simulation report has invalid compiler target: {path}"

    integers: dict[str, int] = {}
    for field in (
        "actors",
        "dynamic_calls",
        "event_count",
        "graphs",
        "wavefront_steps",
    ):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None, (
                "DFG simulation report is not a substantive execution: "
                f"{field} must be a positive integer: {path}"
            )
        integers[field] = value

    if len(actor_refs) != integers["actors"]:
        return (
            None,
            f"DFG simulation report has an incomplete ActorRef inventory: {path}",
        )

    for field in ("value_lanes_compared", "memory_bytes_compared"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None, (
                "DFG simulation report has an invalid observable count: "
                f"{field}: {path}"
            )
        integers[field] = value
    if integers["value_lanes_compared"] == 0 and integers["memory_bytes_compared"] == 0:
        return None, (
            "DFG simulation report is not a substantive execution: no value "
            f"or memory observation was compared: {path}"
        )

    variance_bytes = payload["floating_variance_bytes"]
    variance_kind = payload["floating_variance_kind"]
    if (
        isinstance(variance_bytes, bool)
        or not isinstance(variance_bytes, int)
        or variance_bytes < 0
    ):
        return None, f"DFG simulation report has invalid floating variance: {path}"
    expected_variance_kind = "selected_decision_replay" if variance_bytes else "none"
    if variance_kind != expected_variance_kind:
        return None, f"DFG simulation report has inconsistent floating variance: {path}"

    seconds = payload["simulation_seconds"]
    rate = payload["wavefront_steps_per_second"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds <= 0.0
    ):
        return None, f"DFG simulation report has invalid simulation_seconds: {path}"
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate <= 0.0
    ):
        return None, (
            f"DFG simulation report has invalid wavefront_steps_per_second: {path}"
        )
    derived_rate = integers["wavefront_steps"] / float(seconds)
    if not math.isclose(float(rate), derived_rate, rel_tol=1e-12, abs_tol=1e-9):
        return None, f"DFG simulation report rate is not derived from totals: {path}"

    raw_firings = payload["operation_firings"]
    if not isinstance(raw_firings, dict) or not raw_firings:
        return None, f"DFG simulation report has no operation firings: {path}"
    firings: dict[str, int] = {}
    for operation, count in raw_firings.items():
        if not isinstance(operation, str) or not operation:
            return None, f"DFG simulation report has an invalid operation key: {path}"
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            return None, (
                "DFG simulation report has a non-positive operation firing "
                f"count for {operation!r}: {path}"
            )
        firings[operation] = count

    raw_source_files = payload["selected_source_files"]
    if (
        not isinstance(raw_source_files, list)
        or not raw_source_files
        or any(not isinstance(path, str) or not path for path in raw_source_files)
        or raw_source_files != sorted(set(raw_source_files))
    ):
        return None, (
            f"DFG simulation report has noncanonical selected source files: {path}"
        )

    return (
        DfgSimulationMetrics(
            graphs=integers["graphs"],
            actors=integers["actors"],
            dynamic_calls=integers["dynamic_calls"],
            value_lanes_compared=integers["value_lanes_compared"],
            memory_bytes_compared=integers["memory_bytes_compared"],
            floating_variance_bytes=variance_bytes,
            wavefront_steps=integers["wavefront_steps"],
            event_count=integers["event_count"],
            simulation_seconds=float(seconds),
            operation_firings=firings,
            selected_source_files=tuple(raw_source_files),
            dse_execution=dse_execution,
            canonical_dataflow_identity=artifacts["canonical_dataflow"],
            simulation_workload_identity=artifacts["simulation_workload"],
            simulation_runtime_input_identity=artifacts["simulation_runtime_input"],
            execution_terminal=payload["execution_terminal"],
        ),
        None,
    )
