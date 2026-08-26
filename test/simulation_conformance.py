"""Paired Spatial/System simulation conformance policy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence


SPATIAL_REFERENCE_FLOOR_SECONDS = 0.1
SYSTEM_BUDGET_MULTIPLIER = 3.0
HARD_FAILURE_RATIO = 10.0
REFERENCE_RATE_TARGET_CYCLES_PER_SECOND = 100_000
REFERENCE_RATE_TARGET_HZ = float(REFERENCE_RATE_TARGET_CYCLES_PER_SECOND)
RESERVED_DEVELOPMENT_CPUS = 4
MAX_OUTER_WORKERS = 120
DEFAULT_PROCESS_TIMEOUT_SECONDS = 180.0
PROCESS_TERMINATION_GRACE_SECONDS = 1.0
SPATIAL_WARMUP_RUNS = 1
SPATIAL_MEASUREMENT_RUNS = 3
CGRA_QUALIFICATION_LIMIT_NANOSECONDS = 45_000_000_000
CGRA_QUALIFICATION_WARMUP_RUNS = 1
CGRA_QUALIFICATION_MEASUREMENT_RUNS = 3
CGRA_GATE_CONFIGURATION = (
    Path(__file__).resolve().parent / "data" / "cgra-simulation-gate-v1.json"
)
CGRA_OPERATOR_GATE_RELATIVE_PATH = "test/data/corpus-operator-gate-v1.jsonl"
CGRA_OPERATOR_GATE = (
    Path(__file__).resolve().parents[1] / CGRA_OPERATOR_GATE_RELATIVE_PATH
)
CGRA_REPRESENTATIVE_WORKLOADS = (
    "vecadd",
    "vector_pack",
    "matmul",
    "spmm",
    "gather",
    "edge_update",
    "fir_filter",
    "conv2d",
    "stencil3d",
    "attention",
)


def _require_finite_positive(value: float, what: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{what} must be finite and positive")


def _require_nonnegative(value: float, what: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{what} must be finite and nonnegative")


@dataclass(frozen=True)
class ActiveExecutionTiming:
    active_wall_seconds: float
    reference_cycles: int
    cgra_engine_active_wall_seconds: float = 0.0
    cgra_engine_active_process_cpu_seconds: float = 0.0
    gem5_active_process_cpu_seconds: float = 0.0
    gem5_observation_process_cpu_seconds: float = 0.0
    engine_process_cpu_seconds: float = 0.0
    cgra_input_load_wall_seconds: float = 0.0
    cgra_input_load_process_cpu_seconds: float = 0.0
    bridge_callback_cpu_seconds: float = 0.0
    bridge_wait_wall_seconds: float = 0.0
    setup_wall_seconds: float = 0.0
    preparation_wall_seconds: float = 0.0
    gem5_configuration_wall_seconds: float = 0.0
    provider_wall_seconds: float = 0.0
    provider_cpu_seconds: float = 0.0
    gem5_observation_wall_seconds: float = 0.0
    observation_wall_seconds: float = 0.0
    cgra_observation_projection_wall_seconds: float = 0.0
    cgra_observation_projection_process_cpu_seconds: float = 0.0
    cgra_artifact_publication_wall_seconds: float = 0.0
    cgra_artifact_publication_process_cpu_seconds: float = 0.0
    bridge_message_count: int = 0
    accelerator_invocation_count: int = 0
    cgra_event_frame_count: int = 0
    peak_resident_bytes: int = 0

    def __post_init__(self) -> None:
        _require_finite_positive(self.active_wall_seconds, "active wall time")
        if self.reference_cycles < 0:
            raise ValueError("reference-cycle count must be nonnegative")
        times = {
            "CGRA engine active wall time": self.cgra_engine_active_wall_seconds,
            "CGRA engine active process CPU time": (
                self.cgra_engine_active_process_cpu_seconds
            ),
            "gem5 active process CPU time": self.gem5_active_process_cpu_seconds,
            "gem5 observation process CPU time": (
                self.gem5_observation_process_cpu_seconds
            ),
            "engine lifecycle process CPU time": self.engine_process_cpu_seconds,
            "CGRA input-load wall time": self.cgra_input_load_wall_seconds,
            "CGRA input-load process CPU time": (
                self.cgra_input_load_process_cpu_seconds
            ),
            "Bridge callback CPU time": self.bridge_callback_cpu_seconds,
            "Bridge engine wait time": self.bridge_wait_wall_seconds,
            "setup wall time": self.setup_wall_seconds,
            "preparation wall time": self.preparation_wall_seconds,
            "gem5 configuration wall time": self.gem5_configuration_wall_seconds,
            "provider wall time": self.provider_wall_seconds,
            "provider CPU time": self.provider_cpu_seconds,
            "gem5 observation wall time": self.gem5_observation_wall_seconds,
            "observation wall time": self.observation_wall_seconds,
            "CGRA observation-projection wall time": (
                self.cgra_observation_projection_wall_seconds
            ),
            "CGRA observation-projection process CPU time": (
                self.cgra_observation_projection_process_cpu_seconds
            ),
            "CGRA artifact-publication wall time": (
                self.cgra_artifact_publication_wall_seconds
            ),
            "CGRA artifact-publication process CPU time": (
                self.cgra_artifact_publication_process_cpu_seconds
            ),
        }
        for what, value in times.items():
            _require_nonnegative(value, what)
        counts = {
            "Bridge message count": self.bridge_message_count,
            "accelerator invocation count": self.accelerator_invocation_count,
            "CGRA event-frame count": self.cgra_event_frame_count,
            "peak resident bytes": self.peak_resident_bytes,
        }
        for what, value in counts.items():
            if value < 0:
                raise ValueError(f"{what} must be nonnegative")


@dataclass(frozen=True)
class PairedSystemBudget:
    spatial_reference_seconds: float
    spatial_absolute_budget_seconds: float
    system_budget_seconds: float
    hard_failure_seconds: float


@dataclass(frozen=True)
class PairedExecutionResult:
    spatial_reference_seconds: float
    spatial_absolute_budget_seconds: float
    system_to_spatial_ratio: float
    system_budget_seconds: float
    within_system_budget: bool
    hard_ratio_failure: bool
    reference_cycles_per_second: float
    meets_reference_rate_target: bool
    system_timing: ActiveExecutionTiming


class ProcessDisposition(Enum):
    COMPLETED = "completed"
    NONZERO_EXIT = "nonzero_exit"
    TIMED_OUT = "timed_out"
    CLEANUP_FAILED = "cleanup_failed"


@dataclass(frozen=True)
class ProcessExecution:
    disposition: ProcessDisposition
    command: tuple[str, ...]
    elapsed_seconds: float
    return_code: int | None
    stdout: str
    stderr: str
    process_group_terminated: bool


@dataclass(frozen=True)
class ExecutionMatrixMeasurement:
    cell: str
    paired_work_fingerprint: str
    deterministic_work: int
    accelerator_reference_cycles: int
    gem5_ticks: int | None
    timing: ActiveExecutionTiming


class ConformanceDisposition(Enum):
    PASSED = "passed"
    SPATIAL_EXECUTION_FAILED = "spatial_execution_failed"
    SPATIAL_TIMED_OUT = "spatial_timed_out"
    SPATIAL_BUDGET_EXCEEDED = "spatial_budget_exceeded"
    SYSTEM_EXECUTION_FAILED = "system_execution_failed"
    SYSTEM_TIMED_OUT = "system_timed_out"
    PROCESS_CLEANUP_FAILED = "process_cleanup_failed"
    PAIRED_WORK_MISMATCH = "paired_work_mismatch"
    SYSTEM_BUDGET_EXCEEDED = "system_budget_exceeded"
    HARD_RATIO_EXCEEDED = "hard_ratio_exceeded"
    REFERENCE_RATE_BELOW_TARGET = "reference_rate_below_target"


@dataclass(frozen=True)
class PairedConformanceReport:
    disposition: ConformanceDisposition
    gate_configuration: CgraGateConfiguration
    spatial_measurements: tuple[ExecutionMatrixMeasurement, ...]
    system_measurement: ExecutionMatrixMeasurement | None
    paired_result: PairedExecutionResult | None
    process_disposition: ProcessDisposition | None = None
    diagnostic: str = ""


@dataclass(frozen=True)
class CgraGateConfiguration:
    spatial_absolute_budget_nanoseconds: int
    configuration_sha256: str
    operator_gate_sha256: str
    profiles: tuple[Mapping[str, object], ...]

    @property
    def spatial_absolute_budget_seconds(self) -> float:
        return self.spatial_absolute_budget_nanoseconds / 1_000_000_000.0


@dataclass(frozen=True)
class CgraRepresentativeOperator:
    workload: str
    operator_id: str
    source: str
    compiler_flags: tuple[str, ...]


def load_cgra_representative_operators(
    path: Path = CGRA_OPERATOR_GATE,
) -> tuple[str, tuple[CgraRepresentativeOperator, ...]]:
    encoded = path.read_bytes()
    try:
        records = [json.loads(line) for line in encoded.decode("ascii").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("CGRA operator gate is not canonical ASCII JSONL") from error
    if not records or not isinstance(records[0], dict):
        raise ValueError("CGRA operator gate has no header")
    header = records[0]
    if header.get("schema_version") != 1 or not isinstance(header.get("counts"), dict):
        raise ValueError("CGRA operator gate has a foreign header")
    selected: dict[str, CgraRepresentativeOperator] = {}
    for record in records[1:]:
        if not isinstance(record, dict) or record.get("suite") != "loombench":
            continue
        vector = record.get("vector")
        selector = vector.get("selector") if isinstance(vector, dict) else None
        workload = selector.get("case") if isinstance(selector, dict) else None
        if workload not in CGRA_REPRESENTATIVE_WORKLOADS:
            continue
        producer = record.get("producer")
        sources = producer.get("sources") if isinstance(producer, dict) else None
        compiler_flags = record.get("compiler_flags")
        operator_id = record.get("operator_id")
        if (
            workload in selected
            or not isinstance(operator_id, str)
            or not operator_id
            or not operator_id.isascii()
            or not isinstance(sources, list)
            or len(sources) != 1
            or not isinstance(sources[0], str)
            or not sources[0].isascii()
            or Path(sources[0]).is_absolute()
            or ".." in Path(sources[0]).parts
            or not isinstance(compiler_flags, list)
            or any(
                not isinstance(flag, str) or not flag.isascii()
                for flag in compiler_flags
            )
            or record.get("entry_symbol") != "main"
            or record.get("profile") != "riscv64-portable-scalar"
        ):
            raise ValueError("CGRA representative operator row is invalid")
        selected[workload] = CgraRepresentativeOperator(
            workload,
            operator_id,
            sources[0],
            tuple(compiler_flags),
        )
    if set(selected) != set(CGRA_REPRESENTATIVE_WORKLOADS):
        raise ValueError("CGRA operator gate omits a representative workload")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest, tuple(selected[name] for name in CGRA_REPRESENTATIVE_WORKLOADS)


def _finite_number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{what} must be finite")
    return result


def _nonnegative_integer(value: object, what: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{what} must be {qualifier}")
    return value


def _validate_artifact_reference(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "schema_version",
        "artifact",
    }:
        raise ValueError(f"{what} is not an exact artifact reference")
    schema = value["schema"]
    version = value["schema_version"]
    artifact = value["artifact"]
    if not isinstance(schema, str) or not schema or not schema.isascii():
        raise ValueError(f"{what} has an invalid schema identity")
    if not isinstance(version, str):
        raise ValueError(f"{what} has an invalid schema version")
    version_parts = version.split(".")
    if len(version_parts) != 2 or any(
        not part or not part.isdecimal() for part in version_parts
    ):
        raise ValueError(f"{what} has an invalid schema version")
    if (
        not isinstance(artifact, str)
        or len(artifact) != 64
        or artifact != artifact.lower()
        or any(character not in "0123456789abcdef" for character in artifact)
    ):
        raise ValueError(f"{what} has a noncanonical artifact identity")
    return value


def _validate_cgra_profiles(
    profiles: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    _, representative_operators = load_cgra_representative_operators()
    operator_by_workload = {
        operator.workload: operator for operator in representative_operators
    }
    expected_profile_fields = {
        "schema",
        "workload",
        "operator_id",
        "qualification_limit_nanoseconds",
        "warmup_runs",
        "measurement_runs",
        "batch_peak_resident_bytes",
        "canonical_dataflow",
        "simulation_workload",
        "simulation_runtime_input",
        "resolved_config",
        "fabric",
        "tech_mapping",
        "initial_spatial_mapping",
        "spatial_mapping",
        "repaired_spatial_mapping",
        "parent_system_mapping",
        "transport_repair_constraint",
        "pre_repair_evidence",
        "warmup_evidence",
        "measurements",
    }
    expected_measurement_fields = {
        "active_wall_nanoseconds",
        "active_process_cpu_nanoseconds",
        "input_load_wall_nanoseconds",
        "input_load_process_cpu_nanoseconds",
        "engine_active_wall_nanoseconds",
        "engine_active_process_cpu_nanoseconds",
        "observation_projection_wall_nanoseconds",
        "observation_projection_process_cpu_nanoseconds",
        "artifact_publication_wall_nanoseconds",
        "artifact_publication_process_cpu_nanoseconds",
        "reference_cycles",
        "event_frame_count",
        "physical_request_count",
        "physical_grant_count",
        "physical_retirement_count",
        "physical_grant_wait_cycle_sum",
        "physical_grant_wait_cycle_max",
        "physical_grant_delayed_count",
        "evaluation_evidence",
    }
    if len(profiles) != len(CGRA_REPRESENTATIVE_WORKLOADS):
        raise ValueError("CGRA gate requires the complete representative suite")
    by_workload: dict[str, Mapping[str, object]] = {}
    dataflow_identities: set[str] = set()
    resolved_config_identities: set[str] = set()
    fabric_identities: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, Mapping) or set(profile) != expected_profile_fields:
            raise ValueError("CGRA profile has the wrong shape")
        if profile["schema"] != "loom.cgra_budget_profile.2":
            raise ValueError("CGRA profile has the wrong schema")
        workload = profile["workload"]
        if not isinstance(workload, str) or workload in by_workload:
            raise ValueError("CGRA profile workload is absent or duplicated")
        operator = operator_by_workload.get(workload)
        if operator is None or profile["operator_id"] != operator.operator_id:
            raise ValueError("CGRA profile is not bound to its operator-gate row")
        by_workload[workload] = profile
        qualification_limit = _nonnegative_integer(
            profile["qualification_limit_nanoseconds"],
            "CGRA qualification limit",
            positive=True,
        )
        if qualification_limit != CGRA_QUALIFICATION_LIMIT_NANOSECONDS:
            raise ValueError("CGRA profile used a foreign qualification limit")
        if (
            _nonnegative_integer(profile["warmup_runs"], "CGRA warmup count")
            != CGRA_QUALIFICATION_WARMUP_RUNS
            or _nonnegative_integer(
                profile["measurement_runs"], "CGRA measurement count"
            )
            != CGRA_QUALIFICATION_MEASUREMENT_RUNS
        ):
            raise ValueError("CGRA profile used a foreign sampling protocol")
        for field in (
            "canonical_dataflow",
            "simulation_workload",
            "simulation_runtime_input",
            "resolved_config",
            "fabric",
            "tech_mapping",
            "initial_spatial_mapping",
            "spatial_mapping",
            "warmup_evidence",
        ):
            _validate_artifact_reference(profile[field], f"CGRA profile {field}")
        resolved_config = profile["resolved_config"]
        assert isinstance(resolved_config, Mapping)
        if (
            resolved_config["schema"] != "loom.config.resolved"
            or resolved_config["schema_version"] != "11.0"
        ):
            raise ValueError("CGRA profile uses a foreign ResolvedConfig schema")
        resolved_config_identities.add(str(resolved_config["artifact"]))
        fabric = profile["fabric"]
        assert isinstance(fabric, Mapping)
        fabric_identities.add(str(fabric["artifact"]))
        repair_fields = (
            "repaired_spatial_mapping",
            "parent_system_mapping",
            "transport_repair_constraint",
            "pre_repair_evidence",
        )
        repair_values = tuple(profile[field] for field in repair_fields)
        if any(value is not None for value in repair_values):
            if any(value is None for value in repair_values):
                raise ValueError("CGRA transport repair provenance is fragmentary")
            for field, value in zip(repair_fields, repair_values):
                _validate_artifact_reference(value, f"CGRA profile {field}")
        dataflow = profile["canonical_dataflow"]
        assert isinstance(dataflow, Mapping)
        dataflow_identities.add(str(dataflow["artifact"]))
        _nonnegative_integer(
            profile["batch_peak_resident_bytes"],
            "CGRA workload-batch peak resident memory",
            positive=True,
        )
        measurements = profile["measurements"]
        if not isinstance(measurements, list) or len(measurements) != (
            CGRA_QUALIFICATION_MEASUREMENT_RUNS
        ):
            raise ValueError("CGRA profile has the wrong measurement count")
        deterministic_counts: tuple[int, ...] | None = None
        for measurement in measurements:
            if not isinstance(measurement, Mapping) or set(measurement) != (
                expected_measurement_fields
            ):
                raise ValueError("CGRA measurement has the wrong shape")
            _validate_artifact_reference(
                measurement["evaluation_evidence"],
                "CGRA measurement evaluation evidence",
            )
            active = _nonnegative_integer(
                measurement["active_wall_nanoseconds"],
                "CGRA active wall time",
                positive=True,
            )
            if active > CGRA_QUALIFICATION_LIMIT_NANOSECONDS:
                raise ValueError("CGRA active wall time exceeds qualification")
            input_load = _nonnegative_integer(
                measurement["input_load_wall_nanoseconds"],
                "CGRA input-load wall time",
            )
            engine_active = _nonnegative_integer(
                measurement["engine_active_wall_nanoseconds"],
                "CGRA engine-active wall time",
                positive=True,
            )
            observation_projection = _nonnegative_integer(
                measurement["observation_projection_wall_nanoseconds"],
                "CGRA observation-projection wall time",
            )
            _nonnegative_integer(
                measurement["artifact_publication_wall_nanoseconds"],
                "CGRA artifact-publication wall time",
            )
            active_cpu = measurement["active_process_cpu_nanoseconds"]
            component_cpu = (
                measurement["input_load_process_cpu_nanoseconds"],
                measurement["engine_active_process_cpu_nanoseconds"],
                measurement["observation_projection_process_cpu_nanoseconds"],
            )
            publication_cpu = measurement[
                "artifact_publication_process_cpu_nanoseconds"
            ]
            cpu_values = (active_cpu, *component_cpu, publication_cpu)
            if any(value is None for value in cpu_values):
                if any(value is not None for value in cpu_values):
                    raise ValueError("CGRA process CPU timing is fragmentary")
            else:
                typed_cpu = tuple(
                    _nonnegative_integer(value, "CGRA process CPU time")
                    for value in cpu_values
                )
                if typed_cpu[0] != sum(typed_cpu[1:4]):
                    raise ValueError(
                        "CGRA active process CPU time is not its component sum"
                    )
            if active != input_load + engine_active + observation_projection:
                raise ValueError("CGRA active wall time is not its component sum")
            cycles = _nonnegative_integer(
                measurement["reference_cycles"], "CGRA reference cycles", positive=True
            )
            target_nanoseconds = (
                cycles * 1_000_000_000 + REFERENCE_RATE_TARGET_CYCLES_PER_SECOND - 1
            ) // REFERENCE_RATE_TARGET_CYCLES_PER_SECOND
            if active > target_nanoseconds:
                raise ValueError(
                    "CGRA measurement is below the reference-cycle rate target"
                )
            event_frames = _nonnegative_integer(
                measurement["event_frame_count"],
                "CGRA event-frame count",
                positive=True,
            )
            requests = _nonnegative_integer(
                measurement["physical_request_count"],
                "CGRA physical-request count",
                positive=True,
            )
            grants = _nonnegative_integer(
                measurement["physical_grant_count"],
                "CGRA physical-grant count",
            )
            retirements = _nonnegative_integer(
                measurement["physical_retirement_count"],
                "CGRA physical-retirement count",
            )
            wait_sum = _nonnegative_integer(
                measurement["physical_grant_wait_cycle_sum"],
                "CGRA physical-grant wait sum",
            )
            wait_max = _nonnegative_integer(
                measurement["physical_grant_wait_cycle_max"],
                "CGRA physical-grant wait maximum",
            )
            delayed = _nonnegative_integer(
                measurement["physical_grant_delayed_count"],
                "CGRA delayed-grant count",
            )
            if requests != grants or grants != retirements:
                raise ValueError("CGRA physical lifecycle did not close")
            if delayed > grants or wait_max > wait_sum:
                raise ValueError("CGRA contention counters are inconsistent")
            counts = (
                cycles,
                event_frames,
                requests,
                grants,
                retirements,
                wait_sum,
                wait_max,
                delayed,
            )
            if deterministic_counts is None:
                deterministic_counts = counts
            elif counts != deterministic_counts:
                raise ValueError("CGRA deterministic counts changed across warm runs")
    if set(by_workload) != set(CGRA_REPRESENTATIVE_WORKLOADS):
        raise ValueError("CGRA gate names a foreign representative suite")
    if len(dataflow_identities) != len(CGRA_REPRESENTATIVE_WORKLOADS):
        raise ValueError("CGRA gate profiles do not name distinct Dataflow roots")
    if len(resolved_config_identities) != 1 or len(fabric_identities) != 1:
        raise ValueError("CGRA gate profiles do not share one exact target")
    return tuple(by_workload[name] for name in CGRA_REPRESENTATIVE_WORKLOADS)


def derive_cgra_spatial_budget_nanoseconds(
    profiles: Sequence[Mapping[str, object]],
) -> int:
    validated = _validate_cgra_profiles(profiles)
    budget = 0
    for profile in validated:
        measurements = profile["measurements"]
        assert isinstance(measurements, list)
        for measurement in measurements:
            assert isinstance(measurement, Mapping)
            cycles = int(measurement["reference_cycles"])
            target_nanoseconds = (
                cycles * 1_000_000_000 + REFERENCE_RATE_TARGET_CYCLES_PER_SECOND - 1
            ) // REFERENCE_RATE_TARGET_CYCLES_PER_SECOND
            budget = max(budget, target_nanoseconds)
    if budget <= 0 or budget > CGRA_QUALIFICATION_LIMIT_NANOSECONDS:
        raise ValueError("derived CGRA budget exceeds the qualification limit")
    return budget


def load_cgra_gate_configuration(
    path: Path = CGRA_GATE_CONFIGURATION,
) -> CgraGateConfiguration:
    encoded = path.read_bytes()
    try:
        root = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "CGRA gate configuration is not canonical ASCII JSON"
        ) from error
    expected_fields = {
        "schema",
        "policy",
        "operator_gate",
        "spatial_absolute_budget_nanoseconds",
        "profiles",
    }
    if not isinstance(root, dict) or set(root) != expected_fields:
        raise ValueError("CGRA gate configuration has the wrong shape")
    if root["schema"] != "loom.cgra_simulation_gate.2":
        raise ValueError("CGRA gate configuration has the wrong schema")
    policy = root["policy"]
    expected_policy = {
        "qualification_limit_nanoseconds": CGRA_QUALIFICATION_LIMIT_NANOSECONDS,
        "warmup_runs": CGRA_QUALIFICATION_WARMUP_RUNS,
        "measurement_runs": CGRA_QUALIFICATION_MEASUREMENT_RUNS,
        "reference_rate_target_cycles_per_second": (
            REFERENCE_RATE_TARGET_CYCLES_PER_SECOND
        ),
    }
    if policy != expected_policy:
        raise ValueError("CGRA gate configuration has a foreign policy")
    operator_gate = root["operator_gate"]
    current_operator_gate_sha256, _ = load_cgra_representative_operators()
    if operator_gate != {
        "path": CGRA_OPERATOR_GATE_RELATIVE_PATH,
        "sha256": current_operator_gate_sha256,
    }:
        raise ValueError("CGRA gate configuration names a foreign operator gate")
    profiles = root["profiles"]
    if not isinstance(profiles, list):
        raise ValueError("CGRA gate profiles are not a list")
    validated = _validate_cgra_profiles(profiles)
    derived = derive_cgra_spatial_budget_nanoseconds(validated)
    published = _nonnegative_integer(
        root["spatial_absolute_budget_nanoseconds"],
        "published CGRA budget",
        positive=True,
    )
    if published != derived:
        raise ValueError("published CGRA budget does not match aggregate evidence")
    return CgraGateConfiguration(
        published,
        hashlib.sha256(encoded).hexdigest(),
        current_operator_gate_sha256,
        validated,
    )


def paired_system_budget(
    warmed_spatial_active_seconds: Sequence[float],
    spatial_absolute_budget_seconds: float,
    *,
    reference_floor_seconds: float = SPATIAL_REFERENCE_FLOOR_SECONDS,
    system_multiplier: float = SYSTEM_BUDGET_MULTIPLIER,
) -> PairedSystemBudget:
    if not warmed_spatial_active_seconds:
        raise ValueError("at least one warmed Spatial timing sample is required")
    for sample in warmed_spatial_active_seconds:
        _require_finite_positive(sample, "warmed Spatial timing sample")
    _require_finite_positive(spatial_absolute_budget_seconds, "Spatial budget")
    _require_finite_positive(reference_floor_seconds, "Spatial reference floor")
    _require_finite_positive(system_multiplier, "System budget multiplier")
    reference = max(
        float(statistics.median(warmed_spatial_active_seconds)),
        reference_floor_seconds,
    )
    system_budget = min(
        system_multiplier * reference,
        system_multiplier * spatial_absolute_budget_seconds,
    )
    return PairedSystemBudget(
        spatial_reference_seconds=reference,
        spatial_absolute_budget_seconds=spatial_absolute_budget_seconds,
        system_budget_seconds=system_budget,
        hard_failure_seconds=HARD_FAILURE_RATIO * reference,
    )


def evaluate_paired_execution(
    budget: PairedSystemBudget,
    system_timing: ActiveExecutionTiming,
) -> PairedExecutionResult:
    _require_finite_positive(budget.spatial_reference_seconds, "Spatial reference")
    _require_finite_positive(budget.system_budget_seconds, "System budget")
    _require_finite_positive(budget.hard_failure_seconds, "hard failure time")
    ratio = system_timing.active_wall_seconds / budget.spatial_reference_seconds
    rate = system_timing.reference_cycles / system_timing.active_wall_seconds
    return PairedExecutionResult(
        spatial_reference_seconds=budget.spatial_reference_seconds,
        spatial_absolute_budget_seconds=budget.spatial_absolute_budget_seconds,
        system_to_spatial_ratio=ratio,
        system_budget_seconds=budget.system_budget_seconds,
        within_system_budget=(
            system_timing.active_wall_seconds <= budget.system_budget_seconds
        ),
        hard_ratio_failure=(
            system_timing.active_wall_seconds >= budget.hard_failure_seconds
        ),
        reference_cycles_per_second=rate,
        meets_reference_rate_target=(rate >= REFERENCE_RATE_TARGET_HZ),
        system_timing=system_timing,
    )


def execute_process(
    command: Sequence[str],
    timeout_seconds: float,
    *,
    termination_grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
    environment: Mapping[str, str] | None = None,
    working_directory: Path | None = None,
) -> ProcessExecution:
    if not command or any(not argument for argument in command):
        raise ValueError("process command must contain nonempty arguments")
    _require_finite_positive(timeout_seconds, "process timeout")
    _require_finite_positive(termination_grace_seconds, "termination grace")
    normalized = tuple(str(argument) for argument in command)
    started = time.monotonic()
    process = subprocess.Popen(
        normalized,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=dict(environment) if environment is not None else None,
        cwd=str(working_directory) if working_directory is not None else None,
    )

    def captured_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return value

    def process_group_exists() -> bool:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        return True

    def wait_for_process_group_exit() -> bool:
        deadline = time.monotonic() + termination_grace_seconds
        while process_group_exists() and time.monotonic() < deadline:
            time.sleep(min(0.01, termination_grace_seconds / 10.0))
        return not process_group_exists()

    def terminate_process_group() -> bool:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        if wait_for_process_group_exit():
            return True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        return wait_for_process_group_exit()

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as timeout:
        stdout = captured_text(timeout.stdout)
        stderr = captured_text(timeout.stderr)
        terminated = terminate_process_group()
        try:
            final_stdout, final_stderr = process.communicate(
                timeout=termination_grace_seconds
            )
            stdout = final_stdout or stdout
            stderr = final_stderr or stderr
        except subprocess.TimeoutExpired as cleanup_timeout:
            stdout = captured_text(cleanup_timeout.stdout) or stdout
            stderr = captured_text(cleanup_timeout.stderr) or stderr
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            try:
                process.wait(timeout=termination_grace_seconds)
            except subprocess.TimeoutExpired:
                terminated = False
        return ProcessExecution(
            disposition=(
                ProcessDisposition.TIMED_OUT
                if terminated
                else ProcessDisposition.CLEANUP_FAILED
            ),
            command=normalized,
            elapsed_seconds=time.monotonic() - started,
            return_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            process_group_terminated=terminated,
        )
    if process_group_exists():
        terminated = terminate_process_group()
        return ProcessExecution(
            disposition=ProcessDisposition.CLEANUP_FAILED,
            command=normalized,
            elapsed_seconds=time.monotonic() - started,
            return_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            process_group_terminated=terminated,
        )
    return ProcessExecution(
        disposition=(
            ProcessDisposition.COMPLETED
            if process.returncode == 0
            else ProcessDisposition.NONZERO_EXIT
        ),
        command=normalized,
        elapsed_seconds=time.monotonic() - started,
        return_code=process.returncode,
        stdout=stdout,
        stderr=stderr,
        process_group_terminated=False,
    )


_STATISTICS_FIELDS = {
    "cell",
    "paired_work_fingerprint",
    "deterministic_work",
    "accelerator_reference_cycles",
    "gem5_ticks",
    "setup_wall_us",
    "preparation_wall_us",
    "gem5_configuration_wall_us",
    "provider_wall_us",
    "active_wall_us",
    "provider_cpu_us",
    "gem5_active_process_cpu_us",
    "gem5_observation_process_cpu_us",
    "engine_process_cpu_us",
    "cgra_input_load_wall_us",
    "cgra_input_load_process_cpu_us",
    "cgra_engine_active_wall_us",
    "cgra_engine_active_process_cpu_us",
    "cgra_observation_projection_wall_us",
    "cgra_observation_projection_process_cpu_us",
    "cgra_artifact_publication_wall_us",
    "cgra_artifact_publication_process_cpu_us",
    "bridge_callback_cpu_us",
    "bridge_wait_wall_us",
    "bridge_message_count",
    "accelerator_invocation_count",
    "cgra_event_frame_count",
    "gem5_observation_wall_us",
    "observation_wall_us",
    "peak_rss_kib",
}


def _parse_statistics_row(row: str, expected_cell: str) -> ExecutionMatrixMeasurement:
    fields: dict[str, str] = {}
    items = row.split()
    if not items or items[0] != "execution-matrix":
        raise ValueError("execution matrix statistics row has no record tag")
    for item in items[1:]:
        key, separator, value = item.partition("=")
        if not separator or not key or not value or key in fields:
            raise ValueError("execution matrix statistics row is malformed")
        fields[key] = value
    if set(fields) != _STATISTICS_FIELDS or fields["cell"] != expected_cell:
        raise ValueError("execution matrix statistics row has the wrong shape")
    if expected_cell not in {"paired-spatial-cgra", "paired-system-cgra"}:
        raise ValueError("execution matrix cell is outside paired CGRA conformance")
    fingerprint = fields["paired_work_fingerprint"]
    if (
        len(fingerprint) != 64
        or fingerprint != fingerprint.lower()
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("paired work fingerprint is not canonical")
    integral_fields = _STATISTICS_FIELDS - {
        "cell",
        "gem5_ticks",
        "paired_work_fingerprint",
    }
    maximum_u64 = (1 << 64) - 1

    def parse_u64(value: str) -> int:
        if (
            not value
            or any(character not in "0123456789" for character in value)
            or (len(value) != 1 and value.startswith("0"))
        ):
            raise ValueError("execution matrix statistics are not canonical integers")
        parsed = int(value)
        if parsed > maximum_u64:
            raise ValueError("execution matrix statistics escape the unsigned domain")
        return parsed

    values = {field: parse_u64(fields[field]) for field in integral_fields}
    gem5_ticks = (
        None
        if fields["gem5_ticks"] == "not_applicable"
        else parse_u64(fields["gem5_ticks"])
    )
    if values["active_wall_us"] == 0 or values["provider_wall_us"] == 0:
        raise ValueError("execution matrix active timing must be positive")
    if expected_cell == "paired-spatial-cgra":
        if gem5_ticks is not None:
            raise ValueError("Spatial CGRA statistics contain gem5 ticks")
        if (
            values["bridge_message_count"] != 0
            or values["accelerator_invocation_count"] != 0
            or values["gem5_active_process_cpu_us"] != 0
            or values["gem5_observation_process_cpu_us"] != 0
        ):
            raise ValueError("Spatial CGRA statistics contain System activity")
    elif (
        gem5_ticks is None
        or gem5_ticks == 0
        or values["bridge_message_count"] == 0
        or values["accelerator_invocation_count"] == 0
        or values["cgra_event_frame_count"] == 0
    ):
        raise ValueError("System CGRA statistics omit required activity")
    if (
        values["deterministic_work"] == 0
        or values["accelerator_reference_cycles"] == 0
        or values["cgra_event_frame_count"] == 0
    ):
        raise ValueError("paired CGRA statistics contain no retired work")
    micros = 1_000_000.0
    timing = ActiveExecutionTiming(
        active_wall_seconds=values["active_wall_us"] / micros,
        reference_cycles=values["accelerator_reference_cycles"],
        cgra_engine_active_wall_seconds=(values["cgra_engine_active_wall_us"] / micros),
        cgra_engine_active_process_cpu_seconds=(
            values["cgra_engine_active_process_cpu_us"] / micros
        ),
        gem5_active_process_cpu_seconds=(values["gem5_active_process_cpu_us"] / micros),
        gem5_observation_process_cpu_seconds=(
            values["gem5_observation_process_cpu_us"] / micros
        ),
        engine_process_cpu_seconds=values["engine_process_cpu_us"] / micros,
        cgra_input_load_wall_seconds=values["cgra_input_load_wall_us"] / micros,
        cgra_input_load_process_cpu_seconds=(
            values["cgra_input_load_process_cpu_us"] / micros
        ),
        bridge_callback_cpu_seconds=values["bridge_callback_cpu_us"] / micros,
        bridge_wait_wall_seconds=values["bridge_wait_wall_us"] / micros,
        setup_wall_seconds=values["setup_wall_us"] / micros,
        preparation_wall_seconds=values["preparation_wall_us"] / micros,
        gem5_configuration_wall_seconds=(values["gem5_configuration_wall_us"] / micros),
        provider_wall_seconds=values["provider_wall_us"] / micros,
        provider_cpu_seconds=values["provider_cpu_us"] / micros,
        gem5_observation_wall_seconds=(values["gem5_observation_wall_us"] / micros),
        observation_wall_seconds=values["observation_wall_us"] / micros,
        cgra_observation_projection_wall_seconds=(
            values["cgra_observation_projection_wall_us"] / micros
        ),
        cgra_observation_projection_process_cpu_seconds=(
            values["cgra_observation_projection_process_cpu_us"] / micros
        ),
        cgra_artifact_publication_wall_seconds=(
            values["cgra_artifact_publication_wall_us"] / micros
        ),
        cgra_artifact_publication_process_cpu_seconds=(
            values["cgra_artifact_publication_process_cpu_us"] / micros
        ),
        bridge_message_count=values["bridge_message_count"],
        accelerator_invocation_count=values["accelerator_invocation_count"],
        cgra_event_frame_count=values["cgra_event_frame_count"],
        peak_resident_bytes=values["peak_rss_kib"] * 1024,
    )
    return ExecutionMatrixMeasurement(
        cell=expected_cell,
        paired_work_fingerprint=fingerprint,
        deterministic_work=values["deterministic_work"],
        accelerator_reference_cycles=values["accelerator_reference_cycles"],
        gem5_ticks=gem5_ticks,
        timing=timing,
    )


def parse_execution_matrix_measurements(
    output: str, expected_cell: str, expected_count: int | None = None
) -> tuple[ExecutionMatrixMeasurement, ...]:
    prefix = "execution-matrix cell="
    rows = [line for line in output.splitlines() if line.startswith(prefix)]
    if not rows or (expected_count is not None and len(rows) != expected_count):
        raise ValueError("execution matrix output has the wrong statistics count")
    return tuple(_parse_statistics_row(row, expected_cell) for row in rows)


def parse_execution_matrix_measurement(
    output: str, expected_cell: str
) -> ExecutionMatrixMeasurement:
    return parse_execution_matrix_measurements(output, expected_cell, 1)[0]


def _failed_report(
    disposition: ConformanceDisposition,
    gate_configuration: CgraGateConfiguration,
    spatial: Sequence[ExecutionMatrixMeasurement],
    process: ProcessExecution,
) -> PairedConformanceReport:
    return PairedConformanceReport(
        disposition=disposition,
        gate_configuration=gate_configuration,
        spatial_measurements=tuple(spatial),
        system_measurement=None,
        paired_result=None,
        process_disposition=process.disposition,
        diagnostic=process.stderr.strip() or process.stdout.strip(),
    )


def run_paired_execution_matrix(
    execution_matrix_runner: Path,
    gem5_readiness: Path,
    *,
    process_timeout_seconds: float = DEFAULT_PROCESS_TIMEOUT_SECONDS,
    spatial_warmup_runs: int = SPATIAL_WARMUP_RUNS,
    spatial_measurement_runs: int = SPATIAL_MEASUREMENT_RUNS,
) -> PairedConformanceReport:
    if spatial_warmup_runs < 1 or spatial_measurement_runs < 1:
        raise ValueError("paired conformance requires warmup and measured runs")
    gate_configuration = load_cgra_gate_configuration()
    spatial_absolute_budget_seconds = gate_configuration.spatial_absolute_budget_seconds
    runner = str(execution_matrix_runner.resolve(strict=True))
    readiness = str(gem5_readiness.resolve(strict=True))
    scratch_root = Path(__file__).resolve().parents[1] / "temp"
    scratch_root.mkdir(exist_ok=True)
    process_environment = dict(os.environ)
    process_environment["TMPDIR"] = str(scratch_root)
    spatial_process = execute_process(
        (
            runner,
            "paired-spatial-cgra-batch",
            str(spatial_warmup_runs),
            str(spatial_measurement_runs),
        ),
        process_timeout_seconds,
        environment=process_environment,
    )
    if spatial_process.disposition is not ProcessDisposition.COMPLETED:
        if spatial_process.disposition is ProcessDisposition.CLEANUP_FAILED:
            disposition = ConformanceDisposition.PROCESS_CLEANUP_FAILED
        elif spatial_process.disposition is ProcessDisposition.TIMED_OUT:
            disposition = ConformanceDisposition.SPATIAL_TIMED_OUT
        else:
            disposition = ConformanceDisposition.SPATIAL_EXECUTION_FAILED
        return _failed_report(disposition, gate_configuration, (), spatial_process)
    try:
        spatial_measurements = parse_execution_matrix_measurements(
            spatial_process.stdout,
            "paired-spatial-cgra",
            spatial_measurement_runs,
        )
    except ValueError as error:
        return PairedConformanceReport(
            disposition=ConformanceDisposition.SPATIAL_EXECUTION_FAILED,
            gate_configuration=gate_configuration,
            spatial_measurements=(),
            system_measurement=None,
            paired_result=None,
            process_disposition=spatial_process.disposition,
            diagnostic=str(error),
        )
    if any(
        measurement.timing.active_wall_seconds > spatial_absolute_budget_seconds
        for measurement in spatial_measurements
    ):
        return PairedConformanceReport(
            disposition=ConformanceDisposition.SPATIAL_BUDGET_EXCEEDED,
            gate_configuration=gate_configuration,
            spatial_measurements=spatial_measurements,
            system_measurement=None,
            paired_result=None,
            process_disposition=spatial_process.disposition,
            diagnostic="a warmed Spatial measurement exceeded its absolute budget",
        )
    system_process = execute_process(
        (runner, "paired-system-cgra", readiness),
        process_timeout_seconds,
        environment=process_environment,
    )
    if system_process.disposition is not ProcessDisposition.COMPLETED:
        if system_process.disposition is ProcessDisposition.CLEANUP_FAILED:
            disposition = ConformanceDisposition.PROCESS_CLEANUP_FAILED
        elif system_process.disposition is ProcessDisposition.TIMED_OUT:
            disposition = ConformanceDisposition.SYSTEM_TIMED_OUT
        else:
            disposition = ConformanceDisposition.SYSTEM_EXECUTION_FAILED
        return _failed_report(
            disposition, gate_configuration, spatial_measurements, system_process
        )
    try:
        system_measurement = parse_execution_matrix_measurement(
            system_process.stdout, "paired-system-cgra"
        )
    except ValueError as error:
        return PairedConformanceReport(
            disposition=ConformanceDisposition.SYSTEM_EXECUTION_FAILED,
            gate_configuration=gate_configuration,
            spatial_measurements=spatial_measurements,
            system_measurement=None,
            paired_result=None,
            process_disposition=system_process.disposition,
            diagnostic=str(error),
        )
    expected = spatial_measurements[0]
    if (
        any(
            sample.paired_work_fingerprint != expected.paired_work_fingerprint
            or sample.accelerator_reference_cycles
            != expected.accelerator_reference_cycles
            or sample.timing.cgra_event_frame_count
            != expected.timing.cgra_event_frame_count
            for sample in spatial_measurements[1:]
        )
        or system_measurement.paired_work_fingerprint
        != expected.paired_work_fingerprint
        or system_measurement.accelerator_reference_cycles
        != expected.accelerator_reference_cycles
        or system_measurement.timing.cgra_event_frame_count
        != expected.timing.cgra_event_frame_count
    ):
        return PairedConformanceReport(
            disposition=ConformanceDisposition.PAIRED_WORK_MISMATCH,
            gate_configuration=gate_configuration,
            spatial_measurements=spatial_measurements,
            system_measurement=system_measurement,
            paired_result=None,
            process_disposition=system_process.disposition,
            diagnostic="Spatial and System runs differ in identity or retired work",
        )
    budget = paired_system_budget(
        [sample.timing.active_wall_seconds for sample in spatial_measurements],
        spatial_absolute_budget_seconds,
    )
    paired = evaluate_paired_execution(budget, system_measurement.timing)
    disposition = ConformanceDisposition.PASSED
    if paired.hard_ratio_failure:
        disposition = ConformanceDisposition.HARD_RATIO_EXCEEDED
    elif not paired.within_system_budget:
        disposition = ConformanceDisposition.SYSTEM_BUDGET_EXCEEDED
    elif not paired.meets_reference_rate_target:
        disposition = ConformanceDisposition.REFERENCE_RATE_BELOW_TARGET
    return PairedConformanceReport(
        disposition=disposition,
        gate_configuration=gate_configuration,
        spatial_measurements=spatial_measurements,
        system_measurement=system_measurement,
        paired_result=paired,
        process_disposition=system_process.disposition,
    )


def outer_worker_limit(
    *,
    memory_derived_limit: int,
    cpu_count: int | None = None,
    reserved_cpus: int = RESERVED_DEVELOPMENT_CPUS,
    maximum_workers: int = MAX_OUTER_WORKERS,
) -> int:
    if memory_derived_limit < 1:
        raise ValueError("memory-derived worker limit must be positive")
    if reserved_cpus < 0:
        raise ValueError("reserved CPU count must be nonnegative")
    if maximum_workers < 1:
        raise ValueError("maximum worker count must be positive")
    available_cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    if available_cpus < 1:
        raise ValueError("CPU count must be positive")
    return min(
        max(1, available_cpus - reserved_cpus),
        memory_derived_limit,
        maximum_workers,
    )


def _timing_json(timing: ActiveExecutionTiming) -> dict[str, object]:
    return {
        "active_wall_seconds": timing.active_wall_seconds,
        "reference_cycles": timing.reference_cycles,
        "cgra_engine_active_wall_seconds": (timing.cgra_engine_active_wall_seconds),
        "cgra_engine_active_process_cpu_seconds": (
            timing.cgra_engine_active_process_cpu_seconds
        ),
        "gem5_active_process_cpu_seconds": timing.gem5_active_process_cpu_seconds,
        "gem5_observation_process_cpu_seconds": (
            timing.gem5_observation_process_cpu_seconds
        ),
        "engine_process_cpu_seconds": timing.engine_process_cpu_seconds,
        "cgra_input_load_wall_seconds": timing.cgra_input_load_wall_seconds,
        "cgra_input_load_process_cpu_seconds": (
            timing.cgra_input_load_process_cpu_seconds
        ),
        "bridge_callback_cpu_seconds": timing.bridge_callback_cpu_seconds,
        "bridge_wait_wall_seconds": timing.bridge_wait_wall_seconds,
        "setup_wall_seconds": timing.setup_wall_seconds,
        "preparation_wall_seconds": timing.preparation_wall_seconds,
        "gem5_configuration_wall_seconds": timing.gem5_configuration_wall_seconds,
        "provider_wall_seconds": timing.provider_wall_seconds,
        "provider_process_cpu_seconds": timing.provider_cpu_seconds,
        "gem5_observation_wall_seconds": timing.gem5_observation_wall_seconds,
        "observation_wall_seconds": timing.observation_wall_seconds,
        "cgra_observation_projection_wall_seconds": (
            timing.cgra_observation_projection_wall_seconds
        ),
        "cgra_observation_projection_process_cpu_seconds": (
            timing.cgra_observation_projection_process_cpu_seconds
        ),
        "cgra_artifact_publication_wall_seconds": (
            timing.cgra_artifact_publication_wall_seconds
        ),
        "cgra_artifact_publication_process_cpu_seconds": (
            timing.cgra_artifact_publication_process_cpu_seconds
        ),
        "bridge_message_count": timing.bridge_message_count,
        "accelerator_invocation_count": timing.accelerator_invocation_count,
        "cgra_event_frame_count": timing.cgra_event_frame_count,
        "peak_resident_bytes": timing.peak_resident_bytes,
    }


def _report_json(report: PairedConformanceReport) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": "loom.paired_simulation_conformance.2",
        "cgra_gate": {
            "path": str(
                CGRA_GATE_CONFIGURATION.relative_to(Path(__file__).resolve().parents[1])
            ),
            "sha256": report.gate_configuration.configuration_sha256,
            "operator_gate_sha256": (report.gate_configuration.operator_gate_sha256),
            "spatial_absolute_budget_nanoseconds": (
                report.gate_configuration.spatial_absolute_budget_nanoseconds
            ),
        },
        "disposition": report.disposition.value,
        "process_disposition": (
            report.process_disposition.value
            if report.process_disposition is not None
            else None
        ),
        "diagnostic": report.diagnostic,
        "spatial": [
            _timing_json(sample.timing) for sample in report.spatial_measurements
        ],
        "system": None,
        "paired": None,
    }
    if report.system_measurement is not None:
        result["system"] = {
            **_timing_json(report.system_measurement.timing),
            "gem5_ticks": report.system_measurement.gem5_ticks,
        }
    if report.paired_result is not None:
        result["paired"] = {
            "spatial_reference_seconds": (
                report.paired_result.spatial_reference_seconds
            ),
            "spatial_absolute_budget_seconds": (
                report.paired_result.spatial_absolute_budget_seconds
            ),
            "system_budget_seconds": report.paired_result.system_budget_seconds,
            "system_to_spatial_ratio": (report.paired_result.system_to_spatial_ratio),
            "reference_cycles_per_second": (
                report.paired_result.reference_cycles_per_second
            ),
            "within_system_budget": report.paired_result.within_system_budget,
            "hard_ratio_failure": report.paired_result.hard_ratio_failure,
            "meets_reference_rate_target": (
                report.paired_result.meets_reference_rate_target
            ),
        }
    return result


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--execution-matrix-runner", required=True, type=Path)
    parser.add_argument("--gem5-readiness", required=True, type=Path)
    parser.add_argument(
        "--process-timeout-seconds",
        type=float,
        default=DEFAULT_PROCESS_TIMEOUT_SECONDS,
    )
    parser.add_argument("--spatial-warmup-runs", type=int, default=SPATIAL_WARMUP_RUNS)
    parser.add_argument(
        "--spatial-measurement-runs",
        type=int,
        default=SPATIAL_MEASUREMENT_RUNS,
    )
    arguments = parser.parse_args()
    report = run_paired_execution_matrix(
        arguments.execution_matrix_runner,
        arguments.gem5_readiness,
        process_timeout_seconds=arguments.process_timeout_seconds,
        spatial_warmup_runs=arguments.spatial_warmup_runs,
        spatial_measurement_runs=arguments.spatial_measurement_runs,
    )
    print(json.dumps(_report_json(report), sort_keys=True))
    return 0 if report.disposition is ConformanceDisposition.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
