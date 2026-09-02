"""Shared conformance policy for paired Spatial and System simulations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.timeout_budgets import Tier, seconds as timeout_seconds  # noqa: E402


SPATIAL_REFERENCE_FLOOR_SECONDS = 0.1
SYSTEM_BUDGET_MULTIPLIER = 3.0
HARD_FAILURE_RATIO = 10.0
REFERENCE_RATE_TARGET_CYCLES_PER_SECOND = 100_000
REFERENCE_RATE_TARGET_HZ = float(REFERENCE_RATE_TARGET_CYCLES_PER_SECOND)
DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS = float(timeout_seconds(Tier.FAST))
CGRA_SPATIAL_BOOTSTRAP_BUDGET_SECONDS = float(timeout_seconds(Tier.MEDIUM))
RESERVED_DEVELOPMENT_CPUS = 4
MAX_OUTER_WORKERS = 120
CGRA_QUALIFICATION_LIMIT_NANOSECONDS = 45_000_000_000
CGRA_QUALIFICATION_WARMUP_RUNS = 1
CGRA_QUALIFICATION_MEASUREMENT_RUNS = 3
MAX_CANDIDATE_PROOF_KIND = (1 << 32) - 1
CGRA_GATE_CONFIGURATION = (
    Path(__file__).resolve().parent / "data" / "cgra-simulation-gate-v1.json"
)
CGRA_OPERATOR_GATE_RELATIVE_PATH = "test/data/corpus-operator-gate-v1.jsonl"
CGRA_OPERATOR_GATE = ROOT / CGRA_OPERATOR_GATE_RELATIVE_PATH
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
    engine_cpu_seconds: float = 0.0
    bridge_cpu_seconds: float = 0.0
    host_cpu_seconds: float = 0.0
    observation_cpu_seconds: float = 0.0
    event_count: int = 0
    activation_count: int = 0
    peak_resident_bytes: int = 0

    def __post_init__(self) -> None:
        _require_finite_positive(self.active_wall_seconds, "active wall time")
        if self.reference_cycles < 0:
            raise ValueError("reference-cycle count must be nonnegative")
        for value, what in (
            (self.engine_cpu_seconds, "engine CPU time"),
            (self.bridge_cpu_seconds, "Bridge CPU time"),
            (self.host_cpu_seconds, "host CPU time"),
            (self.observation_cpu_seconds, "observation CPU time"),
        ):
            _require_nonnegative(value, what)
        for value, what in (
            (self.event_count, "event count"),
            (self.activation_count, "activation count"),
            (self.peak_resident_bytes, "peak resident bytes"),
        ):
            if value < 0:
                raise ValueError(f"{what} must be nonnegative")


@dataclass(frozen=True)
class PairedSystemBudget:
    spatial_reference_seconds: float
    system_budget_seconds: float
    hard_failure_seconds: float


@dataclass(frozen=True)
class PairedExecutionResult:
    spatial_reference_seconds: float
    system_active_wall_seconds: float
    system_to_spatial_ratio: float
    system_budget_seconds: float
    within_system_budget: bool
    hard_ratio_failure: bool
    reference_cycles: int
    reference_cycles_per_second: float
    reference_rate_basis_seconds: float
    meets_reference_rate_target: bool
    engine_cpu_seconds: float
    bridge_cpu_seconds: float
    host_cpu_seconds: float
    observation_cpu_seconds: float
    event_count: int
    activation_count: int
    peak_resident_bytes: int


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
    _require_finite_positive(spatial_absolute_budget_seconds, "Spatial absolute budget")
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
        system_budget_seconds=system_budget,
        hard_failure_seconds=HARD_FAILURE_RATIO * reference,
    )


def evaluate_paired_execution(
    budget: PairedSystemBudget,
    system_timing: ActiveExecutionTiming,
) -> PairedExecutionResult:
    _require_finite_positive(budget.spatial_reference_seconds, "Spatial reference time")
    _require_finite_positive(budget.system_budget_seconds, "System budget")
    _require_finite_positive(budget.hard_failure_seconds, "hard failure time")

    ratio = system_timing.active_wall_seconds / budget.spatial_reference_seconds
    # The rate gate qualifies the simulator's throughput, not the host
    # scheduler: the simulator process's CPU time (the engine slot for direct
    # engine measurements, the host slot for the gem5 child) is far less
    # load-sensitive than active wall time, so a loaded parallel suite must
    # not turn a healthy engine into a rate failure. Wall time remains the
    # basis for the paired budget and hard-ratio contracts above.
    rate_basis_seconds = system_timing.active_wall_seconds
    if system_timing.engine_cpu_seconds > 0.0:
        rate_basis_seconds = system_timing.engine_cpu_seconds
    elif system_timing.host_cpu_seconds > 0.0:
        rate_basis_seconds = system_timing.host_cpu_seconds
    rate = system_timing.reference_cycles / rate_basis_seconds
    return PairedExecutionResult(
        spatial_reference_seconds=budget.spatial_reference_seconds,
        system_active_wall_seconds=system_timing.active_wall_seconds,
        system_to_spatial_ratio=ratio,
        system_budget_seconds=budget.system_budget_seconds,
        within_system_budget=(
            system_timing.active_wall_seconds <= budget.system_budget_seconds
        ),
        hard_ratio_failure=(ratio >= HARD_FAILURE_RATIO),
        reference_cycles=system_timing.reference_cycles,
        reference_cycles_per_second=rate,
        reference_rate_basis_seconds=rate_basis_seconds,
        meets_reference_rate_target=(rate >= REFERENCE_RATE_TARGET_HZ),
        engine_cpu_seconds=system_timing.engine_cpu_seconds,
        bridge_cpu_seconds=system_timing.bridge_cpu_seconds,
        host_cpu_seconds=system_timing.host_cpu_seconds,
        observation_cpu_seconds=system_timing.observation_cpu_seconds,
        event_count=system_timing.event_count,
        activation_count=system_timing.activation_count,
        peak_resident_bytes=system_timing.peak_resident_bytes,
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
    available_workers = max(1, available_cpus - reserved_cpus)
    return min(available_workers, memory_derived_limit, maximum_workers)


class ProcessDisposition(str, Enum):
    COMPLETED = "completed"
    LAUNCH_FAILED = "launch_failed"
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
    attempt: str
    invocation: str
    work_fingerprint: str
    config_fingerprint: str
    gem5_ticks: int | None
    setup_wall_seconds: float
    active_process_cpu_seconds: float
    measurement_source: str
    rss_scope: str
    timing: ActiveExecutionTiming


class MeasurementDisposition(str, Enum):
    MEASURED = "measured"
    SPATIAL_EXECUTION_FAILED = "spatial_execution_failed"
    SPATIAL_TIMED_OUT = "spatial_timed_out"
    SPATIAL_BOOTSTRAP_BUDGET_EXCEEDED = "spatial_bootstrap_budget_exceeded"
    SYSTEM_EXECUTION_FAILED = "system_execution_failed"
    SYSTEM_TIMED_OUT = "system_timed_out"
    PROCESS_CLEANUP_FAILED = "process_cleanup_failed"
    PAIRED_WORK_MISMATCH = "paired_work_mismatch"
    SYSTEM_BOOTSTRAP_BUDGET_EXCEEDED = "system_bootstrap_budget_exceeded"
    HARD_RATIO_EXCEEDED = "hard_ratio_exceeded"
    REFERENCE_RATE_BELOW_TARGET = "reference_rate_below_target"


@dataclass(frozen=True)
class PairedMeasurementReport:
    disposition: MeasurementDisposition
    spatial_measurements: tuple[ExecutionMatrixMeasurement, ...]
    system_measurement: ExecutionMatrixMeasurement | None
    paired_result: PairedExecutionResult | None
    spatial_process: ProcessExecution | None
    system_process: ProcessExecution | None
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
    protocol_symbol: str
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
        protocol = record.get("protocol")
        protocol_symbol = (
            protocol[0].get("symbol")
            if isinstance(protocol, list)
            and len(protocol) == 1
            and isinstance(protocol[0], dict)
            else None
        )
        if (
            workload in selected
            or not isinstance(operator_id, str)
            or not operator_id
            or not operator_id.isascii()
            or not isinstance(protocol_symbol, str)
            or not protocol_symbol
            or not protocol_symbol.isascii()
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
            protocol_symbol,
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


def _owned_schema(relative_path: str, name: str) -> tuple[str, str]:
    """Derive an artifact schema version from its C++ semantic owner.

    The named descriptor constant is the only version authority; this
    parser is a mechanical derivation and fails loudly when the owner
    moves or the descriptor shape changes, instead of drifting on a
    hand-copied constant.
    """
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    anchor = source.find(f'"{name}"')
    if anchor < 0:
        raise RuntimeError(
            f"schema owner {relative_path} does not define {name}"
        )
    version = re.search(r"\{\s*(\d+)\s*,\s*(\d+)\s*\}", source[anchor:])
    if version is None:
        raise RuntimeError(
            f"schema owner {relative_path} has no version for {name}"
        )
    return (name, f"{version.group(1)}.{version.group(2)}")


_CANONICAL_DATAFLOW_SCHEMA = _owned_schema(
    "include/Dataflow/IR/DataflowCanonicalEntity.h", "loom.canonical_dataflow"
)
_SIMULATION_WORKLOAD_SCHEMA = _owned_schema(
    "include/Simulator/SimulationArtifacts.h", "loom.simulation_workload"
)
_SIMULATION_RUNTIME_INPUT_SCHEMA = _owned_schema(
    "include/Simulator/SimulationArtifacts.h", "loom.simulation_runtime_input"
)
_RESOLVED_CONFIG_SCHEMA = _owned_schema(
    "include/Config/ResolvedConfig.h", "loom.config.resolved"
)
_FABRIC_SCHEMA = _owned_schema(
    "include/Fabric/Artifact/FabricArtifactCodec.h", "loom.fabric"
)
_MAPPING_SCHEMA = _owned_schema(
    "include/Mapping/IR/MappingSchema.h", "loom.mapping"
)
_MAPPING_CONSTRAINT_SET_SCHEMA = _owned_schema(
    "include/Mapping/Artifact/MappingConstraintSet.h", "loom.mapping_constraints"
)
_EVALUATION_EVIDENCE_SCHEMA = _owned_schema(
    "lib/Evaluation/Evidence.cpp", "evaluation.evidence"
)


def _validate_artifact_reference(
    value: object,
    what: str,
    expected_schema: tuple[str, str] | None = None,
) -> Mapping[str, object]:
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
    if expected_schema is not None and (schema, version) != expected_schema:
        raise ValueError(f"{what} has a foreign artifact schema")
    return value


_CANDIDATE_INCOMPLETE_REASONS = frozenset(
    {
        "candidate_proof_not_established",
        "candidate_semantic_limit_reached",
        "candidate_provider_unavailable",
        "candidate_generation_unsupported",
        "candidate_execution_failed",
        "candidate_cancelled_or_timeout",
    }
)
_TECH_MAPPING_WORK_UNITS = (
    "match_row_attempt",
    "partial_cover_expansion",
    "candidate_evaluation",
    "publication_slot",
)
_SPATIAL_PNR_WORK_UNITS = (
    "seed_attempt",
    "assignment_attempt_per_seed",
    "endpoint_expansion",
    "negotiation_iteration",
    "calibration_proposal",
    "proposal_per_level_base",
    "proposal_per_movable_decision",
    "exact_repair_region_decision",
    "exact_repair_solver_call",
)


def _validate_candidate_generator_result(
    value: object,
    expected_generator_units: Sequence[str],
    what: str,
    *,
    require_completed: bool,
    candidate_schema: tuple[str, str],
) -> str:
    if not isinstance(value, Mapping) or set(value) != {
        "outcome",
        "incomplete_reason",
        "infeasibility_proof",
        "candidates",
        "work_units",
    }:
        raise ValueError(f"{what} result has the wrong shape")
    outcome = value["outcome"]
    reason = value["incomplete_reason"]
    proof = value["infeasibility_proof"]
    if outcome not in {"completed", "incomplete", "proven_infeasible"}:
        raise ValueError(f"{what} result has an unknown outcome")
    if outcome == "incomplete":
        if reason not in _CANDIDATE_INCOMPLETE_REASONS:
            raise ValueError(f"{what} result has a noncanonical incomplete reason")
    elif reason is not None:
        raise ValueError(f"{what} completed result has an incomplete reason")
    if outcome == "proven_infeasible":
        if not isinstance(proof, Mapping) or set(proof) != {"kind", "witness"}:
            raise ValueError(f"{what} infeasibility has no typed proof")
        proof_kind = _nonnegative_integer(proof["kind"], f"{what} proof kind")
        if proof_kind > MAX_CANDIDATE_PROOF_KIND:
            raise ValueError(f"{what} proof kind exceeds its wire domain")
        witness = proof["witness"]
        if (
            not isinstance(witness, str)
            or len(witness) % 2 != 0
            or witness != witness.lower()
            or any(character not in "0123456789abcdef" for character in witness)
        ):
            raise ValueError(f"{what} proof witness is not canonical hex")
    elif proof is not None:
        raise ValueError(f"{what} non-infeasible result carries a proof")
    candidates = value["candidates"]
    if not isinstance(candidates, list):
        raise ValueError(f"{what} candidates are not a list")
    candidate_keys: list[tuple[str, str, str]] = []
    for candidate in candidates:
        reference = _validate_artifact_reference(
            candidate, f"{what} candidate", candidate_schema
        )
        candidate_keys.append(
            (
                str(reference["schema"]),
                str(reference["schema_version"]),
                str(reference["artifact"]),
            )
        )
    if candidate_keys != sorted(set(candidate_keys)):
        raise ValueError(f"{what} candidates are not canonical and unique")
    if outcome == "proven_infeasible" and candidates:
        raise ValueError(f"{what} infeasibility retained a candidate")
    if require_completed and outcome != "completed":
        raise ValueError(f"CGRA gate profile contains a non-completed {what} result")

    work_units = value["work_units"]
    if not isinstance(work_units, list) or len(work_units) != len(
        expected_generator_units
    ):
        raise ValueError(f"{what} generator summary has the wrong width")
    for expected_unit, entry in zip(expected_generator_units, work_units):
        if not isinstance(entry, Mapping) or set(entry) != {
            "unit",
            "planned",
            "consumed",
        }:
            raise ValueError(f"{what} generator work entry has the wrong shape")
        if entry["unit"] != expected_unit:
            raise ValueError(f"{what} generator work order is not canonical")
        planned = _nonnegative_integer(
            entry["planned"], f"planned {what} work {expected_unit}"
        )
        consumed = _nonnegative_integer(
            entry["consumed"], f"consumed {what} work {expected_unit}"
        )
        if consumed > planned:
            raise ValueError(f"{what} consumed work exceeds its plan")
        if (
            outcome in {"completed", "proven_infeasible"}
            and consumed != planned
        ):
            raise ValueError(f"terminal {what} left planned work unconsumed")
    return str(outcome)


def validate_cgra_tech_mapping_result(
    value: object, *, require_completed: bool
) -> str:
    return _validate_candidate_generator_result(
        value,
        _TECH_MAPPING_WORK_UNITS,
        "CGRA TechMapping",
        require_completed=require_completed,
        candidate_schema=_MAPPING_SCHEMA,
    )


def validate_cgra_pnr_result(value: object, *, require_completed: bool) -> str:
    if not isinstance(value, Mapping) or set(value) != {
        "completion_goal",
        "configured_seed_attempts",
        "outcome",
        "incomplete_reason",
        "infeasibility_proof",
        "candidates",
        "work_units",
    }:
        raise ValueError("CGRA PnR result has the wrong shape")
    if value["completion_goal"] != "exhaust_configured_work":
        raise ValueError("CGRA qualification used a prefix PnR completion goal")
    configured_seed_attempts = _nonnegative_integer(
        value["configured_seed_attempts"],
        "CGRA configured PnR seed attempts",
        positive=True,
    )
    base = {key: value[key] for key in (
        "outcome",
        "incomplete_reason",
        "infeasibility_proof",
        "candidates",
        "work_units",
    )}
    outcome = _validate_candidate_generator_result(
        base,
        _SPATIAL_PNR_WORK_UNITS,
        "CGRA Spatial PnR",
        require_completed=require_completed,
        candidate_schema=_MAPPING_SCHEMA,
    )
    if outcome == "completed":
        seed_work = value["work_units"][0]
        assert isinstance(seed_work, Mapping)
        if seed_work["planned"] < configured_seed_attempts:
            raise ValueError("CGRA PnR did not plan the configured restart domain")
    return outcome


def validate_cgra_profile_outcome(value: object) -> tuple[str, str | None]:
    expected_fields = {
        "schema",
        "workload",
        "operator_id",
        "protocol_symbol",
        "stage",
        "resolved_config",
        "fabric",
        "tech_mapping_search",
        "spatial_pnr",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("CGRA profile outcome has the wrong shape")
    if value["schema"] != "loom.cgra_budget_profile_outcome.2":
        raise ValueError("CGRA profile outcome has the wrong schema")
    if (
        not isinstance(value["workload"], str)
        or not isinstance(value["operator_id"], str)
        or not isinstance(value["protocol_symbol"], str)
        or not value["protocol_symbol"]
        or not value["protocol_symbol"].isascii()
    ):
        raise ValueError("CGRA profile outcome has no workload identity")
    _validate_artifact_reference(
        value["resolved_config"], "resolved config", _RESOLVED_CONFIG_SCHEMA
    )
    _validate_artifact_reference(value["fabric"], "Fabric", _FABRIC_SCHEMA)
    tech_outcome = validate_cgra_tech_mapping_result(
        value["tech_mapping_search"], require_completed=False
    )
    stage = value["stage"]
    if stage == "tech_mapping":
        tech_result = value["tech_mapping_search"]
        assert isinstance(tech_result, Mapping)
        if (
            value["spatial_pnr"] is not None
            or tech_result["candidates"]
        ):
            raise ValueError("CGRA TechMapping outcome has an invalid boundary")
        result = value["tech_mapping_search"]
    elif stage == "spatial_pnr":
        tech_result = value["tech_mapping_search"]
        assert isinstance(tech_result, Mapping)
        if not tech_result["candidates"] or tech_outcome not in {
            "completed",
            "incomplete",
        } or (
            tech_outcome == "incomplete"
            and tech_result["incomplete_reason"]
            != "candidate_semantic_limit_reached"
        ):
            raise ValueError("CGRA Spatial PnR ran after unusable TechMapping")
        pnr_outcome = validate_cgra_pnr_result(
            value["spatial_pnr"], require_completed=False
        )
        pnr_result = value["spatial_pnr"]
        assert isinstance(pnr_result, Mapping)
        if pnr_outcome == "completed" and pnr_result["candidates"]:
            raise ValueError(
                "CGRA profile outcome contains a usable completed PnR result"
            )
        result = value["spatial_pnr"]
    else:
        raise ValueError("CGRA profile outcome has an unknown stage")
    assert isinstance(result, Mapping)
    reason = result["incomplete_reason"]
    return str(result["outcome"]), reason if isinstance(reason, str) else None


def _validate_cgra_profiles(
    profiles: Sequence[Mapping[str, object]],
    representative_operators: Sequence[CgraRepresentativeOperator] | None = None,
) -> tuple[Mapping[str, object], ...]:
    if representative_operators is None:
        _, representative_operators = load_cgra_representative_operators()
    operator_by_workload = {
        operator.workload: operator for operator in representative_operators
    }
    expected_profile_fields = {
        "schema",
        "workload",
        "operator_id",
        "protocol_symbol",
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
        "tech_mapping_search",
        "initial_spatial_mapping",
        "spatial_mapping",
        "spatial_pnr",
        "transport_repair",
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
        if profile["schema"] != "loom.cgra_budget_profile.5":
            raise ValueError("CGRA profile has the wrong schema")
        workload = profile["workload"]
        if not isinstance(workload, str) or workload in by_workload:
            raise ValueError("CGRA profile workload is absent or duplicated")
        operator = operator_by_workload.get(workload)
        if (
            operator is None
            or profile["operator_id"] != operator.operator_id
            or profile["protocol_symbol"] != operator.protocol_symbol
        ):
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
        reference_schemas = {
            "canonical_dataflow": _CANONICAL_DATAFLOW_SCHEMA,
            "simulation_workload": _SIMULATION_WORKLOAD_SCHEMA,
            "simulation_runtime_input": _SIMULATION_RUNTIME_INPUT_SCHEMA,
            "resolved_config": _RESOLVED_CONFIG_SCHEMA,
            "fabric": _FABRIC_SCHEMA,
            "tech_mapping": _MAPPING_SCHEMA,
            "initial_spatial_mapping": _MAPPING_SCHEMA,
            "spatial_mapping": _MAPPING_SCHEMA,
            "warmup_evidence": _EVALUATION_EVIDENCE_SCHEMA,
        }
        for field, schema in reference_schemas.items():
            _validate_artifact_reference(
                profile[field], f"CGRA profile {field}", schema
            )
        resolved_config = profile["resolved_config"]
        assert isinstance(resolved_config, Mapping)
        if (
            resolved_config["schema"] != "loom.config.resolved"
            or resolved_config["schema_version"] != "11.4"
        ):
            raise ValueError("CGRA profile uses a foreign ResolvedConfig schema")
        resolved_config_identities.add(str(resolved_config["artifact"]))
        fabric = profile["fabric"]
        assert isinstance(fabric, Mapping)
        fabric_identities.add(str(fabric["artifact"]))
        tech_outcome = validate_cgra_tech_mapping_result(
            profile["tech_mapping_search"], require_completed=False
        )
        tech_search = profile["tech_mapping_search"]
        assert isinstance(tech_search, Mapping)
        if tech_outcome not in {"completed", "incomplete"} or (
            tech_outcome == "incomplete"
            and tech_search["incomplete_reason"]
            != "candidate_semantic_limit_reached"
        ):
            raise ValueError("CGRA profile has no usable TechMapping frontier")
        if profile["tech_mapping"] not in tech_search["candidates"]:
            raise ValueError("selected TechMapping is absent from the complete search")
        validate_cgra_pnr_result(profile["spatial_pnr"], require_completed=True)
        initial_pnr = profile["spatial_pnr"]
        assert isinstance(initial_pnr, Mapping)
        if profile["initial_spatial_mapping"] not in initial_pnr["candidates"]:
            raise ValueError("CGRA initial Mapping is absent from its PnR result")
        transport_repair = profile["transport_repair"]
        if transport_repair is None:
            if profile["spatial_mapping"] != profile["initial_spatial_mapping"]:
                raise ValueError("CGRA final Mapping has no repair lineage")
        else:
            if not isinstance(transport_repair, Mapping) or set(
                transport_repair
            ) != {"parent_system_mapping", "pre_repair_evidence", "attempts"}:
                raise ValueError("CGRA transport repair has the wrong shape")
            _validate_artifact_reference(
                transport_repair["parent_system_mapping"],
                "CGRA repair parent SystemMapping",
                _MAPPING_SCHEMA,
            )
            _validate_artifact_reference(
                transport_repair["pre_repair_evidence"],
                "CGRA pre-repair Evidence",
                _EVALUATION_EVIDENCE_SCHEMA,
            )
            attempts = transport_repair["attempts"]
            if not isinstance(attempts, list) or not attempts:
                raise ValueError("CGRA transport repair has no attempt")
            accepted_children: list[Mapping[str, object]] = []
            for ordinal, attempt in enumerate(attempts):
                if not isinstance(attempt, Mapping) or set(attempt) != {
                    "parent_spatial_mapping",
                    "constraint_set",
                    "spatial_pnr",
                    "child_spatial_mapping",
                    "accepted_for_simulation",
                }:
                    raise ValueError("CGRA transport repair attempt has the wrong shape")
                if attempt["parent_spatial_mapping"] != profile[
                    "initial_spatial_mapping"
                ]:
                    raise ValueError("CGRA transport repair has a foreign parent")
                _validate_artifact_reference(
                    attempt["constraint_set"],
                    "CGRA repair constraint",
                    _MAPPING_CONSTRAINT_SET_SCHEMA,
                )
                repair_outcome = validate_cgra_pnr_result(
                    attempt["spatial_pnr"], require_completed=False
                )
                repair_pnr = attempt["spatial_pnr"]
                assert isinstance(repair_pnr, Mapping)
                if repair_pnr["configured_seed_attempts"] != initial_pnr[
                    "configured_seed_attempts"
                ]:
                    raise ValueError("CGRA repair used a foreign PnR seed plan")
                child = attempt["child_spatial_mapping"]
                accepted = attempt["accepted_for_simulation"]
                if not isinstance(accepted, bool):
                    raise ValueError("CGRA repair acceptance is not boolean")
                if child is None:
                    if accepted:
                        raise ValueError("CGRA repair accepted no child Mapping")
                else:
                    child_reference = _validate_artifact_reference(
                        child, "CGRA repair child Mapping", _MAPPING_SCHEMA
                    )
                    if repair_outcome != "completed" or child not in repair_pnr[
                        "candidates"
                    ]:
                        raise ValueError("CGRA repair child lacks a completed PnR result")
                    if accepted:
                        if ordinal + 1 != len(attempts):
                            raise ValueError("CGRA accepted repair is not terminal")
                        accepted_children.append(child_reference)
            if len(accepted_children) != 1 or profile[
                "spatial_mapping"
            ] != accepted_children[0]:
                raise ValueError("CGRA final Mapping disagrees with repair receipt")
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
                _EVALUATION_EVIDENCE_SCHEMA,
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


def _derive_cgra_spatial_budget_from_validated_profiles(
    validated: Sequence[Mapping[str, object]],
) -> int:
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


def derive_cgra_spatial_budget_nanoseconds(
    profiles: Sequence[Mapping[str, object]],
) -> int:
    return _derive_cgra_spatial_budget_from_validated_profiles(
        _validate_cgra_profiles(profiles)
    )


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
    if root["schema"] != "loom.cgra_simulation_gate.5":
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
    current_operator_gate_sha256, representative_operators = (
        load_cgra_representative_operators()
    )
    if operator_gate != {
        "path": CGRA_OPERATOR_GATE_RELATIVE_PATH,
        "sha256": current_operator_gate_sha256,
    }:
        raise ValueError("CGRA gate configuration names a foreign operator gate")
    profiles = root["profiles"]
    if not isinstance(profiles, list):
        raise ValueError("CGRA gate profiles are not a list")
    validated = _validate_cgra_profiles(profiles, representative_operators)
    derived = _derive_cgra_spatial_budget_from_validated_profiles(validated)
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


def _live_process_group_members(process_group: int) -> tuple[int, ...]:
    proc_root = Path("/proc")
    if proc_root.is_dir():
        members: list[int] = []
        for entry in proc_root.iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="ascii")
                suffix = stat[stat.rfind(")") + 2 :].split()
                state = suffix[0]
                member_group = int(suffix[2])
            except (OSError, UnicodeDecodeError, ValueError, IndexError):
                continue
            if member_group == process_group and state != "Z":
                members.append(int(entry.name))
        return tuple(sorted(members))
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return ()
    except PermissionError:
        pass
    return (process_group,)


def _wait_for_process_group_exit(process_group: int, grace_seconds: float) -> bool:
    deadline = time.monotonic() + grace_seconds
    while _live_process_group_members(process_group):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        time.sleep(min(0.01, remaining))
    return True


def _terminate_process_group(process_group: int, grace_seconds: float) -> bool:
    if not _live_process_group_members(process_group):
        return True
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return True
    if _wait_for_process_group_exit(process_group, grace_seconds):
        return True
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return _wait_for_process_group_exit(process_group, grace_seconds)


def _captured_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def execute_process(
    command: Sequence[str | os.PathLike[str]],
    timeout_budget_seconds: float,
    *,
    termination_grace_seconds: float | None = None,
    environment: Mapping[str, str] | None = None,
) -> ProcessExecution:
    _require_finite_positive(timeout_budget_seconds, "process timeout budget")
    if termination_grace_seconds is None:
        termination_grace_seconds = float(timeout_seconds(Tier.ULTRAFAST))
    _require_finite_positive(termination_grace_seconds, "termination grace")
    normalized = tuple(os.fspath(argument) for argument in command)
    if not normalized:
        raise ValueError("process command must not be empty")
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            normalized,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=None if environment is None else dict(environment),
        )
    except OSError as error:
        return ProcessExecution(
            ProcessDisposition.LAUNCH_FAILED,
            normalized,
            time.monotonic() - started,
            None,
            "",
            str(error),
            False,
        )

    try:
        stdout, stderr = process.communicate(timeout=timeout_budget_seconds)
    except subprocess.TimeoutExpired as timeout:
        terminated = _terminate_process_group(process.pid, termination_grace_seconds)
        try:
            stdout, stderr = process.communicate(timeout=termination_grace_seconds)
        except subprocess.TimeoutExpired as cleanup_timeout:
            stdout = _captured_text(cleanup_timeout.stdout or timeout.stdout)
            stderr = _captured_text(cleanup_timeout.stderr or timeout.stderr)
            terminated = False
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            try:
                process.wait(timeout=termination_grace_seconds)
            except subprocess.TimeoutExpired:
                pass
        return ProcessExecution(
            (
                ProcessDisposition.TIMED_OUT
                if terminated
                else ProcessDisposition.CLEANUP_FAILED
            ),
            normalized,
            time.monotonic() - started,
            process.returncode,
            _captured_text(stdout),
            _captured_text(stderr),
            terminated,
        )

    if _live_process_group_members(process.pid):
        terminated = _terminate_process_group(process.pid, termination_grace_seconds)
        return ProcessExecution(
            ProcessDisposition.CLEANUP_FAILED,
            normalized,
            time.monotonic() - started,
            process.returncode,
            stdout,
            stderr,
            terminated,
        )
    return ProcessExecution(
        (
            ProcessDisposition.COMPLETED
            if process.returncode == 0
            else ProcessDisposition.NONZERO_EXIT
        ),
        normalized,
        time.monotonic() - started,
        process.returncode,
        stdout,
        stderr,
        False,
    )


_MEASUREMENT_FIELDS = {
    "schema",
    "cell",
    "attempt",
    "invocation",
    "work_fingerprint",
    "config_fingerprint",
    "accelerator_reference_cycles",
    "cgra_event_frames",
    "active_wall_ns",
    "active_cpu_ns",
    "gem5_ticks",
    "setup_wall_ns",
    "process_peak_rss_bytes",
    "measurement_source",
    "rss_scope",
}
_MEASUREMENT_SCHEMA = "loom.paired_simulation_measurement.2"
_MEASUREMENT_PREFIX = "paired-simulation "


def _parse_u64(value: str) -> int:
    if (
        not value
        or any(character not in "0123456789" for character in value)
        or (len(value) != 1 and value.startswith("0"))
    ):
        raise ValueError("measurement contains a noncanonical unsigned integer")
    parsed = int(value)
    if parsed > (1 << 64) - 1:
        raise ValueError("measurement escapes the unsigned 64-bit domain")
    return parsed


def _parse_fingerprint(value: str, what: str) -> str:
    if (
        len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{what} fingerprint is not canonical")
    return value


def _parse_measurement_row(row: str, expected_cell: str) -> ExecutionMatrixMeasurement:
    fields: dict[str, str] = {}
    for item in row.split()[1:]:
        key, separator, value = item.partition("=")
        if not separator or not key or not value or key in fields:
            raise ValueError("paired measurement row is malformed")
        fields[key] = value
    if set(fields) != _MEASUREMENT_FIELDS:
        raise ValueError("paired measurement row has the wrong shape")
    if fields["schema"] != _MEASUREMENT_SCHEMA or fields["cell"] != expected_cell:
        raise ValueError("paired measurement row has the wrong identity")
    expected_invocation = {
        "paired-spatial-cgra": ("ordinary", "paired-spatial-cgra"),
        "paired-system-cgra": ("diagnostic", "paired-system-cgra"),
    }.get(expected_cell)
    if expected_invocation is None:
        raise ValueError("measurement cell is outside the paired CGRA domain")
    if (fields["attempt"], fields["invocation"]) != expected_invocation:
        raise ValueError("paired measurement has the wrong attempt identity")

    work_fingerprint = _parse_fingerprint(fields["work_fingerprint"], "work")
    config_fingerprint = _parse_fingerprint(fields["config_fingerprint"], "config")
    cycles = _parse_u64(fields["accelerator_reference_cycles"])
    frames = _parse_u64(fields["cgra_event_frames"])
    active_wall_ns = _parse_u64(fields["active_wall_ns"])
    active_cpu_ns = _parse_u64(fields["active_cpu_ns"])
    setup_wall_ns = _parse_u64(fields["setup_wall_ns"])
    peak_rss_bytes = _parse_u64(fields["process_peak_rss_bytes"])
    if cycles == 0 or frames == 0 or active_wall_ns == 0:
        raise ValueError("paired measurement contains no completed active work")

    if expected_cell == "paired-spatial-cgra":
        expected_source = "direct_spatial_attempt"
        expected_rss_scope = "self_process_lifetime"
        if fields["gem5_ticks"] != "not_applicable":
            raise ValueError("Spatial measurement contains gem5 ticks")
        gem5_ticks = None
        engine_cpu_seconds = active_cpu_ns / 1_000_000_000.0
        host_cpu_seconds = 0.0
    else:
        expected_source = "fresh_system_diagnostic"
        expected_rss_scope = "child_process_lifetime"
        gem5_ticks = _parse_u64(fields["gem5_ticks"])
        if gem5_ticks == 0:
            raise ValueError("System measurement contains no gem5 progress")
        engine_cpu_seconds = 0.0
        host_cpu_seconds = active_cpu_ns / 1_000_000_000.0
    if (
        fields["measurement_source"] != expected_source
        or fields["rss_scope"] != expected_rss_scope
    ):
        raise ValueError("paired measurement has the wrong ownership boundary")

    return ExecutionMatrixMeasurement(
        cell=expected_cell,
        attempt=fields["attempt"],
        invocation=fields["invocation"],
        work_fingerprint=work_fingerprint,
        config_fingerprint=config_fingerprint,
        gem5_ticks=gem5_ticks,
        setup_wall_seconds=setup_wall_ns / 1_000_000_000.0,
        active_process_cpu_seconds=active_cpu_ns / 1_000_000_000.0,
        measurement_source=expected_source,
        rss_scope=expected_rss_scope,
        timing=ActiveExecutionTiming(
            active_wall_seconds=active_wall_ns / 1_000_000_000.0,
            reference_cycles=cycles,
            engine_cpu_seconds=engine_cpu_seconds,
            host_cpu_seconds=host_cpu_seconds,
            event_count=frames,
            peak_resident_bytes=peak_rss_bytes,
        ),
    )


def parse_execution_matrix_measurements(
    output: str, expected_cell: str, expected_count: int
) -> tuple[ExecutionMatrixMeasurement, ...]:
    if expected_count < 1:
        raise ValueError("expected measurement count must be positive")
    rows = [
        line for line in output.splitlines() if line.startswith(_MEASUREMENT_PREFIX)
    ]
    if len(rows) != expected_count:
        raise ValueError("execution matrix output has the wrong measurement count")
    return tuple(_parse_measurement_row(row, expected_cell) for row in rows)


def parse_execution_matrix_measurement(
    output: str, expected_cell: str
) -> ExecutionMatrixMeasurement:
    return parse_execution_matrix_measurements(output, expected_cell, 1)[0]


def _failed_report(
    disposition: MeasurementDisposition,
    spatial_measurements: Sequence[ExecutionMatrixMeasurement],
    *,
    spatial_process: ProcessExecution | None,
    system_process: ProcessExecution | None,
    diagnostic: str,
    system_measurement: ExecutionMatrixMeasurement | None = None,
) -> PairedMeasurementReport:
    return PairedMeasurementReport(
        disposition=disposition,
        spatial_measurements=tuple(spatial_measurements),
        system_measurement=system_measurement,
        paired_result=None,
        spatial_process=spatial_process,
        system_process=system_process,
        diagnostic=diagnostic,
    )


def _process_failure_disposition(
    process: ProcessExecution, *, spatial: bool
) -> MeasurementDisposition:
    if process.disposition is ProcessDisposition.CLEANUP_FAILED:
        return MeasurementDisposition.PROCESS_CLEANUP_FAILED
    if process.disposition is ProcessDisposition.TIMED_OUT:
        return (
            MeasurementDisposition.SPATIAL_TIMED_OUT
            if spatial
            else MeasurementDisposition.SYSTEM_TIMED_OUT
        )
    return (
        MeasurementDisposition.SPATIAL_EXECUTION_FAILED
        if spatial
        else MeasurementDisposition.SYSTEM_EXECUTION_FAILED
    )


def run_paired_execution_matrix(
    execution_matrix_runner: Path,
    gem5_readiness: Path,
    *,
    spatial_warmup_runs: int = 1,
    spatial_measurement_runs: int = 3,
    spatial_process_timeout_seconds: float | None = None,
    system_process_timeout_seconds: float | None = None,
    termination_grace_seconds: float | None = None,
) -> PairedMeasurementReport:
    if spatial_warmup_runs < 1 or spatial_measurement_runs < 1:
        raise ValueError("paired measurement requires warmup and measured runs")
    spatial_timeout = (
        float(timeout_seconds(Tier.MEDIUM))
        if spatial_process_timeout_seconds is None
        else spatial_process_timeout_seconds
    )
    system_timeout = (
        float(timeout_seconds(Tier.XLONG))
        if system_process_timeout_seconds is None
        else system_process_timeout_seconds
    )
    runner = str(execution_matrix_runner.resolve(strict=True))
    readiness = str(gem5_readiness.resolve(strict=True))
    scratch_root = ROOT / "temp"
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
        spatial_timeout,
        termination_grace_seconds=termination_grace_seconds,
        environment=process_environment,
    )
    if spatial_process.disposition is not ProcessDisposition.COMPLETED:
        return _failed_report(
            _process_failure_disposition(spatial_process, spatial=True),
            (),
            spatial_process=spatial_process,
            system_process=None,
            diagnostic=spatial_process.stderr.strip() or spatial_process.stdout.strip(),
        )
    try:
        spatial_measurements = parse_execution_matrix_measurements(
            spatial_process.stdout,
            "paired-spatial-cgra",
            spatial_measurement_runs,
        )
    except ValueError as error:
        return _failed_report(
            MeasurementDisposition.SPATIAL_EXECUTION_FAILED,
            (),
            spatial_process=spatial_process,
            system_process=None,
            diagnostic=str(error),
        )

    expected = spatial_measurements[0]
    expected_work = (
        expected.work_fingerprint,
        expected.config_fingerprint,
        expected.timing.reference_cycles,
        expected.timing.event_count,
    )
    if any(
        (
            sample.work_fingerprint,
            sample.config_fingerprint,
            sample.timing.reference_cycles,
            sample.timing.event_count,
        )
        != expected_work
        for sample in spatial_measurements[1:]
    ):
        return _failed_report(
            MeasurementDisposition.PAIRED_WORK_MISMATCH,
            spatial_measurements,
            spatial_process=spatial_process,
            system_process=None,
            diagnostic="Spatial measurements do not name one exact work/config pair",
        )
    if any(
        sample.timing.active_wall_seconds > CGRA_SPATIAL_BOOTSTRAP_BUDGET_SECONDS
        for sample in spatial_measurements
    ):
        return _failed_report(
            MeasurementDisposition.SPATIAL_BOOTSTRAP_BUDGET_EXCEEDED,
            spatial_measurements,
            spatial_process=spatial_process,
            system_process=None,
            diagnostic="a Spatial sample exceeded the provisional bootstrap budget",
        )

    system_process = execute_process(
        (runner, "paired-system-cgra", readiness),
        system_timeout,
        termination_grace_seconds=termination_grace_seconds,
        environment=process_environment,
    )
    if system_process.disposition is not ProcessDisposition.COMPLETED:
        return _failed_report(
            _process_failure_disposition(system_process, spatial=False),
            spatial_measurements,
            spatial_process=spatial_process,
            system_process=system_process,
            diagnostic=system_process.stderr.strip() or system_process.stdout.strip(),
        )
    try:
        system_measurement = parse_execution_matrix_measurement(
            system_process.stdout, "paired-system-cgra"
        )
    except ValueError as error:
        return _failed_report(
            MeasurementDisposition.SYSTEM_EXECUTION_FAILED,
            spatial_measurements,
            spatial_process=spatial_process,
            system_process=system_process,
            diagnostic=str(error),
        )

    observed_system_work = (
        system_measurement.work_fingerprint,
        system_measurement.config_fingerprint,
        system_measurement.timing.reference_cycles,
        system_measurement.timing.event_count,
    )
    if observed_system_work != expected_work:
        return _failed_report(
            MeasurementDisposition.PAIRED_WORK_MISMATCH,
            spatial_measurements,
            spatial_process=spatial_process,
            system_process=system_process,
            diagnostic="System measurement names different work or configuration",
            system_measurement=system_measurement,
        )

    budget = paired_system_budget(
        [sample.timing.active_wall_seconds for sample in spatial_measurements],
        CGRA_SPATIAL_BOOTSTRAP_BUDGET_SECONDS,
    )
    paired = evaluate_paired_execution(budget, system_measurement.timing)
    if paired.hard_ratio_failure:
        disposition = MeasurementDisposition.HARD_RATIO_EXCEEDED
    elif not paired.within_system_budget:
        disposition = MeasurementDisposition.SYSTEM_BOOTSTRAP_BUDGET_EXCEEDED
    elif not paired.meets_reference_rate_target:
        disposition = MeasurementDisposition.REFERENCE_RATE_BELOW_TARGET
    else:
        disposition = MeasurementDisposition.MEASURED
    return PairedMeasurementReport(
        disposition=disposition,
        spatial_measurements=spatial_measurements,
        system_measurement=system_measurement,
        paired_result=paired,
        spatial_process=spatial_process,
        system_process=system_process,
    )


def _measurement_json(
    measurement: ExecutionMatrixMeasurement | None,
) -> dict[str, object] | None:
    if measurement is None:
        return None
    return {
        "cell": measurement.cell,
        "attempt": measurement.attempt,
        "invocation": measurement.invocation,
        "work_fingerprint": measurement.work_fingerprint,
        "config_fingerprint": measurement.config_fingerprint,
        "accelerator_reference_cycles": measurement.timing.reference_cycles,
        "cgra_event_frames": measurement.timing.event_count,
        "active_wall_seconds": measurement.timing.active_wall_seconds,
        "active_process_cpu_seconds": measurement.active_process_cpu_seconds,
        "gem5_ticks": measurement.gem5_ticks,
        "setup_wall_seconds": measurement.setup_wall_seconds,
        "process_peak_rss_bytes": measurement.timing.peak_resident_bytes,
        "measurement_source": measurement.measurement_source,
        "rss_scope": measurement.rss_scope,
    }


def _process_json(process: ProcessExecution | None) -> dict[str, object] | None:
    if process is None:
        return None
    return {
        "disposition": process.disposition.value,
        "elapsed_seconds": process.elapsed_seconds,
        "return_code": process.return_code,
        "process_group_terminated": process.process_group_terminated,
    }


def report_json(report: PairedMeasurementReport) -> dict[str, object]:
    projected: dict[str, object] = {
        "schema": "loom.paired_simulation_measurement_report.2",
        "publication_status": "provisional_bootstrap_only",
        "disposition": report.disposition.value,
        "bootstrap_budget_source": "config.timeout_budgets:medium",
        "spatial_timeout_source": "config.timeout_budgets:medium",
        "system_timeout_source": "config.timeout_budgets:xlong",
        "cleanup_timeout_source": "config.timeout_budgets:ultrafast",
        "durable_replay_profiles": 0,
        "spatial": [
            _measurement_json(measurement)
            for measurement in report.spatial_measurements
        ],
        "system": _measurement_json(report.system_measurement),
        "spatial_process": _process_json(report.spatial_process),
        "system_process": _process_json(report.system_process),
        "diagnostic": report.diagnostic,
        "paired": None,
    }
    if report.paired_result is not None:
        paired = report.paired_result
        projected["paired"] = {
            "spatial_reference_seconds": paired.spatial_reference_seconds,
            "system_active_wall_seconds": paired.system_active_wall_seconds,
            "system_to_spatial_ratio": paired.system_to_spatial_ratio,
            "system_budget_seconds": paired.system_budget_seconds,
            "within_system_budget": paired.within_system_budget,
            "hard_ratio_failure": paired.hard_ratio_failure,
            "reference_cycles": paired.reference_cycles,
            "reference_cycles_per_second": paired.reference_cycles_per_second,
            "reference_rate_basis_seconds": paired.reference_rate_basis_seconds,
            "meets_reference_rate_target": paired.meets_reference_rate_target,
        }
    return projected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_matrix_runner", type=Path)
    parser.add_argument("gem5_readiness", type=Path)
    parser.add_argument("--spatial-warmup-runs", type=int, default=1)
    parser.add_argument("--spatial-measurement-runs", type=int, default=3)
    arguments = parser.parse_args(argv)
    measurement_attempts = 1
    report = run_paired_execution_matrix(
        arguments.execution_matrix_runner,
        arguments.gem5_readiness,
        spatial_warmup_runs=arguments.spatial_warmup_runs,
        spatial_measurement_runs=arguments.spatial_measurement_runs,
    )
    if report.disposition is MeasurementDisposition.REFERENCE_RATE_BELOW_TARGET:
        # A transient host-load dip can miss the reference rate while the
        # measurement machinery is healthy. One bounded remeasurement keeps
        # the rate gate meaningful: an order-of-magnitude host regression
        # fails both attempts, while a scheduling blip does not fail the
        # suite. Budget and ratio failures are not retried.
        measurement_attempts += 1
        report = run_paired_execution_matrix(
            arguments.execution_matrix_runner,
            arguments.gem5_readiness,
            spatial_warmup_runs=arguments.spatial_warmup_runs,
            spatial_measurement_runs=arguments.spatial_measurement_runs,
        )
    payload = report_json(report)
    payload["measurement_attempts"] = measurement_attempts
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.disposition is MeasurementDisposition.MEASURED else 1


if __name__ == "__main__":
    raise SystemExit(main())
