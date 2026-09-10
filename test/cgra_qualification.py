"""Qualification evidence and the shared CGRA execution-budget authority."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.timeout_budgets import Tier, seconds as timeout_seconds  # noqa: E402


REFERENCE_RATE_TARGET_CYCLES_PER_SECOND = 100_000
CGRA_QUALIFICATION_LIMIT_NANOSECONDS = 45_000_000_000
CGRA_QUALIFICATION_WARMUP_RUNS = 1
CGRA_QUALIFICATION_MEASUREMENT_RUNS = 3
MAX_CANDIDATE_PROOF_KIND = (1 << 32) - 1
CGRA_GATE_RELATIVE_PATH = "test/data/cgra-simulation-gate-v1.json"
CGRA_GATE_SCHEMA = "loom.cgra_simulation_gate.7"
CGRA_GATE_CONFIGURATION = ROOT / CGRA_GATE_RELATIVE_PATH
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


class CgraGateSource(str, Enum):
    #: The tracked ten-profile qualification published by the gate generator.
    TRACKED = "tracked"
    #: No tracked gate exists yet, so the canonical medium tier stands in. The
    #: value is provisional and is labelled as such in every report; it is
    #: never derived from a case, a caller, or a simulator.
    PROVISIONAL_BOOTSTRAP = "provisional_bootstrap"


class CgraTransportRepairTermination(str, Enum):
    """Wire spellings owned by SpatialTransportCegarTermination."""

    RETIRED = "retired"
    PROOF_NOT_ESTABLISHED = "proof_not_established"
    NO_PROGRESS = "no_progress"
    REPEATED_CERTIFICATE = "repeated_certificate"
    REPAIR_TERMINAL = "repair_terminal"
    RUNTIME_INCOMPLETE = "runtime_incomplete"
    ITERATION_BUDGET_EXHAUSTED = "iteration_budget_exhausted"
    CLAUSE_BUDGET_EXHAUSTED = "clause_budget_exhausted"
    TIMED_OUT = "timed_out"


class SpatialExactRepairKind(IntEnum):
    """Wire ordinals owned by PnR/SpatialExactRepair.h."""

    REPAIRED = 0
    REGION_INFEASIBLE_UNDER_FIXED_BOUNDARY = 1
    UNKNOWN_BUDGET_EXHAUSTED = 2
    TIMED_OUT = 3
    ROUTING_INCOMPLETE = 4
    PROOF_NOT_ESTABLISHED = 5
    REGION_TOO_LARGE = 6
    UNSUPPORTED_ENCODING = 7
    INTERNAL_ERROR = 8


@dataclass(frozen=True)
class CgraGateConfiguration:
    spatial_absolute_budget_nanoseconds: int
    configuration_sha256: str
    operator_gate_sha256: str
    profiles: tuple[Mapping[str, object], ...]
    source: CgraGateSource = CgraGateSource.TRACKED

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
        raise RuntimeError(f"schema owner {relative_path} does not define {name}")
    version = re.search(r"\{\s*(\d+)\s*,\s*(\d+)\s*\}", source[anchor:])
    if version is None:
        raise RuntimeError(f"schema owner {relative_path} has no version for {name}")
    return (name, f"{version.group(1)}.{version.group(2)}")


def _profile_schema(name: str) -> str:
    source = (ROOT / "test/system/CgraBudgetProfile.cpp").read_text(encoding="ascii")
    literal = re.search(
        rf'constexpr llvm::StringLiteral {name}\s*=\s*"([^"]+)";', source
    )
    if literal is None:
        raise RuntimeError(f"CGRA profile producer does not define {name}")
    return literal.group(1)


CGRA_PROFILE_SCHEMA = _profile_schema("kProfileSchema")
CGRA_PROFILE_OUTCOME_SCHEMA = _profile_schema("kProfileOutcomeSchema")
CGRA_HARDWARE_SEARCH_SCHEMA = _profile_schema("kHardwareSearchSchema")


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
_MAPPING_SCHEMA = _owned_schema("include/Mapping/IR/MappingSchema.h", "loom.mapping")
_MAPPING_CONSTRAINT_SET_SCHEMA = _owned_schema(
    "include/Mapping/Artifact/MappingConstraintSet.h", "loom.mapping_constraints"
)
_EVALUATION_EVIDENCE_SCHEMA = _owned_schema(
    "lib/Evaluation/Evidence.cpp", "evaluation.evidence"
)

# Every artifact schema version the runner accepts, keyed by schema name and
# derived from the C++ owners above; fixtures take versions from here.
OWNED_SCHEMA_VERSIONS: dict[str, str] = dict(
    (
        _CANONICAL_DATAFLOW_SCHEMA,
        _SIMULATION_WORKLOAD_SCHEMA,
        _SIMULATION_RUNTIME_INPUT_SCHEMA,
        _RESOLVED_CONFIG_SCHEMA,
        _FABRIC_SCHEMA,
        _MAPPING_SCHEMA,
        _MAPPING_CONSTRAINT_SET_SCHEMA,
        _EVALUATION_EVIDENCE_SCHEMA,
    )
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
        if outcome in {"completed", "proven_infeasible"} and consumed != planned:
            raise ValueError(f"terminal {what} left planned work unconsumed")
    return str(outcome)


def validate_cgra_tech_mapping_result(value: object, *, require_completed: bool) -> str:
    return _validate_candidate_generator_result(
        value,
        _TECH_MAPPING_WORK_UNITS,
        "CGRA TechMapping",
        require_completed=require_completed,
        candidate_schema=_MAPPING_SCHEMA,
    )


def validate_cgra_pnr_result(value: object, *, require_completed: bool) -> str:
    required_fields = {
        "completion_goal",
        "configured_seed_attempts",
        "outcome",
        "incomplete_reason",
        "infeasibility_proof",
        "candidates",
        "work_units",
    }
    optional_timing_fields = {"deadline_ns", "deadline_overrun_ns"}
    if (
        not isinstance(value, Mapping)
        or not required_fields.issubset(value)
        or not set(value).issubset(required_fields | optional_timing_fields)
    ):
        raise ValueError("CGRA PnR result has the wrong shape")
    for field in optional_timing_fields:
        if field in value:
            _nonnegative_integer(value[field], f"CGRA PnR {field}")
    if value["completion_goal"] != "exhaust_configured_work":
        raise ValueError("CGRA qualification used a prefix PnR completion goal")
    configured_seed_attempts = _nonnegative_integer(
        value["configured_seed_attempts"],
        "CGRA configured PnR seed attempts",
        positive=True,
    )
    base = {
        key: value[key]
        for key in (
            "outcome",
            "incomplete_reason",
            "infeasibility_proof",
            "candidates",
            "work_units",
        )
    }
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


def _validate_replay_input(value: object) -> tuple[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"workload", "runtime_input"}:
        raise ValueError("CGRA replay input is not an exact reference pair")
    workload = _validate_artifact_reference(
        value["workload"], "replay workload", _SIMULATION_WORKLOAD_SCHEMA
    )
    runtime_input = _validate_artifact_reference(
        value["runtime_input"], "replay runtime input", _SIMULATION_RUNTIME_INPUT_SCHEMA
    )
    return str(workload["artifact"]), str(runtime_input["artifact"])


def _validate_source_replay_cases(value: object, occurrences: object) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ValueError("CGRA source has no replay inputs")
    keys = [_validate_replay_input(entry) for entry in value]
    if len(set(keys)) != len(keys):
        raise ValueError("CGRA source repeats an exact replay input")
    if _nonnegative_integer(occurrences, "source replay occurrences", positive=True) < len(keys):
        raise ValueError("CGRA source lost its replay occurrences")
    return value


def _validate_cgra_screening(
    screening: object, candidates: list[object]
) -> None:
    if not isinstance(screening, list) or len(screening) != len(candidates):
        raise ValueError("CGRA screening does not cover its published Spatial frontier")
    for entry, candidate in zip(screening, candidates):
        if not isinstance(entry, Mapping) or set(entry) != {
            "spatial_mapping",
            "buffered_fifo_traversals",
            "bypass_fifo_traversals",
            "retired",
            "closed_wait_actor_cycle_edges",
            "closed_wait_pending_transfers",
            "closed_wait_certificate_edges",
            "closed_wait_certificate_closed",
            "closed_wait_proof_failure",
            "operand_queue_shared_ingress_pressure",
        }:
            raise ValueError("CGRA candidate screening entry has the wrong shape")
        if entry["spatial_mapping"] != candidate:
            raise ValueError("CGRA screening changed the canonical Spatial frontier")
        if not isinstance(entry["retired"], bool):
            raise ValueError("CGRA screening retirement flag is not boolean")
        for field in ("buffered_fifo_traversals", "bypass_fifo_traversals"):
            _nonnegative_integer(entry[field], f"CGRA screening {field}")
        for field in (
            "closed_wait_actor_cycle_edges",
            "closed_wait_pending_transfers",
            "closed_wait_certificate_edges",
            "closed_wait_proof_failure",
            "operand_queue_shared_ingress_pressure",
        ):
            if entry[field] is not None:
                _nonnegative_integer(entry[field], f"CGRA screening {field}")
        closed = entry["closed_wait_certificate_closed"]
        if closed is not None and not isinstance(closed, bool):
            raise ValueError("CGRA screening certificate closure is not boolean")


def _validate_cgra_phases(ledger: object) -> None:
    if not isinstance(ledger, list) or not ledger:
        raise ValueError("CGRA phase ledger is absent")
    phases: set[str] = set()
    for entry in ledger:
        if not isinstance(entry, Mapping) or set(entry) != {
            "phase",
            "wall_nanoseconds",
            "process_cpu_nanoseconds",
        }:
            raise ValueError("CGRA phase ledger entry has the wrong shape")
        phase = entry["phase"]
        if not isinstance(phase, str) or not phase or phase in phases:
            raise ValueError("CGRA phase ledger name is absent or repeated")
        phases.add(phase)
        _nonnegative_integer(entry["wall_nanoseconds"], "CGRA phase wall time")
        _nonnegative_integer(
            entry["process_cpu_nanoseconds"], "CGRA phase process time"
        )


def _validate_transport_repair(
    value: object, initial_mapping: Mapping[str, object]
) -> tuple[CgraTransportRepairTermination, Mapping[str, object]]:
    if not isinstance(value, Mapping) or set(value) != {
        "parent_system_mapping",
        "pre_repair_evidence",
        "termination",
        "attempts",
    }:
        raise ValueError("CGRA transport repair has the wrong shape")
    _validate_artifact_reference(
        value["parent_system_mapping"],
        "CGRA repair parent SystemMapping",
        _MAPPING_SCHEMA,
    )
    evidence = _validate_artifact_reference(
        value["pre_repair_evidence"],
        "CGRA pre-repair Evidence",
        _EVALUATION_EVIDENCE_SCHEMA,
    )
    termination = CgraTransportRepairTermination(value["termination"])
    attempts = value["attempts"]
    if not isinstance(attempts, list):
        raise ValueError("CGRA transport repair attempts are not a list")
    current = initial_mapping
    retired = False
    for ordinal, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping) or set(attempt) != {
            "parent_spatial_mapping",
            "runtime_evidence",
            "constraint_set",
            "child_spatial_mapping",
            "child_evidence",
            "repair_kind",
            "solver_calls",
            "logical_solver_calls",
            "action_count",
            "retired",
        }:
            raise ValueError("CGRA transport repair attempt has the wrong shape")
        if attempt["parent_spatial_mapping"] != current:
            raise ValueError("CGRA repair Mapping lineage is discontinuous")
        if attempt["runtime_evidence"] != evidence:
            raise ValueError("CGRA repair runtime Evidence lineage is discontinuous")
        _validate_artifact_reference(
            attempt["constraint_set"],
            "CGRA repair constraint",
            _MAPPING_CONSTRAINT_SET_SCHEMA,
        )
        kind = SpatialExactRepairKind(
            _nonnegative_integer(attempt["repair_kind"], "CGRA repair kind")
        )
        calls = _nonnegative_integer(
            attempt["solver_calls"], "CGRA repair solver calls"
        )
        logical_calls = _nonnegative_integer(
            attempt["logical_solver_calls"], "CGRA repair logical solver calls"
        )
        actions = _nonnegative_integer(attempt["action_count"], "CGRA repair actions")
        if calls > logical_calls:
            raise ValueError("CGRA repair solver calls exceed logical work")
        retired = attempt["retired"]
        if not isinstance(retired, bool):
            raise ValueError("CGRA repair retirement is not boolean")
        child = attempt["child_spatial_mapping"]
        child_evidence = attempt["child_evidence"]
        if child is None:
            if child_evidence is not None or retired or ordinal + 1 != len(attempts):
                raise ValueError("CGRA repair continued without a child Mapping")
        else:
            child = _validate_artifact_reference(
                child, "CGRA repair child Mapping", _MAPPING_SCHEMA
            )
            evidence = _validate_artifact_reference(
                child_evidence,
                "CGRA repair child Evidence",
                _EVALUATION_EVIDENCE_SCHEMA,
            )
            if (
                kind is not SpatialExactRepairKind.REPAIRED
                or actions == 0
                or child == current
            ):
                raise ValueError(
                    "CGRA repair child has no successful changed transition"
                )
            current = child
            if retired and ordinal + 1 != len(attempts):
                raise ValueError("CGRA retired repair is not terminal")
    if retired != (termination is CgraTransportRepairTermination.RETIRED):
        raise ValueError("CGRA repair termination disagrees with runtime retirement")
    return termination, current


def validate_cgra_hardware_search(value: object) -> bool:
    """Validate the shared hardware search before any workload is profiled."""
    fields = {
        "schema", "resolved_config", "initial_fabric", "fabric", "ready",
        "deadline_ns", "deadline_overrun_ns", "rounds", "phase_ledger",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("CGRA hardware search has the wrong shape")
    if value["schema"] != CGRA_HARDWARE_SEARCH_SCHEMA:
        raise ValueError("CGRA hardware search has the wrong schema")
    _validate_artifact_reference(
        value["resolved_config"], "hardware resolved config", _RESOLVED_CONFIG_SCHEMA
    )
    expected_fabric = _validate_artifact_reference(
        value["initial_fabric"], "initial hardware Fabric", _FABRIC_SCHEMA
    )
    _validate_artifact_reference(value["fabric"], "selected Fabric", _FABRIC_SCHEMA)
    deadline = _nonnegative_integer(value["deadline_ns"], "hardware deadline", positive=True)
    if deadline != timeout_seconds(Tier.FAST) * 1_000_000_000:
        raise ValueError("CGRA hardware search has a foreign deadline")
    overrun = _nonnegative_integer(value["deadline_overrun_ns"], "hardware overrun")
    if type(value["ready"]) is not bool or (value["ready"] and overrun):
        raise ValueError("CGRA hardware search has invalid readiness")
    rounds = value["rounds"]
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("CGRA hardware search has no rounds")
    _, operators = load_cgra_representative_operators()
    source_identities: list[dict[str, object]] | None = None
    all_mapped = False
    final_growth_completed = False
    for ordinal, round_ in enumerate(rounds):
        if not isinstance(round_, Mapping) or set(round_) != {
            "fabric", "evaluations", "hardware_growth"
        }:
            raise ValueError("CGRA hardware round has the wrong shape")
        if round_["fabric"] != expected_fabric:
            raise ValueError("CGRA hardware search breaks its Fabric lineage")
        evaluations = round_["evaluations"]
        if not isinstance(evaluations, list) or len(evaluations) != len(operators):
            raise ValueError("CGRA hardware search omitted part of the source suite")
        identities: list[dict[str, object]] = []
        feedbacks: list[str] = []
        all_mapped = True
        for evaluation, operator in zip(evaluations, operators):
            if not isinstance(evaluation, Mapping) or set(evaluation) != {
                "workload", "operator_id", "protocol_symbol", "canonical_dataflow",
                "replay_cases", "replay_case_occurrences",
                "tech_mapping_search", "owner_feedback",
            }:
                raise ValueError("CGRA hardware evaluation has the wrong shape")
            if (evaluation["workload"], evaluation["operator_id"],
                    evaluation["protocol_symbol"]) != (
                    operator.workload, operator.operator_id, operator.protocol_symbol):
                raise ValueError("CGRA hardware search changed the source inventory")
            _validate_artifact_reference(
                evaluation["canonical_dataflow"], "canonical Dataflow", _CANONICAL_DATAFLOW_SCHEMA
            )
            _validate_source_replay_cases(
                evaluation["replay_cases"], evaluation["replay_case_occurrences"]
            )
            identity = {key: item for key, item in evaluation.items()
                        if key not in {"tech_mapping_search", "owner_feedback"}}
            identities.append(identity)
            result = evaluation["tech_mapping_search"]
            outcome = validate_cgra_tech_mapping_result(result, require_completed=False)
            assert isinstance(result, Mapping)
            usable = bool(result["candidates"]) and (
                outcome == "completed" or (outcome == "incomplete" and
                result["incomplete_reason"] == "candidate_semantic_limit_reached")
            )
            all_mapped &= usable
            feedback = evaluation["owner_feedback"]
            if feedback is not None:
                if not isinstance(feedback, str) or not re.fullmatch(
                        r"(?:[0-9a-f]{2})+", feedback):
                    raise ValueError("CGRA hardware feedback is not canonical bytes")
                if not result["candidates"]:
                    feedbacks.append(feedback)
        if source_identities is not None and identities != source_identities:
            raise ValueError("CGRA hardware search changed a dynamic source case")
        source_identities = identities
        growth = round_["hardware_growth"]
        final_growth_completed = False
        if growth is None:
            if ordinal + 1 != len(rounds):
                raise ValueError("CGRA hardware search resumed without a child")
            continue
        if all_mapped or not isinstance(growth, Mapping) or set(growth) != {
                "owner_feedback", "canonical_config", "result"}:
            raise ValueError("CGRA hardware growth has an invalid boundary")
        if growth["owner_feedback"] not in feedbacks:
            raise ValueError("CGRA hardware growth has no observed source deficit")
        config = growth["canonical_config"]
        if not isinstance(config, str) or not re.fullmatch(r"(?:[0-9a-f]{2})+", config):
            raise ValueError("CGRA hardware growth has no canonical configuration")
        result = growth["result"]
        outcome = _validate_candidate_generator_result(
            result, ("decision_attempt",), "CGRA hardware growth",
            require_completed=False, candidate_schema=_FABRIC_SCHEMA,
        )
        assert isinstance(result, Mapping)
        if outcome != "completed":
            if ordinal + 1 != len(rounds):
                raise ValueError("CGRA hardware search resumed an incomplete rewrite")
            continue
        candidates = result["candidates"]
        if len(candidates) != 1 or candidates[0] == expected_fabric:
            raise ValueError("CGRA hardware growth did not publish one changed child")
        expected_fabric = candidates[0]
        final_growth_completed = True
    if value["fabric"] != expected_fabric:
        raise ValueError("CGRA selected hardware is outside its search lineage")
    if value["ready"] and (not all_mapped or final_growth_completed):
        raise ValueError("CGRA hardware readiness lacks the complete source suite")
    phases = value["phase_ledger"]
    if not isinstance(phases, list) or len(phases) != 1 or not isinstance(phases[0], Mapping):
        raise ValueError("CGRA hardware search has no complete phase accounting")
    phase = phases[0]
    if set(phase) != {"phase", "wall_nanoseconds", "process_cpu_nanoseconds"} or (
            phase["phase"] != "shared_hardware_search"):
        raise ValueError("CGRA hardware phase accounting has the wrong shape")
    _nonnegative_integer(phase["wall_nanoseconds"], "hardware wall time", positive=True)
    _nonnegative_integer(phase["process_cpu_nanoseconds"], "hardware CPU time")
    return value["ready"]


def validate_cgra_profile_outcome(value: object) -> tuple[str, str | None]:
    expected_fields = {
        "schema",
        "workload",
        "operator_id",
        "protocol_symbol",
        "canonical_dataflow",
        "source_replay_cases",
        "replay_case_occurrences",
        "stage",
        "resolved_config",
        "fabric",
        "tech_mapping_search",
        "spatial_pnr",
    }
    if not isinstance(value, Mapping):
        raise ValueError("CGRA profile outcome is not an object")
    if value.get("stage") == "transport_repair":
        expected_fields |= {
            "initial_spatial_mapping",
            "spatial_candidate_screening",
            "transport_repair",
            "phase_ledger",
            "replay_case_phase_ledger",
            "completed_replay_cases",
            "failed_replay_case",
        }
    if set(value) != expected_fields:
        raise ValueError("CGRA profile outcome has the wrong shape")
    if value["schema"] != CGRA_PROFILE_OUTCOME_SCHEMA:
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
    _validate_artifact_reference(
        value["canonical_dataflow"], "canonical Dataflow", _CANONICAL_DATAFLOW_SCHEMA
    )
    source_inputs = _validate_source_replay_cases(
        value["source_replay_cases"], value["replay_case_occurrences"]
    )
    tech_outcome = validate_cgra_tech_mapping_result(
        value["tech_mapping_search"], require_completed=False
    )
    stage = value["stage"]
    if stage == "tech_mapping":
        tech_result = value["tech_mapping_search"]
        assert isinstance(tech_result, Mapping)
        if value["spatial_pnr"] is not None or tech_result["candidates"]:
            raise ValueError("CGRA TechMapping outcome has an invalid boundary")
        result = value["tech_mapping_search"]
    elif stage in {"spatial_pnr", "transport_repair"}:
        tech_result = value["tech_mapping_search"]
        assert isinstance(tech_result, Mapping)
        if (
            not tech_result["candidates"]
            or tech_outcome
            not in {
                "completed",
                "incomplete",
            }
            or (
                tech_outcome == "incomplete"
                and tech_result["incomplete_reason"]
                != "candidate_semantic_limit_reached"
            )
        ):
            raise ValueError("CGRA Spatial PnR ran after unusable TechMapping")
        pnr_outcome = validate_cgra_pnr_result(
            value["spatial_pnr"], require_completed=False
        )
        pnr_result = value["spatial_pnr"]
        assert isinstance(pnr_result, Mapping)
        if stage == "transport_repair":
            initial = _validate_artifact_reference(
                value["initial_spatial_mapping"],
                "CGRA initial Mapping",
                _MAPPING_SCHEMA,
            )
            if initial not in pnr_result["candidates"]:
                raise ValueError("CGRA repair parent is absent from its PnR result")
            _validate_cgra_screening(
                value["spatial_candidate_screening"], pnr_result["candidates"]
            )
            _validate_cgra_phases(value["phase_ledger"])
            _validate_cgra_phases(value["replay_case_phase_ledger"])
            completed = value["completed_replay_cases"]
            if not isinstance(completed, list) or len(completed) >= len(source_inputs):
                raise ValueError("CGRA stopped profile has no remaining source input")
            for replay, source_input in zip(completed, source_inputs):
                _validate_cgra_replay_case(replay, value)
                if replay["input"] != source_input:
                    raise ValueError("CGRA stopped profile changed a completed source input")
            if value["failed_replay_case"] != source_inputs[len(completed)]:
                raise ValueError("CGRA stopped profile lost its failing source input")
            termination, _ = _validate_transport_repair(
                value["transport_repair"], initial
            )
            if termination is CgraTransportRepairTermination.RETIRED:
                raise ValueError("CGRA incomplete outcome contains a retired repair")
            # Completed static PnR cannot describe a non-retiring runtime.
            return "incomplete", termination.value
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


def _validate_cgra_replay_case(
    replay: object, profile: Mapping[str, object]
) -> None:
    expected_fields = {
        "input", "tech_mapping", "initial_spatial_mapping", "spatial_mapping",
        "spatial_candidate_screening", "transport_repair", "warmup_evidence",
        "measurements", "phase_ledger",
    }
    if not isinstance(replay, Mapping) or set(replay) != expected_fields:
        raise ValueError("CGRA replay case has the wrong shape")
    _validate_replay_input(replay["input"])
    for field, schema in (
        ("tech_mapping", _MAPPING_SCHEMA),
        ("initial_spatial_mapping", _MAPPING_SCHEMA),
        ("spatial_mapping", _MAPPING_SCHEMA),
        ("warmup_evidence", _EVALUATION_EVIDENCE_SCHEMA),
    ):
        _validate_artifact_reference(replay[field], f"CGRA replay {field}", schema)
    tech_search = profile["tech_mapping_search"]
    initial_pnr = profile["spatial_pnr"]
    assert isinstance(tech_search, Mapping) and isinstance(initial_pnr, Mapping)
    if replay["tech_mapping"] not in tech_search["candidates"]:
        raise ValueError("selected TechMapping is absent from the complete search")
    _validate_cgra_screening(replay["spatial_candidate_screening"], initial_pnr["candidates"])
    _validate_cgra_phases(replay["phase_ledger"])
    if replay["initial_spatial_mapping"] not in initial_pnr["candidates"]:
        raise ValueError("CGRA initial Mapping is absent from its PnR result")
    transport_repair = replay["transport_repair"]
    if transport_repair is None:
        if replay["spatial_mapping"] != replay["initial_spatial_mapping"]:
            raise ValueError("CGRA final Mapping has no repair lineage")
    else:
        termination, final_mapping = _validate_transport_repair(
            transport_repair, replay["initial_spatial_mapping"]
        )
        if termination is not CgraTransportRepairTermination.RETIRED:
            raise ValueError("CGRA profile contains a non-retiring transport repair")
        if replay["spatial_mapping"] != final_mapping:
            raise ValueError("CGRA final Mapping disagrees with repair receipt")
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
    measurements = replay["measurements"]
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
        "schema", "workload", "operator_id", "protocol_symbol",
        "qualification_limit_nanoseconds", "warmup_runs", "measurement_runs",
        "batch_peak_resident_bytes", "canonical_dataflow", "resolved_config",
        "fabric", "tech_mapping_search", "spatial_pnr", "phase_ledger",
        "source_replay_cases", "replay_case_occurrences", "replay_cases",
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
        if profile["schema"] != CGRA_PROFILE_SCHEMA:
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
            profile["qualification_limit_nanoseconds"], "CGRA qualification limit", positive=True
        )
        if qualification_limit != CGRA_QUALIFICATION_LIMIT_NANOSECONDS:
            raise ValueError("CGRA profile used a foreign qualification limit")
        if (
            _nonnegative_integer(profile["warmup_runs"], "CGRA warmup count")
            != CGRA_QUALIFICATION_WARMUP_RUNS
            or _nonnegative_integer(profile["measurement_runs"], "CGRA measurement count")
            != CGRA_QUALIFICATION_MEASUREMENT_RUNS
        ):
            raise ValueError("CGRA profile used a foreign sampling protocol")
        for field, schema in (
            ("canonical_dataflow", _CANONICAL_DATAFLOW_SCHEMA),
            ("resolved_config", _RESOLVED_CONFIG_SCHEMA),
            ("fabric", _FABRIC_SCHEMA),
        ):
            _validate_artifact_reference(profile[field], f"CGRA profile {field}", schema)
        resolved_config_identities.add(str(profile["resolved_config"]["artifact"]))
        fabric_identities.add(str(profile["fabric"]["artifact"]))
        dataflow_identities.add(str(profile["canonical_dataflow"]["artifact"]))
        tech_outcome = validate_cgra_tech_mapping_result(
            profile["tech_mapping_search"], require_completed=False
        )
        tech_search = profile["tech_mapping_search"]
        assert isinstance(tech_search, Mapping)
        if tech_outcome not in {"completed", "incomplete"} or (
            tech_outcome == "incomplete"
            and tech_search["incomplete_reason"] != "candidate_semantic_limit_reached"
        ):
            raise ValueError("CGRA profile has no usable TechMapping frontier")
        validate_cgra_pnr_result(profile["spatial_pnr"], require_completed=True)
        _validate_cgra_phases(profile["phase_ledger"])
        _nonnegative_integer(
            profile["batch_peak_resident_bytes"], "CGRA workload-batch peak resident memory",
            positive=True,
        )
        inputs = _validate_source_replay_cases(
            profile["source_replay_cases"], profile["replay_case_occurrences"]
        )
        replays = profile["replay_cases"]
        if not isinstance(replays, list) or len(replays) != len(inputs):
            raise ValueError("CGRA profile does not cover every source replay input")
        for replay, source_input in zip(replays, inputs):
            _validate_cgra_replay_case(replay, profile)
            if replay["input"] != source_input:
                raise ValueError("CGRA profile changed a source replay input")
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
        replays = profile["replay_cases"]
        assert isinstance(replays, list)
        for replay in replays:
            assert isinstance(replay, Mapping)
            measurements = replay["measurements"]
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


def resolve_cgra_gate_configuration() -> CgraGateConfiguration:
    """The Spatial absolute budget owner for one paired conformance run.

    The tracked gate is the only published authority. Until the ten-profile
    qualification succeeds there is no published value, so the canonical
    medium tier stands in under an explicit provisional source that every
    report carries; no second tracked value is introduced.
    """
    if CGRA_GATE_CONFIGURATION.is_file():
        return load_cgra_gate_configuration()
    operator_gate_sha256, _ = load_cgra_representative_operators()
    return CgraGateConfiguration(
        int(timeout_seconds(Tier.MEDIUM)) * 1_000_000_000,
        "",
        operator_gate_sha256,
        (),
        CgraGateSource.PROVISIONAL_BOOTSTRAP,
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
    if root["schema"] != CGRA_GATE_SCHEMA:
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
