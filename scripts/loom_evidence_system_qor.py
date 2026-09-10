"""Validate measured System computation and its exact CPU/candidate evidence join."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import re
from typing import Any

from scripts.loom_evidence_projection import (
    artifact_root_reference as _root_reference,
    integer_value as _integer,
    owned_projection_literal,
)

_ROOT = Path(__file__).resolve().parents[1]
_SYSTEM_QOR_OWNER = (_ROOT / "include/Application/SystemQor.h").read_text(
    encoding="utf-8"
)
SYSTEM_QOR_SCHEMA = owned_projection_literal(
    "applicationSystemQorProjectionSchema", _SYSTEM_QOR_OWNER
)
SYSTEM_QOR_VERSION = owned_projection_literal(
    "applicationSystemQorProjectionVersion", _SYSTEM_QOR_OWNER
)


def _system_qor_target(prefix: str) -> Fraction:
    return Fraction(
        *[
            int(re.search(rf"\b{prefix}{part} = (\d+)", _SYSTEM_QOR_OWNER).group(1))
            for part in ("Numerator", "Denominator")
        ]
    )


_UTILIZATION_TARGET = _system_qor_target("applicationMinimumResourceUtilization")
_HOST_BOUND_TARGET = _system_qor_target("applicationHostBoundWindow")


def _ratio(value: Any, expected: Fraction) -> bool:
    return value == {
        "numerator": expected.numerator,
        "denominator": expected.denominator,
    }


def validate_system_qor(workspace: dict[str, Any], require_target: bool) -> list[str]:
    """Validate the Application owner's post-execution projection and root joins."""
    qor = workspace.get("paired_system_execution")
    if not isinstance(qor, dict) or set(qor) != {
        "schema",
        "version",
        "application_runtime_manifest",
        "gem5_binding",
        "host_only",
        "candidate",
        "speedup",
        "status",
        "bottleneck",
        "target",
    }:
        return ["system_qor_projection_missing_or_malformed"]
    errors: list[str] = []
    if qor["schema"] != SYSTEM_QOR_SCHEMA or qor["version"] != SYSTEM_QOR_VERSION:
        errors.append("system_qor_schema_invalid")
    for field in ("application_runtime_manifest", "gem5_binding"):
        if _root_reference(qor[field]) is None or qor[field] != workspace.get(field):
            errors.append(f"system_qor_{field}_mismatch")
    product = workspace.get("product_profile") is not None
    elapsed: dict[str, int] = {}
    for role in ("host_only", "candidate"):
        run = qor[role]
        fields = {
            "request",
            "evidence",
            "execution",
            "elapsed_ticks",
            "shared_memory",
            "computation_interval",
        }
        if product:
            fields |= {"product_oracle_request", "product_oracle_evidence"}
        if not isinstance(run, dict) or set(run) != fields:
            errors.append(f"system_qor_{role}_shape_invalid")
            continue
        roots = [
            ("request", "evaluation.request"),
            ("evidence", "evaluation.evidence"),
            ("execution", "loom.simulation_execution"),
        ]
        if product:
            roots += [
                ("product_oracle_request", "evaluation.request"),
                ("product_oracle_evidence", "evaluation.evidence"),
            ]
        for field, schema in roots:
            if _root_reference(run[field], schema) is None:
                errors.append(f"system_qor_{role}_{field}_invalid")
        duration = _integer(run["elapsed_ticks"])
        memory = run["shared_memory"]
        if (
            duration is None
            or duration <= 0
            or not isinstance(memory, dict)
            or set(memory) != {"occupied_ticks", "utilization"}
        ):
            errors.append(f"system_qor_{role}_window_invalid")
            continue
        busy = _integer(memory["occupied_ticks"])
        if busy is None or busy < 0 or busy > duration:
            errors.append(f"system_qor_{role}_occupancy_invalid")
            continue
        if not _ratio(memory["utilization"], Fraction(busy, duration)):
            errors.append(f"system_qor_{role}_utilization_mismatch")
        elapsed[role] = duration
    if len(elapsed) != 2:
        return errors
    if any(
        qor["host_only"][field] == qor["candidate"][field]
        for field in ("request", "evidence", "execution")
    ):
        errors.append("system_qor_host_candidate_identity_alias")
    runs = workspace.get("runs")
    candidates = (
        [
            run
            for run in runs
            if isinstance(run, dict)
            and run.get("scope") == "system"
            and run.get("engine") == "cgra"
        ]
        if isinstance(runs, list)
        else []
    )
    if len(candidates) != 1 or any(
        candidates[0].get(field) != qor["candidate"][field] for field, _ in roots
    ):
        errors.append("system_qor_candidate_run_join_invalid")
    if qor["target"] != {
        "strict_speedup": True,
        "window_branches": ["memory_service_utilization", "compute_occupancy"],
        "minimum_window_utilization_exclusive": {
            "numerator": _UTILIZATION_TARGET.numerator,
            "denominator": _UTILIZATION_TARGET.denominator,
        },
        "host_bound_window_fraction_exclusive": {
            "numerator": _HOST_BOUND_TARGET.numerator,
            "denominator": _HOST_BOUND_TARGET.denominator,
        },
    }:
        errors.append("system_qor_target_mismatch")
    host_interval = qor["host_only"]["computation_interval"]
    candidate_interval = qor["candidate"]["computation_interval"]
    if host_interval is None or candidate_interval is None:
        if host_interval is not None or candidate_interval is not None:
            errors.append("system_qor_computation_boundary_mismatch")
        if (
            qor["speedup"] is not None
            or qor["status"] != "unmeasured"
            or qor["bottleneck"] != "unmeasured"
        ):
            errors.append("system_qor_unmeasured_projection_invalid")
        if require_target:
            errors.append("system_qor_computation_unmeasured")
        return errors
    host = _validate_system_qor_interval(
        host_interval, elapsed["host_only"], False, errors
    )
    branches = _validate_system_qor_window(
        candidate_interval, elapsed["candidate"], errors
    )
    if host is None or branches is None:
        return errors
    memory_branch, compute_branch, window_ticks = branches
    speedup = Fraction(host[1], window_ticks)
    if not _ratio(qor["speedup"], speedup):
        errors.append("system_qor_speedup_mismatch")
    launched = candidate_interval["compute"]["launched_acc_cores"] != 0
    qualifies = (
        speedup > 1
        and launched
        and (
            memory_branch > _UTILIZATION_TARGET or compute_branch > _UTILIZATION_TARGET
        )
    )
    if qor["status"] != ("qualified" if qualifies else "not_qualified"):
        errors.append("system_qor_status_mismatch")
    if memory_branch > _UTILIZATION_TARGET:
        bottleneck = "memory_bandwidth_bound"
    elif compute_branch > _UTILIZATION_TARGET:
        bottleneck = "compute_bound"
    elif (
        Fraction(candidate_interval["accelerated_ticks"], window_ticks)
        < _HOST_BOUND_TARGET
    ):
        bottleneck = "host_bound"
    else:
        bottleneck = "latency_bound"
    if qor["bottleneck"] != bottleneck:
        errors.append("system_qor_bottleneck_mismatch")
    if require_target and not qualifies:
        errors.append("system_qor_performance_target_not_met")
    return errors


def _validate_system_qor_interval(
    window: Any, program_ticks: int, candidate: bool, errors: list[str]
) -> tuple[Fraction, int] | None:
    fields = {"begin_tick", "end_tick", "elapsed_ticks", "shared_memory"}
    if candidate:
        fields |= {"accelerated_ticks", "compute"}
    if not isinstance(window, dict) or set(window) != fields:
        errors.append("system_qor_computation_shape_invalid")
        return None
    start = _integer(window["begin_tick"])
    completion = _integer(window["end_tick"])
    span = _integer(window["elapsed_ticks"])
    if (
        start is None
        or completion is None
        or span is None
        or start < 0
        or completion - start != span
        or span <= 0
        or span > program_ticks
    ):
        errors.append("system_qor_computation_interval_invalid")
        return None
    memory = window["shared_memory"]
    if not isinstance(memory, dict) or set(memory) != {"occupied_ticks", "utilization"}:
        errors.append("system_qor_computation_shape_invalid")
        return None
    busy = _integer(memory["occupied_ticks"])
    if busy is None or busy < 0 or busy > span:
        errors.append("system_qor_computation_occupancy_invalid")
        return None
    memory_branch = Fraction(busy, span)
    if not _ratio(memory["utilization"], memory_branch):
        errors.append("system_qor_computation_utilization_mismatch")
    return memory_branch, span


def _validate_system_qor_window(
    window: Any, program_ticks: int, errors: list[str]
) -> tuple[Fraction, Fraction, int] | None:
    """Recompute occupancy over the same source-declared computation interval."""
    measured = _validate_system_qor_interval(window, program_ticks, True, errors)
    if measured is None:
        return None
    memory_branch, span = measured
    accelerated = _integer(window["accelerated_ticks"])
    if accelerated is None or accelerated < 0 or accelerated > span:
        errors.append("system_qor_accelerated_interval_invalid")
        return None
    compute = window["compute"]
    if not isinstance(compute, dict) or set(compute) != {
        "retired_compute_firings",
        "mapped_compute_units",
        "launched_acc_cores",
        "reference_cycle_ticks",
        "occupancy",
    }:
        errors.append("system_qor_candidate_window_shape_invalid")
        return None
    firings = _integer(compute["retired_compute_firings"])
    units = _integer(compute["mapped_compute_units"])
    cores = _integer(compute["launched_acc_cores"])
    period = _integer(compute["reference_cycle_ticks"])
    if (
        firings is None
        or firings < 0
        or units is None
        or units < 0
        or cores is None
        or cores < 0
        or period is None
        or period < 0
        or (cores > 0 and (units == 0 or period == 0))
        or (cores == 0 and firings != 0)
    ):
        errors.append("system_qor_candidate_compute_invalid")
        return None
    # One retired compute firing occupies its bound unit for one reference cycle.
    compute_branch = (
        Fraction(firings * period, span * units * cores) if cores else Fraction(0)
    )
    if not _ratio(compute["occupancy"], compute_branch):
        errors.append("system_qor_candidate_compute_occupancy_mismatch")
    return memory_branch, compute_branch, span
