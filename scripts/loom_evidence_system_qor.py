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
_LAUNCH_OVERHEAD_TARGET = _system_qor_target("applicationMaximumLaunchOverhead")

# Evaluation-tier spellings are owned by EvaluationTier in
# include/Application/Manifest.h. Only a qualified row carries the saturation
# target; a functional row publishes the same measurements without it.
FUNCTIONAL_TIER = "functional"
QUALIFIED_TIER = "qualified"
EVALUATION_TIERS = (FUNCTIONAL_TIER, QUALIFIED_TIER)


def _ratio(value: Any, expected: Fraction) -> bool:
    return value == {
        "numerator": expected.numerator,
        "denominator": expected.denominator,
    }


def validate_system_qor(
    workspace: dict[str, Any], declared_tier: str | None
) -> list[str]:
    """Validate the Application owner's post-execution projection and root joins.

    `declared_tier` is the Application manifest's own evaluation tier for the
    selected row, or None when no portfolio row was selected. The projection
    carries the tier its Deployment was built with; the manifest owner only
    cross-checks it and never supplies a competing target.
    """
    qor = workspace.get("paired_system_execution")
    if not isinstance(qor, dict) or set(qor) != {
        "schema",
        "version",
        "evaluation_tier",
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
    tier = qor["evaluation_tier"]
    if tier not in EVALUATION_TIERS:
        return errors + ["system_qor_evaluation_tier_invalid"]
    if declared_tier is not None and tier != declared_tier:
        errors.append("system_qor_evaluation_tier_mismatch")
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
        "maximum_launch_overhead_exclusive": {
            "numerator": _LAUNCH_OVERHEAD_TARGET.numerator,
            "denominator": _LAUNCH_OVERHEAD_TARGET.denominator,
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
        if declared_tier is not None:
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
    memory_branch, compute_branch, window_ticks, accelerated, overhead = branches
    speedup = Fraction(host[1], window_ticks)
    if not _ratio(qor["speedup"], speedup):
        errors.append("system_qor_speedup_mismatch")
    launched = candidate_interval["compute"]["launched_acc_cores"] != 0
    launch_bound = overhead >= _LAUNCH_OVERHEAD_TARGET
    qualifies = (
        speedup > 1
        and launched
        and not launch_bound
        and (
            memory_branch > _UTILIZATION_TARGET or compute_branch > _UTILIZATION_TARGET
        )
    )
    expected_status = (
        FUNCTIONAL_TIER
        if tier == FUNCTIONAL_TIER
        else ("qualified" if qualifies else "not_qualified")
    )
    if qor["status"] != expected_status:
        errors.append("system_qor_status_mismatch")
    if launch_bound:
        bottleneck = "launch_bound"
    elif memory_branch > _UTILIZATION_TARGET:
        bottleneck = "memory_bandwidth_bound"
    elif compute_branch > _UTILIZATION_TARGET:
        bottleneck = "compute_bound"
    elif Fraction(accelerated, window_ticks) < _HOST_BOUND_TARGET:
        bottleneck = "host_bound"
    else:
        bottleneck = "latency_bound"
    if qor["bottleneck"] != bottleneck:
        errors.append("system_qor_bottleneck_mismatch")
    if tier == QUALIFIED_TIER and not qualifies:
        errors.append("system_qor_performance_target_not_met")
    return errors


def _validate_system_qor_interval(
    window: Any, program_ticks: int, candidate: bool, errors: list[str]
) -> tuple[Fraction, int] | None:
    fields = {"begin_tick", "end_tick", "elapsed_ticks", "shared_memory"}
    if candidate:
        fields |= {"accelerated_window", "compute"}
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


def _validate_system_qor_phase(
    phase: Any, errors: list[str]
) -> tuple[int, int, int] | None:
    if not isinstance(phase, dict) or set(phase) != {
        "begin_tick",
        "end_tick",
        "elapsed_ticks",
    }:
        errors.append("system_qor_accelerated_phase_shape_invalid")
        return None
    begin = _integer(phase["begin_tick"])
    end = _integer(phase["end_tick"])
    span = _integer(phase["elapsed_ticks"])
    if (
        begin is None
        or end is None
        or span is None
        or begin < 0
        or end < begin
        or end - begin != span
    ):
        errors.append("system_qor_accelerated_phase_invalid")
        return None
    return begin, end, span


def _validate_system_qor_window(
    window: Any, program_ticks: int, errors: list[str]
) -> tuple[Fraction, Fraction, int, int, Fraction] | None:
    """Recompute saturation over the invocation phase of the accelerated window.

    Configuration residency moves the binary configuration image, which the
    service observer never counts as application data, so charging its ticks to
    the saturation denominator would credit a fat configuration image as memory
    appetite. It is reported as the launch overhead of the whole window instead.
    """
    measured = _validate_system_qor_interval(window, program_ticks, True, errors)
    if measured is None:
        return None
    _, span = measured
    accelerated_window = window["accelerated_window"]
    if not isinstance(accelerated_window, dict) or set(accelerated_window) != {
        "configuration_residency",
        "invocation",
        "elapsed_ticks",
        "launch_overhead",
        "shared_memory",
    }:
        errors.append("system_qor_accelerated_window_shape_invalid")
        return None
    residency = _validate_system_qor_phase(
        accelerated_window["configuration_residency"], errors
    )
    invocation = _validate_system_qor_phase(accelerated_window["invocation"], errors)
    accelerated = _integer(accelerated_window["elapsed_ticks"])
    if residency is None or invocation is None or accelerated is None:
        return None
    if (
        invocation[0] < residency[0]
        or invocation[1] < residency[1]
        or invocation[1] - residency[0] != accelerated
        or accelerated > span
    ):
        errors.append("system_qor_accelerated_window_invalid")
        return None
    memory = accelerated_window["shared_memory"]
    if not isinstance(memory, dict) or set(memory) != {"occupied_ticks", "utilization"}:
        errors.append("system_qor_accelerated_window_shape_invalid")
        return None
    busy = _integer(memory["occupied_ticks"])
    if busy is None or busy < 0 or busy > invocation[2]:
        errors.append("system_qor_accelerated_window_occupancy_invalid")
        return None
    memory_branch = Fraction(busy, invocation[2]) if invocation[2] else Fraction(0)
    if not _ratio(memory["utilization"], memory_branch):
        errors.append("system_qor_accelerated_window_utilization_mismatch")
    overhead = Fraction(residency[2], accelerated) if accelerated else Fraction(0)
    if not _ratio(accelerated_window["launch_overhead"], overhead):
        errors.append("system_qor_launch_overhead_mismatch")
    compute = window["compute"]
    if not isinstance(compute, dict) or set(compute) != {
        "launched_acc_cores",
        "reference_cycle_ticks",
        "classes",
        "occupancy",
        "binding_class",
        "placement_utilization",
    }:
        errors.append("system_qor_candidate_window_shape_invalid")
        return None
    cores = _integer(compute["launched_acc_cores"])
    period = _integer(compute["reference_cycle_ticks"])
    classes = compute["classes"]
    if (
        cores is None
        or cores < 0
        or period is None
        or period < 0
        or (cores > 0 and period == 0)
        or not isinstance(classes, list)
    ):
        errors.append("system_qor_candidate_compute_invalid")
        return None
    # Each class is measured against its own speed of light: the element lanes
    # every launched Fabric could have issued for it across the invocation
    # phase. A Temporal PE's FU issues once per cycle; its resident contexts are
    # placement slots, which explain a mapping and never gate it.
    compute_branch = Fraction(0)
    placement = Fraction(0)
    binding = None
    seen: set[tuple[str, int]] = set()
    for entry in classes:
        if not isinstance(entry, dict) or set(entry) != {
            "schema",
            "element_bits",
            "retired_element_firings",
            "peak_issue_lanes_per_cycle",
            "placement_slots",
            "bound_realizations",
            "occupancy",
            "placement_utilization",
        }:
            errors.append("system_qor_candidate_compute_class_shape_invalid")
            return None
        bits = _integer(entry["element_bits"])
        firings = _integer(entry["retired_element_firings"])
        peak = _integer(entry["peak_issue_lanes_per_cycle"])
        slots = _integer(entry["placement_slots"])
        bound = _integer(entry["bound_realizations"])
        schema = entry["schema"]
        if (
            not isinstance(schema, str)
            or bits is None
            or bits <= 0
            or firings is None
            or firings < 0
            or peak is None
            or peak < 0
            or slots is None
            or slots < 0
            or bound is None
            or bound < 0
            or (cores == 0 and firings != 0)
            or (firings != 0 and peak == 0)
            or (schema, bits) in seen
        ):
            errors.append("system_qor_candidate_compute_class_invalid")
            return None
        seen.add((schema, bits))
        capacity = invocation[2] * peak * cores
        occupancy = Fraction(firings * period, capacity) if capacity else Fraction(0)
        if not _ratio(entry["occupancy"], occupancy):
            errors.append("system_qor_candidate_compute_occupancy_mismatch")
        slot_capacity = slots * cores
        utilization = Fraction(bound, slot_capacity) if slot_capacity else Fraction(0)
        if not _ratio(entry["placement_utilization"], utilization):
            errors.append("system_qor_candidate_placement_utilization_mismatch")
        if occupancy > compute_branch:
            compute_branch = occupancy
            binding = schema
        placement = max(placement, utilization)
    if not _ratio(compute["occupancy"], compute_branch):
        errors.append("system_qor_candidate_compute_occupancy_mismatch")
    if compute["binding_class"] != binding:
        errors.append("system_qor_candidate_binding_class_mismatch")
    if not _ratio(compute["placement_utilization"], placement):
        errors.append("system_qor_candidate_placement_utilization_mismatch")
    if (cores != 0) != (accelerated != 0):
        errors.append("system_qor_accelerated_window_launch_mismatch")
    return memory_branch, compute_branch, span, accelerated, overhead
