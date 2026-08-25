"""Shared conformance policy for paired Spatial and System simulations."""

from __future__ import annotations

import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.timeout_budgets import Tier, seconds as timeout_seconds  # noqa: E402


SPATIAL_REFERENCE_FLOOR_SECONDS = 0.1
SYSTEM_BUDGET_MULTIPLIER = 3.0
HARD_FAILURE_RATIO = 10.0
REFERENCE_RATE_TARGET_HZ = 100_000.0
DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS = float(timeout_seconds(Tier.FAST))
CGRA_SPATIAL_BOOTSTRAP_BUDGET_SECONDS = float(timeout_seconds(Tier.MEDIUM))
RESERVED_DEVELOPMENT_CPUS = 4
MAX_OUTER_WORKERS = 120


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
    rate = system_timing.reference_cycles / system_timing.active_wall_seconds
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
