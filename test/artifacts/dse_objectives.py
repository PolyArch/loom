#!/usr/bin/env python3
"""Shared DSE objective metadata for generators and artifact audits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DseObjectiveSpec:
    direction: str
    units: str
    ordering_rule: str
    metric_entity: str
    metric_name: str


OBJECTIVE_SPECS = {
    "minimize_runtime": DseObjectiveSpec(
        direction="minimize",
        units="cycles",
        ordering_rule="runtime_score_then_candidate_id",
        metric_entity="workload",
        metric_name="cgra_sim_cycles",
    ),
    "maximize_throughput": DseObjectiveSpec(
        direction="maximize",
        units="items_per_s",
        ordering_rule="throughput_score_then_candidate_id",
        metric_entity="workload",
        metric_name="throughput_items_per_s",
    ),
    "maximize_performance_per_watt": DseObjectiveSpec(
        direction="maximize",
        units="items_per_s_per_w",
        ordering_rule="performance_per_watt_score_then_candidate_id",
        metric_entity="workload",
        metric_name="performance_per_watt",
    ),
    "maximize_performance_per_area": DseObjectiveSpec(
        direction="maximize",
        units="items_per_s_per_um2",
        ordering_rule="performance_per_area_score_then_candidate_id",
        metric_entity="workload",
        metric_name="performance_per_area",
    ),
    "minimize_area": DseObjectiveSpec(
        direction="minimize",
        units="um2",
        ordering_rule="area_score_then_candidate_id",
        metric_entity="hardware",
        metric_name="area_um2",
    ),
    "minimize_energy": DseObjectiveSpec(
        direction="minimize",
        units="nJ",
        ordering_rule="energy_score_then_candidate_id",
        metric_entity="workload",
        metric_name="energy_nj",
    ),
    "minimize_power": DseObjectiveSpec(
        direction="minimize",
        units="nJ",
        ordering_rule="energy_score_then_candidate_id",
        metric_entity="workload",
        metric_name="energy_nj",
    ),
}


def known_objective_specs() -> dict[str, DseObjectiveSpec]:
    return dict(OBJECTIVE_SPECS)


def objective_spec(objective: str) -> DseObjectiveSpec | None:
    return OBJECTIVE_SPECS.get(objective)


def policy_id_for_objective(objective: str) -> str:
    return f"deterministic_{objective}_v1"


def ordering_rule_for_objective(objective: str) -> str:
    spec = objective_spec(objective)
    if spec is not None:
        return spec.ordering_rule
    return OBJECTIVE_SPECS["minimize_runtime"].ordering_rule


def objective_semantics(objective: str) -> tuple[str, str] | None:
    spec = objective_spec(objective)
    if spec is None:
        return None
    return spec.direction, spec.units


def metric_id_for_objective(objective: str, workload: str, hardware: str) -> str | None:
    spec = objective_spec(objective)
    if spec is None:
        return None
    entity = workload if spec.metric_entity == "workload" else hardware
    if not entity:
        return None
    return f"metric::{entity}::{spec.metric_name}"
