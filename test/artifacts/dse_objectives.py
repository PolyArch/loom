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
    "minimize_dynamic_power": DseObjectiveSpec(
        direction="minimize",
        units="mW",
        ordering_rule="dynamic_power_score_then_candidate_id",
        metric_entity="hardware",
        metric_name="dynamic_power_mw",
    ),
    "minimize_leakage_power": DseObjectiveSpec(
        direction="minimize",
        units="mW",
        ordering_rule="leakage_power_score_then_candidate_id",
        metric_entity="hardware",
        metric_name="leakage_power_mw",
    ),
    "minimize_unsupported_scope_diagnostics": DseObjectiveSpec(
        direction="minimize",
        units="count",
        ordering_rule="unsupported_scope_diagnostics_score_then_candidate_id",
        metric_entity="candidate",
        metric_name="unsupported_scope_diagnostics_count",
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


def require_objective_spec(objective: str) -> DseObjectiveSpec:
    spec = objective_spec(objective)
    if spec is None:
        raise ValueError(f"config_unknown_objective: {objective}")
    return spec


def policy_id_for_objective(objective: str) -> str:
    require_objective_spec(objective)
    return f"deterministic_{objective}_v1"


def ordering_rule_for_objective(objective: str) -> str:
    return require_objective_spec(objective).ordering_rule


def objective_semantics(objective: str) -> tuple[str, str] | None:
    spec = objective_spec(objective)
    if spec is None:
        return None
    return spec.direction, spec.units


def metric_id_for_objective(
    objective: str,
    workload: str,
    hardware: str,
    mapping_id: str = "",
) -> str | None:
    spec = objective_spec(objective)
    if spec is None:
        return None
    if spec.metric_entity == "workload":
        if not workload:
            return None
        return f"metric::{workload}::{spec.metric_name}"
    if spec.metric_entity == "hardware":
        if not hardware:
            return None
        return f"metric::{hardware}::{spec.metric_name}"
    if spec.metric_entity == "candidate":
        if not workload or not hardware or not mapping_id:
            return None
        return f"metric::{workload}::{hardware}::{mapping_id}::{spec.metric_name}"
    entity = workload or hardware
    if not entity:
        return None
    return f"metric::{entity}::{spec.metric_name}"
