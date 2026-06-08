#!/usr/bin/env python3
"""Regression test for shared DSE objective metadata."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))
sys.path.insert(0, str(ROOT / "test" / "dse"))

import dse_objectives  # noqa: E402
import intermediate_artifacts  # noqa: E402
import candidate_summary  # noqa: E402


EXPECTED = {
    "minimize_runtime": ("minimize", "cycles", "runtime_score_then_candidate_id", "workload", "cgra_sim_cycles"),
    "maximize_throughput": ("maximize", "items_per_s", "throughput_score_then_candidate_id", "workload", "throughput_items_per_s"),
    "maximize_performance_per_watt": (
        "maximize",
        "items_per_s_per_w",
        "performance_per_watt_score_then_candidate_id",
        "workload",
        "performance_per_watt",
    ),
    "maximize_performance_per_area": (
        "maximize",
        "items_per_s_per_um2",
        "performance_per_area_score_then_candidate_id",
        "workload",
        "performance_per_area",
    ),
    "minimize_area": ("minimize", "um2", "area_score_then_candidate_id", "hardware", "area_um2"),
    "minimize_dynamic_power": (
        "minimize",
        "mW",
        "dynamic_power_score_then_candidate_id",
        "hardware",
        "dynamic_power_mw",
    ),
    "minimize_leakage_power": (
        "minimize",
        "mW",
        "leakage_power_score_then_candidate_id",
        "hardware",
        "leakage_power_mw",
    ),
    "minimize_unsupported_scope_diagnostics": (
        "minimize",
        "count",
        "unsupported_scope_diagnostics_score_then_candidate_id",
        "candidate",
        "unsupported_scope_diagnostics_count",
    ),
    "minimize_energy": ("minimize", "nJ", "energy_score_then_candidate_id", "workload", "energy_nj"),
    "minimize_power": ("minimize", "nJ", "energy_score_then_candidate_id", "workload", "energy_nj"),
}


def main() -> int:
    known = dse_objectives.known_objective_specs()
    if set(known) != set(EXPECTED):
        raise AssertionError(f"unexpected objective spec set: {sorted(known)}")
    for objective, (direction, units, ordering, metric_entity, metric_name) in EXPECTED.items():
        spec = known[objective]
        if (spec.direction, spec.units, spec.ordering_rule, spec.metric_entity, spec.metric_name) != (
            direction,
            units,
            ordering,
            metric_entity,
            metric_name,
        ):
            raise AssertionError(f"unexpected spec for {objective}: {spec}")
        if dse_objectives.policy_id_for_objective(objective) != f"deterministic_{objective}_v1":
            raise AssertionError(f"unexpected policy id for {objective}")
        if candidate_summary.ordering_rule_for_objective(objective) != ordering:
            raise AssertionError(f"candidate summary ordering drifted for {objective}")
        if intermediate_artifacts.dse_ordering_rule_for_objective(objective) != ordering:
            raise AssertionError(f"artifact audit ordering drifted for {objective}")
        if intermediate_artifacts.dse_objective_semantics(objective) != (direction, units):
            raise AssertionError(f"artifact audit semantics drifted for {objective}")
    if candidate_summary.ordering_rule_for_objective("custom_latency") != "runtime_score_then_candidate_id":
        raise AssertionError("unknown objective fallback changed")
    if intermediate_artifacts.dse_objective_semantics("custom_latency") is not None:
        raise AssertionError("unknown objective audit semantics changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
