#!/usr/bin/env python3
"""Validate the public product visualization closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fabric-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-spectrum-class",
        choices=("intermediate", "max_spatial", "max_temporal"),
    )
    parser.add_argument("--minimum-region-count", type=int, default=1)
    parser.add_argument("--require-transition", action="store_true")
    arguments = parser.parse_args()
    if arguments.minimum_region_count < 1:
        raise ValueError("minimum region count must be positive")

    fabric = read_object(arguments.fabric_root)
    bundle = read_object(arguments.bundle)
    if bundle.get("schema") != "loom.visualization_bundle":
        raise ValueError("visualization bundle has the wrong schema")
    if bundle.get("version") != "1.2":
        raise ValueError("visualization bundle has the wrong version")
    if bundle.get("fabric") != fabric:
        raise ValueError("visualization bundle names a different Fabric root")
    for field in ("tech_mappings", "spatial_mappings", "system_mappings"):
        if not isinstance(bundle.get(field), list) or not bundle[field]:
            raise ValueError(f"visualization bundle has no {field}")
    deployment = bundle.get("deployment")
    if not isinstance(deployment, dict) or deployment.get("schema") != (
        "loom.deployment"
    ):
        raise ValueError("visualization bundle has no Deployment reference")
    pair = bundle.get("pair_decision")
    successful = {
        "verified_acceleration",
        "verified_feasible_but_not_beneficial",
        "hardware_dse_alternative",
    }
    if not isinstance(pair, dict) or pair.get("disposition") not in successful:
        raise ValueError("visualization bundle has no successful pair decision")
    if pair.get("quality_disposition") not in {
        "not_requested",
        "complete",
        "unsupported",
        "proof_not_established",
        "execution_failed",
        "cancelled_or_timeout",
    }:
        raise ValueError("pair decision has no typed quality disposition")
    for field in (
        "quality_observations",
        "hardware_promotion_observations",
        "quality_invocations",
    ):
        if not isinstance(pair.get(field), list):
            raise ValueError(f"pair decision has no {field} array")
    selected_mapping = pair.get("selected_system_mapping")
    if not isinstance(selected_mapping, str) or not selected_mapping:
        raise ValueError("pair decision has no selected SystemMapping")
    selected_candidates = [
        candidate
        for candidate in pair.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("selected") is True
    ]
    selected_attempt_mappings = {
        mapping
        for candidate in selected_candidates
        for observation in candidate.get("mapping_observations", [])
        if isinstance(observation, dict)
        for mapping in observation.get("system_mappings", [])
        if isinstance(mapping, str)
    }
    if selected_mapping not in selected_attempt_mappings:
        raise ValueError("selected SystemMapping has no exact attempt join")
    for field in ("resource_time_endpoints", "resource_time_transitions"):
        if not isinstance(bundle.get(field), list):
            raise ValueError(f"visualization bundle has no {field} array")
    if not bundle["resource_time_endpoints"]:
        raise ValueError("visualization bundle has no resource-time endpoint")
    spectrum = bundle.get("resource_time_spectrum")
    if not isinstance(spectrum, dict) or spectrum.get("status") != "verified":
        raise ValueError("visualization bundle has no verified resource-time spectrum")
    scenarios = spectrum.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("resource-time spectrum has no scenario")
    active_allocations = 0
    active_regions: set[tuple[str, int]] = set()
    spectrum_classes: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("resource-time scenario is not an object")
        spectrum_class = scenario.get("spectrum_class")
        if not isinstance(spectrum_class, str):
            raise ValueError("resource-time scenario has no spectrum class")
        spectrum_classes.add(spectrum_class)
        if not isinstance(
            scenario.get("analytic_schedule_makespan_picoseconds"), int
        ):
            raise ValueError("resource-time scenario has no analytic makespan")
        mappings = scenario.get("system_mappings")
        states = scenario.get("states")
        if not isinstance(mappings, list) or not mappings:
            raise ValueError("resource-time scenario has no SystemMapping")
        if not isinstance(states, list) or not states:
            raise ValueError("resource-time scenario has no event-relative state")
        for state in states:
            if (
                not isinstance(state, dict)
                or not isinstance(state.get("event"), str)
                or not isinstance(state.get("time_picoseconds"), int)
                or not isinstance(state.get("mapping"), dict)
                or not isinstance(state.get("active"), list)
            ):
                raise ValueError("resource-time state is incomplete")
            for allocation in state["active"]:
                resources = (
                    allocation.get("resources")
                    if isinstance(allocation, dict)
                    else None
                )
                if not isinstance(resources, list) or not resources:
                    raise ValueError("resource-time allocation has no resource")
                artifact = allocation.get("region_artifact")
                entity = allocation.get("region_entity")
                if not isinstance(artifact, str) or not isinstance(entity, int):
                    raise ValueError("resource-time allocation has no region")
                active_regions.add((artifact, entity))
                active_allocations += 1
    if active_allocations == 0:
        raise ValueError("resource-time spectrum has no active allocation")
    if len(active_regions) < arguments.minimum_region_count:
        raise ValueError("resource-time spectrum covers too few active regions")
    if (
        arguments.expected_spectrum_class
        and arguments.expected_spectrum_class not in spectrum_classes
    ):
        raise ValueError("resource-time spectrum has the wrong endpoint class")

    transitions = bundle["resource_time_transitions"]
    if arguments.require_transition and not transitions:
        raise ValueError("visualization bundle has no resource-time transition")
    for transition in transitions:
        if not isinstance(transition, dict) or transition.get("status") != "verified":
            raise ValueError("resource-time transition is not verified")
        for field in (
            "trigger",
            "safe_point",
            "parent",
            "child",
            "before_active",
            "after_active",
            "completed_before",
            "before_live_work",
            "after_live_work",
            "resource_delta",
            "configuration_delta",
            "route_delta",
        ):
            if field not in transition:
                raise ValueError(f"resource-time transition has no {field}")
        repair = transition.get("repair")
        if not isinstance(repair, dict):
            raise ValueError("resource-time transition has no repair evidence")
        roots = repair.get("reopened_roots")
        if not isinstance(roots, list) or not roots:
            raise ValueError("resource-time transition has no repair roots")
        for field in (
            "cold_wall_time_ns",
            "incremental_wall_time_ns",
            "cold_verifier_retained_bytes",
            "incremental_verifier_retained_bytes",
            "cold_verifier_work",
            "incremental_verifier_work",
        ):
            if not isinstance(repair.get(field), int) or repair[field] <= 0:
                raise ValueError(f"resource-time repair has no {field}")
        if repair.get("mapping_reuse_disposition") not in {
            "preserved",
            "local_repair",
            "cold_fallback",
        }:
            raise ValueError("resource-time repair has no typed disposition")
        if repair.get("cold_dfg_cycles") is None and repair.get(
            "cold_cgra_cycles"
        ) is None:
            raise ValueError("resource-time cold replay has no QoR")
        if repair.get("incremental_dfg_cycles") is None and repair.get(
            "incremental_cgra_cycles"
        ) is None:
            raise ValueError("resource-time incremental replay has no QoR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
