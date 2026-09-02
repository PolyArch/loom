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


def canonical_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


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
    if bundle.get("version") != "1.3":
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
    endpoints = bundle["resource_time_endpoints"]
    if not endpoints:
        raise ValueError("visualization bundle has no resource-time endpoint")
    endpoint_keys: set[str] = set()
    for endpoint in endpoints:
        if (
            not isinstance(endpoint, dict)
            or not isinstance(endpoint.get("mapping"), dict)
            or not isinstance(endpoint.get("deployment"), dict)
        ):
            raise ValueError("resource-time endpoint is incomplete")
        key = canonical_key(endpoint)
        if key in endpoint_keys:
            raise ValueError("resource-time endpoint is duplicated")
        endpoint_keys.add(key)
    entry_endpoints = [
        endpoint
        for endpoint in endpoints
        if isinstance(endpoint["mapping"].get("artifact"), str)
        and selected_mapping.endswith(endpoint["mapping"]["artifact"])
    ]
    if len(entry_endpoints) != 1:
        raise ValueError("selected SystemMapping does not identify one entry endpoint")
    entry_mapping = entry_endpoints[0]["mapping"]
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
    edge_keys: set[str] = set()
    entry_parent_spectra: list[dict[str, Any]] = []
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
            "logical_memories",
            "resource_delta",
            "configuration_delta",
            "route_delta",
            "reprogramming_time_picoseconds",
            "migration_time_picoseconds",
        ):
            if field not in transition:
                raise ValueError(f"resource-time transition has no {field}")
        parent_endpoint = transition["parent"]
        child_endpoint = transition["child"]
        if (
            not isinstance(parent_endpoint, dict)
            or not isinstance(child_endpoint, dict)
            or canonical_key(parent_endpoint) not in endpoint_keys
            or canonical_key(child_endpoint) not in endpoint_keys
        ):
            raise ValueError("resource-time transition names a foreign endpoint")
        edge_key = canonical_key(
            {
                field: transition[field]
                for field in (
                    "trigger",
                    "safe_point",
                    "parent",
                    "child",
                    "before_active",
                    "after_active",
                    "completed_before",
                    "logical_memories",
                    "resource_delta",
                    "configuration_delta",
                    "route_delta",
                    "reprogramming_time_picoseconds",
                    "migration_time_picoseconds",
                    "status",
                )
            }
        )
        if edge_key in edge_keys:
            raise ValueError("resource-time transition is duplicated")
        edge_keys.add(edge_key)

        parent_spectrum = transition.get("parent_spectrum")
        child_spectrum = transition.get("child_spectrum")
        if (
            not isinstance(parent_spectrum, dict)
            or parent_spectrum.get("status") != "verified"
            or not isinstance(child_spectrum, dict)
            or child_spectrum.get("status") != "verified"
        ):
            raise ValueError("resource-time transition has no verified spectra")
        if (
            parent_spectrum.get("dataflow") != child_spectrum.get("dataflow")
            or parent_spectrum.get("fabric") != child_spectrum.get("fabric")
        ):
            raise ValueError("resource-time endpoint spectra have different owners")
        if parent_endpoint.get("mapping") == entry_mapping:
            entry_parent_spectra.append(parent_spectrum)

        parent_mapping = parent_endpoint.get("mapping")
        child_mapping = child_endpoint.get("mapping")
        boundary_matches = 0
        for scenario in parent_spectrum.get("scenarios", []):
            states = scenario.get("states") if isinstance(scenario, dict) else None
            if not isinstance(states, list):
                raise ValueError("parent spectrum scenario has no state path")
            for before, after in zip(states, states[1:]):
                if (
                    isinstance(before, dict)
                    and isinstance(after, dict)
                    and before.get("mapping") == parent_mapping
                    and after.get("mapping") == child_mapping
                    and after.get("event") == transition["trigger"]
                    and canonical_key(before.get("active"))
                    == canonical_key(transition["before_active"])
                    and canonical_key(after.get("active"))
                    == canonical_key(transition["after_active"])
                ):
                    boundary_matches += 1
        if boundary_matches != 1:
            raise ValueError("parent spectrum does not bind one exact edge boundary")

        child_has_active_state = False
        child_scenarios = child_spectrum.get("scenarios")
        if not isinstance(child_scenarios, list) or not child_scenarios:
            raise ValueError("child spectrum has no scenario")
        for scenario in child_scenarios:
            if (
                not isinstance(scenario, dict)
                or scenario.get("system_mappings") != [child_mapping]
                or not isinstance(scenario.get("states"), list)
            ):
                raise ValueError("child spectrum is not bound to its exact Mapping")
            for state in scenario["states"]:
                if not isinstance(state, dict) or state.get("mapping") != child_mapping:
                    raise ValueError("child spectrum state names another Mapping")
                child_has_active_state |= bool(state.get("active"))
        if not child_has_active_state:
            raise ValueError("child spectrum has no active exact-Mapping state")

        repair = transition.get("repair")
        if not isinstance(repair, dict):
            raise ValueError("resource-time transition has no repair evidence")
        roots = repair.get("reopened_roots")
        if not isinstance(roots, list) or not roots:
            raise ValueError("resource-time transition has no repair roots")
        if len({canonical_key(root) for root in roots}) != len(roots):
            raise ValueError("resource-time transition repeats a repair root")
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
        for mode in ("cold", "incremental"):
            provider_work = repair.get(f"{mode}_provider_work")
            if not isinstance(provider_work, dict):
                raise ValueError(f"resource-time repair has no {mode} provider work")
            for provider in ("tech_mapping", "spatial_pnr", "system_pnr"):
                invocations = provider_work.get(f"{provider}_invocations")
                dispatches = provider_work.get(f"{provider}_dispatches")
                replays = provider_work.get(f"{provider}_journal_replays")
                if (
                    not isinstance(invocations, int)
                    or not isinstance(dispatches, int)
                    or not isinstance(replays, int)
                    or min(invocations, dispatches, replays) < 0
                    or invocations != dispatches + replays
                ):
                    raise ValueError(
                        f"resource-time {mode} {provider} work does not reconcile"
                    )
            if provider_work["system_pnr_invocations"] == 0:
                raise ValueError(f"resource-time repair has no {mode} child PnR")
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
        if (repair.get("cold_dfg_cycles") is None) != (
            repair.get("incremental_dfg_cycles") is None
        ) or (repair.get("cold_cgra_cycles") is None) != (
            repair.get("incremental_cgra_cycles") is None
        ):
            raise ValueError("resource-time repair QoR domains do not match")
        expected_disposition = "cold_fallback"
        if repair.get("preserved_tech_mappings", 0) or repair.get(
            "preserved_spatial_mappings", 0
        ):
            expected_disposition = (
                "local_repair"
                if repair.get("repaired_tech_mappings", 0)
                or repair.get("repaired_spatial_mappings", 0)
                else "preserved"
            )
        if repair.get("mapping_reuse_disposition") != expected_disposition:
            raise ValueError("resource-time repair disposition is inconsistent")
    if transitions and (
        not entry_parent_spectra
        or any(parent_spectrum != spectrum for parent_spectrum in entry_parent_spectra)
    ):
        raise ValueError("top-level spectrum is not bound to the entry Mapping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
