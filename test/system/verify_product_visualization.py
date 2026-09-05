#!/usr/bin/env python3
"""Validate the public product visualization closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def canonical_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def artifact_reference(
    value: Any,
    expected_schema: str | None = None,
    expected_version: str | None = None,
) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("schema"), str)
        and (expected_schema is None or value["schema"] == expected_schema)
        and isinstance(value.get("schema_version"), str)
        and (
            expected_version is None
            or value["schema_version"] == expected_version
        )
        and isinstance(value.get("artifact"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["artifact"]) is not None
    )


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
    if bundle.get("version") != "1.5":
        raise ValueError("visualization bundle has the wrong version")
    if bundle.get("fabric") != fabric:
        raise ValueError("visualization bundle names a different Fabric root")
    for field in ("tech_mappings", "spatial_mappings", "system_mappings"):
        if not isinstance(bundle.get(field), list) or not bundle[field]:
            raise ValueError(f"visualization bundle has no {field}")
    runtime_inventory = bundle.get("runtime_evidence")
    if (
        not isinstance(runtime_inventory, list)
        or not runtime_inventory
        or any(
            not artifact_reference(reference, "evaluation.evidence", "1.0")
            for reference in runtime_inventory
        )
        or len({canonical_key(reference) for reference in runtime_inventory})
        != len(runtime_inventory)
    ):
        raise ValueError("visualization bundle has no canonical runtime Evidence")
    runtime_inventory_keys = {
        canonical_key(reference) for reference in runtime_inventory
    }
    selected_system = bundle.get("selected_system")
    if not isinstance(selected_system, dict) or not isinstance(
        selected_system.get("artifact"), str
    ):
        raise ValueError("visualization bundle has no selected System")
    mapping_domains = bundle.get("mapping_domains")
    if not isinstance(mapping_domains, list) or not mapping_domains:
        raise ValueError("visualization bundle has no Mapping domains")
    domain_systems: set[str] = set()
    flattened = {
        "tech_mappings": set(),
        "spatial_mappings": set(),
        "system_mappings": set(),
    }
    for domain in mapping_domains:
        if not isinstance(domain, dict) or not isinstance(domain.get("system"), dict):
            raise ValueError("visualization Mapping domain has no System")
        if any(
            not isinstance(domain["system"].get(field), str)
            for field in ("schema", "schema_version", "artifact")
        ):
            raise ValueError("visualization Mapping domain has an invalid System")
        system_key = canonical_key(domain["system"])
        if system_key in domain_systems:
            raise ValueError("visualization repeats a Mapping System domain")
        domain_systems.add(system_key)
        expected_projection = (
            "fabric"
            if domain["system"] == bundle["fabric"]
            else f"fabric-{domain['system'].get('artifact')}"
        )
        if domain.get("fabric_projection") != expected_projection:
            raise ValueError("visualization Mapping domain has the wrong projection")
        for suffix in ("mlir", "html"):
            if not (
                arguments.bundle.parent / f"{expected_projection}.{suffix}"
            ).is_file():
                raise ValueError("visualization Mapping domain projection is missing")
        for field in flattened:
            values = domain.get(field)
            if not isinstance(values, list) or (
                field == "system_mappings" and not values
            ):
                raise ValueError(f"visualization Mapping domain has no {field}")
            keys = [canonical_key(value) for value in values]
            if any(
                not isinstance(value, dict)
                or not isinstance(value.get("artifact"), str)
                for value in values
            ) or len(set(keys)) != len(keys):
                raise ValueError(
                    f"visualization Mapping domain has invalid {field}"
                )
            flattened[field].update(keys)
    for field, values in flattened.items():
        if values != {canonical_key(value) for value in bundle[field]}:
            raise ValueError(f"visualization {field} is not its domain union")
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
    if (
        not isinstance(pair, dict)
        or pair.get("schema") != "loom.application_pair_decision"
        or pair.get("version") != "1.2"
        or pair.get("disposition") not in successful
    ):
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
    repair_attempts = pair.get("resource_time_mapping_repair_attempt_count")
    verified_repairs = pair.get("resource_time_mapping_repair_verified_count")
    repair_incomplete_reason = pair.get(
        "resource_time_mapping_repair_incomplete_reason"
    )
    if (
        not isinstance(repair_attempts, int)
        or isinstance(repair_attempts, bool)
        or repair_attempts < 0
        or not isinstance(verified_repairs, int)
        or isinstance(verified_repairs, bool)
        or not 0 <= verified_repairs <= repair_attempts
        or (verified_repairs == repair_attempts)
        != (repair_incomplete_reason is None)
        or (
            verified_repairs < repair_attempts
            and (
                not isinstance(repair_incomplete_reason, str)
                or not repair_incomplete_reason
            )
        )
    ):
        raise ValueError("pair decision has an invalid Mapping repair summary")
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
    candidates = pair.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("pair decision has no candidate inventory")
    selected_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("selected") is True
    ]
    if len(selected_candidates) != 1:
        raise ValueError("pair decision does not select one candidate")
    selected_candidate = selected_candidates[0]
    selected_plan = selected_candidate.get("plan_ordinal")
    selected_hint = pair.get("selected_schedule_hint_digest")
    if (
        isinstance(selected_plan, bool)
        or not isinstance(selected_plan, int)
        or selected_plan < 0
        or not isinstance(selected_hint, str)
        or len(selected_hint) != 64
    ):
        raise ValueError("selected candidate has no exact plan and schedule")
    mapping_observations = selected_candidate.get("mapping_observations")
    if not isinstance(mapping_observations, list):
        raise ValueError("selected candidate has no Mapping observations")
    selected_observations = [
        observation
        for observation in mapping_observations
        if isinstance(observation, dict)
        and observation.get("plan_ordinal") == selected_plan
        and observation.get("schedule_hint_digest") == selected_hint
        and observation.get("runtime_mapping") == selected_mapping
    ]
    if len(selected_observations) != 1:
        raise ValueError(
            "selected plan, schedule, and Mapping do not identify one observation"
        )
    selected_observation = selected_observations[0]
    if (
        selected_observation.get("mapping_disposition") != "verified"
        or selected_observation.get("runtime_disposition") != "completed"
        or selected_mapping
        not in selected_observation.get("system_mappings", [])
        or not isinstance(selected_observation.get("runtime_evidence"), list)
        or not selected_observation["runtime_evidence"]
        or not isinstance(selected_observation.get("oracle_evidence"), list)
        or not selected_observation["oracle_evidence"]
    ):
        raise ValueError(
            "selected Mapping lacks completed runtime and oracle Evidence"
        )
    pair_selected_system = pair.get("selected_system")
    if (
        not isinstance(pair_selected_system, str)
        or not pair_selected_system.endswith(selected_system["artifact"])
    ):
        raise ValueError("pair decision and bundle select different Systems")
    selected_domains = [
        domain
        for domain in mapping_domains
        if domain["system"] == selected_system
        and any(
            isinstance(mapping, dict)
            and isinstance(mapping.get("artifact"), str)
            and selected_mapping.endswith(mapping["artifact"])
            for mapping in domain["system_mappings"]
        )
    ]
    if len(selected_domains) != 1:
        raise ValueError("selected Mapping has no unique System domain")
    selected_domain_mapping_keys = {
        canonical_key(mapping) for mapping in selected_domains[0]["system_mappings"]
    }

    repair_records = bundle.get("hardware_mutation_repair_records")
    selected_repair = bundle.get("selected_hardware_mutation_repair_record")
    if not isinstance(repair_records, list) or any(
        not isinstance(record, dict)
        or not isinstance(record.get("artifact"), str)
        for record in repair_records
    ) or len({canonical_key(record) for record in repair_records}) != len(
        repair_records
    ):
        raise ValueError("visualization bundle has no repair-record inventory")
    if pair["disposition"] == "hardware_dse_alternative":
        observed_repair = selected_observation.get(
            "hardware_mutation_repair_record"
        )
        if observed_repair is None:
            if selected_repair is not None:
                raise ValueError("selected repair record changed across projections")
        elif (
            not isinstance(observed_repair, str)
            or not isinstance(selected_repair, dict)
            or not isinstance(selected_repair.get("artifact"), str)
            or selected_repair not in repair_records
            or not observed_repair.endswith(selected_repair["artifact"])
        ):
            raise ValueError("selected repair record changed across projections")
    elif selected_repair is not None:
        raise ValueError("non-hardware selection names a selected repair record")
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
            or canonical_key(endpoint["mapping"])
            not in selected_domain_mapping_keys
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
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("resource-time scenario is not an object")
        spectrum_class = scenario.get("spectrum_class")
        if not isinstance(spectrum_class, str):
            raise ValueError("resource-time scenario has no spectrum class")
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
    transitions = bundle["resource_time_transitions"]
    if verified_repairs < len(transitions):
        raise ValueError("resource-time transitions exceed verified repairs")
    if arguments.require_transition and not transitions:
        raise ValueError("visualization bundle has no resource-time transition")
    if len(endpoints) != len(transitions) + 1:
        raise ValueError("resource-time graph is not one finite path")
    edge_keys: set[str] = set()
    entry_parent_spectra: list[dict[str, Any]] = []
    expected_parent_endpoint = entry_endpoints[0]
    path_endpoint_keys = {canonical_key(expected_parent_endpoint)}
    path_mappings = [entry_mapping]
    path_mapping_keys = {canonical_key(entry_mapping)}
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
        child_endpoint_key = canonical_key(child_endpoint)
        child_mapping_key = canonical_key(child_endpoint["mapping"])
        if (
            parent_endpoint != expected_parent_endpoint
            or child_endpoint_key in path_endpoint_keys
            or child_mapping_key in path_mapping_keys
        ):
            raise ValueError("resource-time transitions do not form one ordered path")
        expected_parent_endpoint = child_endpoint
        path_endpoint_keys.add(child_endpoint_key)
        path_mappings.append(child_endpoint["mapping"])
        path_mapping_keys.add(child_mapping_key)
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
            if (
                arguments.expected_spectrum_class
                and scenario.get("spectrum_class")
                != arguments.expected_spectrum_class
            ):
                continue
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
                child_has_active_state |= (
                    not arguments.expected_spectrum_class
                    or scenario.get("spectrum_class")
                    == arguments.expected_spectrum_class
                ) and bool(state.get("active"))
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
            runtime_evidence = repair.get(f"{mode}_runtime_evidence")
            oracle_evidence = repair.get(f"{mode}_oracle_evidence")
            if (
                not isinstance(runtime_evidence, list)
                or not runtime_evidence
                or not isinstance(oracle_evidence, list)
                or not oracle_evidence
                or any(
                    not artifact_reference(
                        reference, "evaluation.evidence", "1.0"
                    )
                    or canonical_key(reference) not in runtime_inventory_keys
                    for reference in runtime_evidence
                )
                or any(
                    canonical_key(reference)
                    not in {canonical_key(value) for value in runtime_evidence}
                    for reference in oracle_evidence
                )
            ):
                raise ValueError(
                    f"resource-time repair has no exact {mode} runtime Evidence join"
                )
            for domain in ("dfg", "cgra"):
                cycles = repair.get(f"{mode}_{domain}_cycles")
                if cycles is not None and (
                    isinstance(cycles, bool)
                    or not isinstance(cycles, int)
                    or cycles < 0
                ):
                    raise ValueError(
                        f"resource-time repair has invalid {mode} {domain} cycles"
                    )
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
    if path_endpoint_keys != endpoint_keys:
        raise ValueError("resource-time path does not cover its exact endpoints")
    matching_entry_scenarios: list[dict[str, Any]] = []
    for scenario in scenarios:
        if (
            arguments.expected_spectrum_class
            and scenario.get("spectrum_class")
            != arguments.expected_spectrum_class
        ):
            continue
        state_path: list[dict[str, Any]] = []
        for state in scenario["states"]:
            mapping = state["mapping"]
            if not state_path or state_path[-1] != mapping:
                state_path.append(mapping)
        if state_path == path_mappings:
            matching_entry_scenarios.append(scenario)
    if len(matching_entry_scenarios) != 1:
        raise ValueError(
            "resource-time endpoint class does not bind one exact selected path"
        )
    selected_active_allocations = 0
    selected_active_regions: set[tuple[str, int]] = set()
    for state in matching_entry_scenarios[0]["states"]:
        for allocation in state["active"]:
            selected_active_regions.add(
                (allocation["region_artifact"], allocation["region_entity"])
            )
            selected_active_allocations += 1
    if selected_active_allocations == 0:
        raise ValueError("selected resource-time path has no active allocation")
    if len(selected_active_regions) < arguments.minimum_region_count:
        raise ValueError("selected resource-time path covers too few active regions")
    if transitions and (
        not entry_parent_spectra
        or any(parent_spectrum != spectrum for parent_spectrum in entry_parent_spectra)
    ):
        raise ValueError("top-level spectrum is not bound to the entry Mapping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
