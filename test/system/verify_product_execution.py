#!/usr/bin/env python3
"""Validate one source-to-Deployment product execution contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from loom_evidence_portfolio import (  # noqa: E402
    collect_portfolio_inventory,
    validate_portfolio_pair,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def read_diagnostics(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        value = json.loads(line)
        if value.get("schema") != "loom.invocation.diagnostic.1":
            continue
        require(
            isinstance(value.get("payload"), dict),
            "diagnostic payload must be an object",
        )
        events.append(value)
    require(events, "product build emitted no structured diagnostics")
    return events


def matching_payloads(
    events: list[dict[str, Any]], *, stage: str, event: str
) -> list[dict[str, Any]]:
    return [
        row["payload"]
        for row in events
        if row.get("stage") == stage and row.get("event") == event
    ]


def validate_context(
    events: list[dict[str, Any]],
    stage: str,
    context_kind: str,
    expected_contexts: int | None = 1,
) -> None:
    rows = [
        payload
        for payload in matching_payloads(events, stage=stage, event="derived_context")
        if payload.get("context_kind") == context_kind
    ]
    contexts: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get("context_key")
        require(
            isinstance(key, str)
            and len(key) == 64
            and key == key.lower()
            and all(character in "0123456789abcdef" for character in key),
            f"{context_kind} lacks a complete immutable context key",
        )
        contexts.setdefault(key, []).append(row)
    require(contexts, f"{context_kind} emitted no context identities")
    if expected_contexts is not None:
        require(
            len(contexts) == expected_contexts,
            f"{context_kind} has the wrong distinct context count",
        )
    for key, context_rows in contexts.items():
        require(
            sum(row.get("cache_misses", -1) for row in context_rows) == 1,
            f"{context_kind} {key} has the wrong construction count",
        )
        require(
            sum(row.get("cache_hits", -1) for row in context_rows) >= 1,
            f"{context_kind} {key} was not reused",
        )
        require(
            all(row.get("construction_count") == 1 for row in context_rows),
            f"{context_kind} {key} construction count is inconsistent",
        )
        for field in (
            "construction_time_ns",
            "retained_bytes",
            "deterministic_work",
        ):
            values = {row.get(field) for row in context_rows}
            require(
                len(values) == 1
                and all(isinstance(value, int) and value > 0 for value in values),
                f"{context_kind} {key} lacks stable positive {field}",
            )


def search_invocations(
    events: list[dict[str, Any]], stage: str, statistics_kind: str
) -> list[dict[str, Any]]:
    rows = [
        payload
        for payload in matching_payloads(events, stage=stage, event="statistics")
        if payload.get("statistics_kind") == statistics_kind
    ]
    require(rows, f"expected at least one {statistics_kind} row")
    return rows


def validate_mapping_work(
    events: list[dict[str, Any]],
    expected_system_active_contexts: int | None,
    spatial_search_frontier: bool,
    portfolio_application: str | None,
    portfolio_input: str | None,
) -> dict[str, Any]:
    reopen_attempts = [
        payload
        for payload in matching_payloads(events, stage="system_pnr", event="candidate")
        if payload.get("operation") == "hardware_reopen_mapping_attempt"
    ]
    hardware_reopen = bool(reopen_attempts)
    if hardware_reopen:
        require(
            any(
                row.get("added_temporal_contexts", 0) > 0
                and row.get("system_mapping_count", 0) > 0
                for row in reopen_attempts
            ),
            "hardware reopen published no verified grown-System Mapping",
        )
    tech_rows = [
        payload
        for payload in matching_payloads(
            events, stage="tech_mapping", event="statistics"
        )
        if payload.get("statistics_kind") == "application_tech_root_supply_frontier"
    ]
    nonempty_tech = [
        row for row in tech_rows if row.get("candidate_publications", 0) > 0
    ]
    require(nonempty_tech, "no software alternative published a TechMapping frontier")
    selected_tech = nonempty_tech[-1]
    require(
        selected_tech.get("candidate_publications", 0) >= 2,
        "product gate did not exercise multiple TechMapping candidates",
    )

    context_count = None if hardware_reopen else 1
    validate_context(events, "spatial_pnr", "fabric_static", context_count)
    validate_context(events, "spatial_pnr", "fabric_timing", context_count)
    validate_context(events, "system_pnr", "system_static", context_count)
    validate_context(
        events,
        "system_pnr",
        "system_active",
        expected_system_active_contexts,
    )

    spatial = search_invocations(events, "spatial_pnr", "spatial_pnr_invocation")
    if spatial_search_frontier:
        require(
            len(spatial) > 1,
            "Spatial search frontier did not exercise multiple searches",
        )
    system = search_invocations(events, "system_pnr", "system_pnr_invocation")
    join_rows = [
        payload
        for payload in matching_payloads(events, stage="system_pnr", event="statistics")
        if payload.get("domain") == "application_mapping_join"
    ]
    require(len(join_rows) == 1, "expected one application Mapping join summary")
    join = join_rows[0]
    pair_decision = join.get("pair_decision")
    require(
        isinstance(pair_decision, dict),
        "application Mapping join omitted the pair-level decision",
    )
    pair_identity = pair_decision.get("pair_identity")
    require(
        isinstance(pair_identity, str)
        and len(pair_identity) == 64
        and pair_identity == pair_identity.lower()
        and all(character in "0123456789abcdef" for character in pair_identity),
        "pair-level decision has no stable identity",
    )
    for root_key in ("source_program", "fabric", "workload", "runtime_input"):
        root = pair_decision.get(root_key)
        require(
            isinstance(root, str) and len(root) > 0,
            "pair-level decision omitted " + root_key,
        )
    manifest_run_key = pair_decision.get("invocation_manifest_run_key")
    require(
        isinstance(manifest_run_key, str)
        and len(manifest_run_key) == 64
        and manifest_run_key == manifest_run_key.lower()
        and all(character in "0123456789abcdef" for character in manifest_run_key),
        "pair-level decision has no InvocationManifest run-key join",
    )
    require(
        pair_decision.get("invocation_manifest_join_status")
        == "owner_scoped_planning_closure",
        "successful product decision did not join its planning Manifest",
    )
    require(
        pair_decision.get("disposition")
        in {
            "verified_acceleration",
            "verified_feasible_but_not_beneficial",
            "hardware_dse_alternative",
        },
        "successful product Mapping published a non-success pair decision",
    )
    require(
        pair_decision.get("host_only_baseline_complete") is True,
        "successful product decision omitted the host-only baseline",
    )
    require(
        pair_decision.get("final_application_qor_complete") is True,
        "successful product decision omitted application QoR evidence",
    )
    if portfolio_application is not None:
        portfolio = pair_decision.get("portfolio_input")
        require(
            isinstance(portfolio, dict),
            "portfolio product decision omitted its manifest selection",
        )
        require(
            portfolio.get("application_identity") == portfolio_application
            and portfolio.get("input_name") == portfolio_input,
            "portfolio product decision names the wrong manifest selection",
        )
        require(
            portfolio.get("execution_binding") == "canonical_simulation_and_oracle"
            and portfolio.get("execution_binding_established") is True,
            "portfolio selection lacks canonical Simulation and oracle Evidence",
        )
        profile = portfolio.get("declared_profile")
        require(
            isinstance(profile, dict)
            and profile.get("warmup_samples") == 0
            and profile.get("measured_samples") == 1
            and profile.get("total_samples") == 1
            and profile.get("oracle_coverage") == "all_measured_samples"
            and isinstance(profile.get("deadline_milliseconds"), int)
            and profile["deadline_milliseconds"] > 0,
            "portfolio product decision changed its bounded profile",
        )
    selected_values = {
        observation.get("dimension"): observation
        for observation in pair_decision.get("selected_objective", [])
        if isinstance(observation, dict)
    }
    baseline_values = {
        observation.get("dimension"): observation
        for observation in pair_decision.get("host_only_baseline", [])
        if isinstance(observation, dict)
    }
    require(
        isinstance(baseline_values.get("host_only_work"), dict)
        and isinstance(baseline_values["host_only_work"].get("value"), int)
        and baseline_values["host_only_work"].get("evidence")
        in {"exact", "sound_bound", "analytic", "calibrated", "runtime_measured"},
        "host-only baseline has no typed work observation",
    )
    for dimension in ("dfg_cycles", "cgra_cycles"):
        require(
            isinstance(selected_values.get(dimension), dict)
            and isinstance(selected_values[dimension].get("value"), int)
            and selected_values[dimension].get("evidence") == "runtime_measured",
            f"successful product decision lacks measured {dimension}",
        )
    require(
        isinstance(pair_decision.get("candidates"), list)
        and pair_decision["candidates"],
        "pair-level decision omitted its bounded candidate inventory",
    )
    require(
        isinstance(pair_decision.get("planning_record_count"), int)
        and isinstance(pair_decision.get("non_candidate_planning_record_count"), int)
        and pair_decision["planning_record_count"] >= len(pair_decision["candidates"])
        and pair_decision["non_candidate_planning_record_count"]
        == pair_decision["planning_record_count"] - len(pair_decision["candidates"]),
        "pair-level planning/candidate inventory counts do not reconcile",
    )
    selected_candidates = []
    for candidate in pair_decision["candidates"]:
        candidate_identity = candidate.get("candidate_identity")
        require(
            isinstance(candidate_identity, str)
            and len(candidate_identity) == 64
            and candidate_identity == candidate_identity.lower()
            and all(
                character in "0123456789abcdef" for character in candidate_identity
            ),
            "semantic candidate is missing its stable identity",
        )
        if candidate.get("entered_mapping"):
            observations = candidate.get("mapping_observations")
            require(
                isinstance(observations, list) and observations,
                "mapped candidate omitted its Mapping observations",
            )
            for observation in observations:
                require(
                    isinstance(observation, dict),
                    "candidate Mapping observation is malformed",
                )
                hint = observation.get("schedule_hint_digest")
                require(
                    isinstance(hint, str) and len(hint) == 64,
                    "candidate Mapping observation lacks schedule identity",
                )
                require(
                    isinstance(observation.get("system"), str),
                    "candidate Mapping observation lacks System identity",
                )
                require(
                    isinstance(observation.get("system_mappings"), list),
                    "candidate Mapping observation lacks Mapping witness list",
                )
                require(
                    observation.get("mapping_disposition")
                    in {"verified", "proven_no_feasible_candidate", "incomplete"},
                    "candidate Mapping observation lacks typed disposition",
                )
        if candidate.get("selected") is True:
            selected_candidates.append(candidate)
    require(
        len(selected_candidates) == 1,
        "pair-level decision does not identify one selected candidate",
    )
    selected_candidate = selected_candidates[0]
    selected_plan = join.get("selected_plan_ordinal")
    selected_mapping = join.get("selected_mapping")
    require(
        isinstance(selected_plan, int)
        and selected_plan >= 0
        and selected_candidate.get("plan_ordinal") == selected_plan,
        "selected candidate does not bind the selected plan checkpoint",
    )
    require(
        isinstance(selected_mapping, str)
        and selected_mapping == pair_decision.get("selected_system_mapping"),
        "selected Mapping checkpoint disagrees with the pair decision",
    )
    selected_observations = [
        observation
        for observation in selected_candidate.get("mapping_observations", [])
        if isinstance(observation, dict)
        and observation.get("plan_ordinal") == selected_plan
        and selected_mapping in observation.get("system_mappings", [])
    ]
    require(
        len(selected_observations) == 1,
        "selected plan and Mapping do not identify one candidate observation",
    )
    selected_observation = selected_observations[0]
    require(
        selected_observation.get("mapping_disposition") == "verified"
        and selected_observation.get("runtime_disposition") == "completed"
        and isinstance(selected_observation.get("runtime_evidence"), list)
        and selected_observation["runtime_evidence"]
        and isinstance(selected_observation.get("oracle_evidence"), list)
        and selected_observation["oracle_evidence"],
        "selected Mapping lacks completed runtime and comparison Evidence",
    )
    require(
        selected_observation.get("system") == pair_decision.get("selected_system"),
        "selected Mapping observation disagrees with the selected System",
    )
    for dimension, field in (
        ("dfg_cycles", "dfg_cycles"),
        ("cgra_cycles", "cgra_cycles"),
        ("resource_core_cost", "resource_core_cost"),
    ):
        observation = selected_values.get(dimension)
        require(
            isinstance(selected_observation.get(field), int)
            and isinstance(observation, dict)
            and observation.get("evidence") == "runtime_measured"
            and observation.get("value") == selected_observation[field],
            f"selected {dimension} is not bound to its Mapping observation",
        )
    objective_vectors = [pair_decision.get("host_only_baseline", [])]
    objective_vectors.extend(
        candidate.get("objective", []) for candidate in pair_decision["candidates"]
    )
    selected_objective = pair_decision.get("selected_objective")
    if isinstance(selected_objective, list) and selected_objective:
        objective_vectors.append(selected_objective)
    for vector in objective_vectors:
        require(
            isinstance(vector, list) and len(vector) == 11,
            "pair decision objective vector is not structurally complete",
        )
        for observation in vector:
            require(
                isinstance(observation, dict),
                "pair decision objective observation is malformed",
            )
            if observation.get("evidence") == "unsupported":
                require(
                    observation.get("value") is None,
                    "unsupported objective dimension was encoded as zero",
                )
    initial_system_invocations = join.get("system_pnr_invocation_count")
    verified_alternatives = join.get("verified_alternatives")
    transitions = join.get("application_incremental_mapping_transitions")
    require(
        isinstance(initial_system_invocations, int)
        and initial_system_invocations > 0
        and isinstance(verified_alternatives, int)
        and verified_alternatives > 0
        and isinstance(join.get("system_pnr_dispatch_count"), int)
        and join["system_pnr_dispatch_count"] >= initial_system_invocations
        and isinstance(transitions, list),
        "System invocation ledger disagrees with the application join",
    )
    for transition in transitions:
        require(
            isinstance(transition, dict)
            and isinstance(transition.get("cold_mapping"), str)
            and isinstance(transition.get("child_mapping"), str),
            "incremental transition lacks cold and repaired Mapping witnesses",
        )
    verified_plan_ordinals = {
        observation.get("plan_ordinal")
        for candidate in pair_decision["candidates"]
        for observation in candidate.get("mapping_observations", [])
        if isinstance(observation, dict)
        and observation.get("mapping_disposition") == "verified"
        and observation.get("system_mappings")
        and isinstance(observation.get("plan_ordinal"), int)
    }
    verified_mapping_ids = {
        mapping
        for candidate in pair_decision["candidates"]
        for observation in candidate.get("mapping_observations", [])
        if isinstance(observation, dict)
        and observation.get("mapping_disposition") == "verified"
        for mapping in observation.get("system_mappings", [])
        if isinstance(mapping, str)
    }
    published_slots = sum(row.get("candidate_publications", 0) for row in system)
    require(
        verified_plan_ordinals
        and verified_mapping_ids
        and verified_alternatives <= len(verified_mapping_ids)
        and len(verified_plan_ordinals) <= published_slots,
        "verified Mapping inventory does not reconcile with published roots",
    )
    if transitions:
        # Incremental hardware-reopen work runs under the first-verified
        # product goal, where every System row publishes exactly one
        # candidate: one row per verified alternative plus a cold and an
        # incremental row per transition.
        require(
            len(system) == verified_alternatives + 2 * len(transitions),
            "verified System rows do not reconcile with cold and incremental"
            " work",
        )
    else:
        # A provider publication is a concrete SystemMapping slot. One
        # verified planning alternative may publish more than one root, and a
        # runtime-qualified tail may contribute roots from several invocations;
        # compare the aggregate only as a bound while the candidate inventory
        # above checks the exact plan and root witnesses.
        require(
            verified_alternatives <= published_slots,
            "verified System alternatives exceed published roots",
        )
    incremental_system_rows = [
        row for row in system if row.get("migration_seed_attempt_slots") == 1
    ]
    require(
        len(incremental_system_rows) == len(transitions)
        and all(
            row.get("migration_seed_prepared") == 1 for row in incremental_system_rows
        ),
        "incremental System searches do not reconcile with transition work",
    )
    # A builtin product build stops at its first verified candidate; a build
    # under an explicit ResolvedConfig may instead exhaust its configured
    # restarts. The closure status names the profile each row obeyed.
    for name, rows in (("Spatial", spatial), ("System", system)):
        published_rows = 0
        for row in rows:
            closure_status = row.get("closure_status")
            publications = row.get("candidate_publications")
            seed_slots = row.get("seed_attempt_slots")
            require(
                isinstance(seed_slots, int)
                and seed_slots >= 1
                and row.get("prepared_seeds") == seed_slots,
                f"{name} search prepared unexpected restart work",
            )
            if closure_status == "closed":
                # Exhaustive work: every prepared restart finalizes and may
                # publish its own verified candidate.
                require(
                    isinstance(publications, int)
                    and 0 <= publications <= seed_slots,
                    f"{name} search published more than its finalized restarts",
                )
                require(
                    row.get("finalized_restarts") == seed_slots
                    and row.get("publication_slots") == seed_slots,
                    f"{name} search did not finalize its configured restarts",
                )
                require(
                    isinstance(row.get("final_closure_attempts"), int)
                    and row["final_closure_attempts"] >= 1,
                    f"{name} search skipped final closure",
                )
            else:
                require(
                    closure_status == "semantic_limit_reached",
                    f"{name} search did not stop at its verified product"
                    " result",
                )
                # A joint pair whose search exhausts its seeds without a
                # feasible incumbent is a typed empty outcome: the joint
                # frontier falls to the next pair. Such an invocation
                # publishes and finalizes nothing; every successful
                # invocation finalizes exactly one. The first-verified goal
                # is a bounded prefix: a seed whose search finds no feasible
                # incumbent legitimately falls through to the next attempt.
                require(
                    publications in (0, 1),
                    f"{name} search published more than one candidate",
                )
                if name == "System" and row.get("migration_seed_attempt_slots") == 1:
                    require(
                        row.get("final_closure_attempts") == 0,
                        "incremental System search repeated cold final closure",
                    )
                else:
                    # The bounded repair/global-closure loop may retry closure
                    # within one seed, so attempts are bounded by the work
                    # ledger rather than the seed count.
                    require(
                        isinstance(row.get("final_closure_attempts"), int)
                        and row["final_closure_attempts"] >= 1,
                        f"{name} search skipped final closure",
                    )
                require(
                    row.get("finalized_restarts") == publications
                    and row.get("publication_slots") == publications,
                    f"{name} search did not finalize exactly one result",
                )
            published_rows += publications
        require(
            published_rows >= 1,
            f"no {name} search finalized a verified product result",
        )
    require(
        all(
            isinstance(row.get("final_verification_attempts"), int)
            and row["final_verification_attempts"]
            >= row.get("candidate_publications", 0)
            for row in system
        ),
        "System search skipped independent candidate verification",
    )
    require(
        all(row.get("endpoint_expansion_slots", 0) > 0 for row in spatial),
        "a Spatial search did not exercise endpoint routing",
    )

    breakdowns = [
        row["payload"]
        for row in events
        if row.get("event") == "statistics"
        and row.get("payload", {}).get("operation") == "simulation_cycle_breakdown"
    ]
    require(
        {row.get("engine") for row in breakdowns} >= {"dfg", "cgra"},
        "product replay omitted DFG/CGRA cycle breakdown profiling",
    )
    for row in breakdowns:
        require(
            row.get("measurement_kind") == "direct_and_derived"
            and isinstance(row.get("direct"), dict)
            and isinstance(row.get("derived"), dict),
            "cycle breakdown did not separate direct and derived metrics",
        )
        direct = row["direct"]
        require(
            isinstance(direct.get("cycle_count"), int) and direct["cycle_count"] > 0,
            "cycle breakdown has no positive direct cycle count",
        )
        if row.get("engine") == "cgra":
            static_plan = direct.get("static_plan")
            require(
                isinstance(static_plan, dict),
                "CGRA breakdown omitted its static physical plan",
            )
            for key in (
                "physical_use_acquire_rank_sum",
                "physical_use_release_rank_sum",
                "physical_use_max_acquire_rank",
                "physical_use_max_release_rank",
                "compute_transition_timing_count",
                "memory_transition_timing_count",
                "produced_transport_timing_count",
                "consumed_transport_timing_count",
                "traversal_transport_timing_count",
                "maximum_route_node_depth",
                "temporal_compute_actor_count",
                "spatial_compute_actor_count",
                "temporal_dispatch_domain_count",
                "operand_buffer_count",
            ):
                require(
                    isinstance(static_plan.get(key), int),
                    "CGRA static timing profile omitted " + key,
                )
            for key in (
                "launch_reference_cycle_numerator",
                "graph_retirement_reference_cycle_numerator",
                "terminal_reference_cycle_numerator",
                "terminal_event_delta",
                "maximum_reference_cycle_numerator",
                "maximum_event_delta",
                "physical_grant_wait_cycle_sum",
                "physical_grant_wait_cycle_max",
                "physical_action_lifetime_cycle_sum",
                "physical_action_lifetime_cycle_max",
                "physical_granted_lifetime_cycle_sum",
                "physical_granted_lifetime_cycle_max",
                "physical_grant_same_cycle_count",
                "physical_grant_delayed_count",
                "non_integral_timing_observation_count",
            ):
                require(
                    isinstance(direct.get(key), int),
                    "CGRA runtime timing profile omitted " + key,
                )
            require(
                isinstance(
                    row["derived"].get("post_retirement_drain_cycles"),
                    (dict, type(None)),
                ),
                "CGRA timing profile omitted post-retirement drain",
            )
            require(
                direct.get("physical_request_count")
                == direct.get("physical_grant_count")
                == direct.get("physical_retirement_count"),
                "CGRA physical lifecycle counts do not close",
            )
    comparisons = [
        payload
        for payload in matching_payloads(events, stage="system_pnr", event="statistics")
        if payload.get("operation") == "simulation_cycle_comparison"
    ]
    require(comparisons, "product replay omitted DFG/CGRA cycle comparison")
    for comparison in comparisons:
        require(
            comparison.get("measurement_kind") == "direct_and_derived"
            and isinstance(comparison.get("direct"), dict)
            and isinstance(comparison.get("derived"), dict),
            "cycle comparison did not separate direct and derived metrics",
        )
        require(
            isinstance(comparison["direct"].get("dfg_cycles"), int)
            and isinstance(comparison["direct"].get("cgra_cycles"), int),
            "cycle comparison lacks direct DFG/CGRA counts",
        )
    return join


def validate_spatial_unconditional_handshake(
    events: list[dict[str, Any]],
) -> None:
    static_rows = [
        payload
        for payload in matching_payloads(
            events, stage="spatial_pnr", event="derived_context"
        )
        if payload.get("context_kind") == "fabric_static"
    ]
    unconditional_counts = {
        row.get("handshake_unconditional_arc_count") for row in static_rows
    }
    require(
        len(unconditional_counts) == 1,
        "Fabric static handshake arc count is inconsistent",
    )
    unconditional_count = next(iter(unconditional_counts))
    require(
        isinstance(unconditional_count, int) and unconditional_count > 0,
        "Fabric static context has no unconditional handshake arcs",
    )

    active_rows = [
        payload
        for payload in matching_payloads(
            events, stage="spatial_pnr", event="derived_context"
        )
        if payload.get("context_kind") == "spatial_active_handshake"
    ]
    require(active_rows, "Spatial search emitted no active handshake context")
    require(
        all(
            row.get("fabric_unconditional_arc_count") == unconditional_count
            and row.get("materialized_arc_count", 0) > unconditional_count
            for row in active_rows
        ),
        "active handshake graph omitted Fabric unconditional dependencies",
    )


def validate_reference(value: Any, context: str) -> None:
    require(isinstance(value, dict), f"{context} must be an artifact reference")
    identity = value.get("artifact")
    require(
        isinstance(identity, str)
        and len(identity) == 64
        and all(character in "0123456789abcdef" for character in identity),
        f"{context} has an invalid artifact identity",
    )


def validate_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    spatial_invocations: int | None,
    required_dataflow_text: list[str],
    mapping_inspector: str | None,
    require_actor_multicast: bool,
    require_operand_queue_atomic_fanout: bool,
    require_memory_engine: bool,
    require_memory_internal_edge: bool,
    require_temporal_memory: bool,
    require_register_fifo: bool,
    require_packed_switch_row: bool,
    require_temporal_dispatch: bool,
    dense_coordinate_rank: int | None,
    require_unique_dense_coordinates: bool,
    minimum_unique_acc_cores: int,
) -> None:
    require(
        manifest.get("schema") == "loom.execution_matrix_workspace.1.2",
        "execution workspace has the wrong schema",
    )
    for field in ("deployment", "workload", "runtime_input", "gem5_binding"):
        validate_reference(manifest.get(field), field)
    require(
        manifest.get("value_results") == [["0"]],
        "execution cells did not agree with the independent product oracle",
    )

    runs = manifest.get("runs")
    require(isinstance(runs, list), "execution workspace runs must be an array")
    system_runs = [run for run in runs if run.get("scope") == "system"]
    spatial_runs = [run for run in runs if run.get("scope") == "spatial"]
    require(
        len(system_runs) == 2
        and {run.get("engine") for run in system_runs} == {"dfg", "cgra"},
        "execution workspace does not contain both System cells",
    )
    observed_invocations = sorted(
        {
            run.get("invocation_ordinal")
            for run in spatial_runs
            if isinstance(run.get("invocation_ordinal"), int)
            and run.get("invocation_ordinal") >= 0
        }
    )
    require(
        observed_invocations
        and observed_invocations == list(range(len(observed_invocations))),
        "execution workspace has a sparse or empty Spatial invocation set",
    )
    if spatial_invocations is not None:
        require(
            len(observed_invocations) == spatial_invocations,
            "execution workspace has the wrong Spatial invocation count",
        )
    else:
        spatial_invocations = len(observed_invocations)
    require(
        len(spatial_runs) == spatial_invocations * 2,
        "execution workspace has an incomplete Spatial execution matrix",
    )
    for ordinal in range(spatial_invocations):
        invocation_runs = [
            run for run in spatial_runs if run.get("invocation_ordinal") == ordinal
        ]
        require(
            len(invocation_runs) == 2
            and {run.get("engine") for run in invocation_runs} == {"dfg", "cgra"},
            f"Spatial invocation {ordinal} lacks both execution cells",
        )
        coordinates = [run.get("dense_coordinates") for run in invocation_runs]
        require(
            len(coordinates) == 2 and coordinates[0] == coordinates[1],
            f"Spatial invocation {ordinal} engines disagree on coordinates",
        )
        require(
            isinstance(coordinates[0], list)
            and all(
                isinstance(coordinate, int) and coordinate >= 0
                for coordinate in coordinates[0]
            ),
            f"Spatial invocation {ordinal} has invalid coordinates",
        )
        if dense_coordinate_rank is not None:
            require(
                len(coordinates[0]) == dense_coordinate_rank,
                f"Spatial invocation {ordinal} has the wrong coordinate rank",
            )
        target_ordinals = [
            run.get("dispatch_target_ordinal") for run in invocation_runs
        ]
        acc_core_references = [run.get("acc_core_ref") for run in invocation_runs]
        context_keys = [run.get("execution_context_key") for run in invocation_runs]
        require(
            len(target_ordinals) == 2
            and target_ordinals[0] == target_ordinals[1]
            and isinstance(target_ordinals[0], int)
            and target_ordinals[0] >= 0,
            f"Spatial invocation {ordinal} engines disagree on dispatch target",
        )
        for values, name in (
            (acc_core_references, "AccCore reference"),
            (context_keys, "execution-context key"),
        ):
            require(
                len(values) == 2
                and values[0] == values[1]
                and isinstance(values[0], str)
                and len(values[0]) > 0
                and len(values[0]) % 2 == 0
                and all(character in "0123456789abcdef" for character in values[0]),
                f"Spatial invocation {ordinal} has an invalid {name}",
            )
    if require_unique_dense_coordinates:
        coordinate_points = {
            tuple(run["dense_coordinates"])
            for run in spatial_runs
            if run.get("engine") == "dfg"
        }
        require(
            len(coordinate_points) == spatial_invocations,
            "Spatial invocations do not have unique dense coordinates",
        )
    unique_acc_cores = {
        run["acc_core_ref"] for run in spatial_runs if run.get("engine") == "dfg"
    }
    require(
        len(unique_acc_cores) >= minimum_unique_acc_cores,
        "Spatial invocations use fewer AccCores than required",
    )
    for run in runs:
        label = f"{run.get('scope')}/{run.get('engine')}"
        for field in ("request", "evidence", "execution"):
            validate_reference(run.get(field), f"{label} {field}")
        if run.get("scope") == "spatial":
            for field in ("dataflow", "spatial_mapping", "hardware_implementation"):
                validate_reference(run.get(field), f"{label} {field}")
            require(
                isinstance(run.get("terminal_cycle"), dict),
                f"{label} lacks a terminal cycle",
            )
        else:
            require(
                all(
                    isinstance(run.get(field), int)
                    for field in ("entry_tick", "exit_tick", "terminal_tick")
                ),
                f"{label} lacks exact gem5 ticks",
            )

    dataflow_identities = {run["dataflow"]["artifact"] for run in spatial_runs}
    require(
        len(dataflow_identities) == 1,
        "Spatial invocations do not share one exact Dataflow artifact",
    )
    dataflow_path = manifest_path.parent / "objects" / next(iter(dataflow_identities))
    dataflow = dataflow_path.read_bytes()
    for text in required_dataflow_text:
        require(
            text.encode("ascii") in dataflow,
            f"canonical Dataflow does not contain {text}",
        )

    if (
        require_actor_multicast
        or require_operand_queue_atomic_fanout
        or require_memory_engine
        or require_memory_internal_edge
        or require_temporal_memory
        or require_register_fifo
        or require_packed_switch_row
        or require_temporal_dispatch
    ):
        require(
            mapping_inspector is not None,
            "Mapping feature validation requires a Mapping inspector",
        )
        mapping_identities = {
            run["spatial_mapping"]["artifact"] for run in spatial_runs
        }
        reports: list[dict[str, Any]] = []
        for identity in sorted(mapping_identities):
            completed = subprocess.run(
                [mapping_inspector, str(manifest_path.parent / "objects"), identity],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            require(
                isinstance(report, dict), "Mapping inspection report must be an object"
            )
            require(
                report.get("schema") == "loom.test.product_mapping_inspection.1",
                "Mapping inspection report has the wrong schema",
            )
            reports.append(report)
        if require_actor_multicast:
            require(
                any(
                    report.get("actor_multicast_route_count", 0) > 0
                    and report.get("maximum_actor_multicast_sinks", 0) >= 2
                    for report in reports
                ),
                "Spatial Mapping contains no complete actor-result multicast",
            )
        if require_operand_queue_atomic_fanout:
            require(
                any(
                    report.get("operand_queue_atomic_fanout_group_count", 0) > 0
                    and report.get("maximum_operand_queue_matches", 0) >= 2
                    for report in reports
                ),
                "Spatial Mapping contains no atomic operand-queue fanout",
            )
        if require_memory_engine:
            require(
                any(
                    report.get("spatial_memory_engine_binding_count", 0) > 0
                    and report.get("configured_memory_occurrence_count", 0) > 0
                    and report.get("configured_memory_active_operation_row_count", 0)
                    > 0
                    for report in reports
                ),
                "Spatial Mapping contains no configured Memory Engine",
            )
        if require_memory_internal_edge:
            require(
                any(
                    report.get("memory_internal_edge_count", 0) > 0
                    and report.get(
                        "fabric_memory_template_internal_connection_count", 0
                    )
                    > 0
                    and report.get(
                        "fabric_memory_template_with_internal_connection_count",
                        0,
                    )
                    > 0
                    for report in reports
                ),
                "Spatial Mapping contains no Fabric-backed memory internal edge",
            )
        if require_temporal_memory:
            require(
                any(
                    report.get("temporal_memory_engine_binding_count", 0) > 0
                    and report.get("temporal_memory_operation_count", 0) > 0
                    and report.get("temporal_memory_occurrence_count", 0)
                    == report.get("dense_temporal_memory_occurrence_count", -1)
                    and report.get("temporal_memory_external_ingress_claim_count", 0)
                    > 0
                    and report.get("temporal_memory_external_ingress_claim_count", 0)
                    == report.get(
                        "unique_temporal_memory_external_ingress_claim_count",
                        -1,
                    )
                    for report in reports
                ),
                "Spatial Mapping contains no closed Temporal Memory Engine",
            )
        if require_register_fifo:
            require(
                any(
                    report.get("register_fifo_transfer_count", 0) > 0
                    for report in reports
                ),
                "Spatial Mapping contains no RegFIFO local transfer",
            )
        if require_packed_switch_row:
            require(
                any(
                    report.get("shared_packed_switch_row_count", 0) > 0
                    and report.get("maximum_packed_switch_row_signatures", 0) >= 2
                    for report in reports
                ),
                "Spatial Mapping contains no shared packed switch row",
            )
        if require_temporal_dispatch:
            require(
                any(
                    report.get("temporal_compute_binding_count", 0) > 0
                    and report.get("temporal_dispatch_domain_count", 0) > 0
                    and report.get("temporal_dispatch_candidate_count", 0) > 0
                    for report in reports
                ),
                "Spatial Mapping contains no Temporal PE dispatch domain",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spatial-invocations", type=int)
    parser.add_argument("--expected-system-active-contexts", type=int)
    parser.add_argument("--spatial-search-frontier", action="store_true")
    parser.add_argument(
        "--require-spatial-unconditional-handshake", action="store_true"
    )
    parser.add_argument("--required-dataflow-text", action="append", default=[])
    parser.add_argument("--mapping-inspector")
    parser.add_argument("--require-actor-multicast", action="store_true")
    parser.add_argument("--require-operand-queue-atomic-fanout", action="store_true")
    parser.add_argument("--require-memory-engine", action="store_true")
    parser.add_argument("--require-memory-internal-edge", action="store_true")
    parser.add_argument("--require-temporal-memory", action="store_true")
    parser.add_argument("--require-register-fifo", action="store_true")
    parser.add_argument("--require-packed-switch-row", action="store_true")
    parser.add_argument("--require-temporal-dispatch", action="store_true")
    parser.add_argument("--dense-coordinate-rank", type=int)
    parser.add_argument("--require-unique-dense-coordinates", action="store_true")
    parser.add_argument("--minimum-unique-acc-cores", type=int, default=1)
    parser.add_argument("--portfolio-application")
    parser.add_argument("--portfolio-input")
    parser.add_argument("--portfolio-inventory", type=Path)
    arguments = parser.parse_args()
    require(
        (arguments.portfolio_application is None)
        == (arguments.portfolio_input is None),
        "portfolio application and input must be selected together",
    )
    require(
        (arguments.portfolio_application is None)
        == (arguments.portfolio_inventory is None),
        "portfolio selections require the canonical manifest inventory",
    )
    require(
        arguments.spatial_invocations is None or arguments.spatial_invocations > 0,
        "Spatial invocation count must be positive",
    )
    require(
        arguments.expected_system_active_contexts is None
        or arguments.expected_system_active_contexts > 0,
        "expected System active context count must be positive",
    )
    require(
        arguments.minimum_unique_acc_cores > 0,
        "minimum unique AccCore count must be positive",
    )
    require(
        arguments.dense_coordinate_rank is None or arguments.dense_coordinate_rank >= 0,
        "dense coordinate rank must be nonnegative",
    )
    events = read_diagnostics(arguments.diagnostics)
    pair_evidence = validate_mapping_work(
        events,
        arguments.expected_system_active_contexts,
        arguments.spatial_search_frontier,
        arguments.portfolio_application,
        arguments.portfolio_input,
    )
    if arguments.portfolio_inventory is not None:
        inventory, inventory_errors = collect_portfolio_inventory(
            read_json(arguments.portfolio_inventory)
        )
        require(
            not inventory_errors,
            f"canonical portfolio inventory is invalid: {inventory_errors}",
        )
        expected = next(
            (
                row
                for row in inventory
                if row.get("application_identity") == arguments.portfolio_application
                and row.get("input_name") == arguments.portfolio_input
            ),
            None,
        )
        evaluation = validate_portfolio_pair(pair_evidence, expected)
        require(
            evaluation is not None
            and evaluation["typed_complete"]
            and evaluation["canonical_qor_complete"],
            f"production pair evidence did not close canonical QoR: {evaluation}",
        )
    if arguments.require_spatial_unconditional_handshake:
        validate_spatial_unconditional_handshake(events)
    validate_manifest(
        read_json(arguments.manifest),
        arguments.manifest,
        arguments.spatial_invocations,
        arguments.required_dataflow_text,
        arguments.mapping_inspector,
        arguments.require_actor_multicast,
        arguments.require_operand_queue_atomic_fanout,
        arguments.require_memory_engine,
        arguments.require_memory_internal_edge,
        arguments.require_temporal_memory,
        arguments.require_register_fifo,
        arguments.require_packed_switch_row,
        arguments.require_temporal_dispatch,
        arguments.dense_coordinate_rank,
        arguments.require_unique_dense_coordinates,
        arguments.minimum_unique_acc_cores,
    )


if __name__ == "__main__":
    main()
