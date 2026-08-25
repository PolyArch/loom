#!/usr/bin/env python3
"""Build a deterministic Loom evidence manifest from invocation artifacts.

The manifest is a derived report. It never becomes an artifact identity and it
does not copy an old result store: every input is hashed and parsed at the
time the report is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return records
    stripped = text.strip()
    if stripped:
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            metrics: dict[str, Any] = {}
            for key, raw in re.findall(
                r"\b(WALL_SECONDS|MAX_RSS_KB|STATUS)=([0-9]+(?:\.[0-9]+)?)",
                line,
            ):
                if key == "STATUS":
                    metrics["exit_status"] = int(float(raw))
                elif key == "WALL_SECONDS":
                    metrics["wall_seconds"] = float(raw)
                else:
                    metrics["max_rss_kb"] = int(float(raw))
            if metrics:
                records.append(metrics)
            if "pre_mapping_spectrum_endpoint_unsupported" in line:
                records.append(
                    {
                        "outcome": "unsupported",
                        "diagnostic": "pre_mapping_spectrum_endpoint_unsupported",
                    }
                )
            start = line.find("{")
            if start < 0:
                continue
            line = line[start:]
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            metrics: dict[str, Any] = {}
            for key, raw in re.findall(
                r"\b(WALL_SECONDS|MAX_RSS_KB|STATUS)=([0-9]+(?:\.[0-9]+)?)",
                line,
            ):
                if key == "STATUS":
                    metrics["exit_status"] = int(float(raw))
                elif key == "WALL_SECONDS":
                    metrics["wall_seconds"] = float(raw)
                else:
                    metrics["max_rss_kb"] = int(float(raw))
            if metrics:
                records.append(metrics)
            if "pre_mapping_spectrum_endpoint_unsupported" in line:
                records.append(
                    {
                        "outcome": "unsupported",
                        "diagnostic": "pre_mapping_spectrum_endpoint_unsupported",
                    }
                )
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def expand_evidence_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose the existing nested CLI evidence without inventing facts."""
    def tagged(value: dict[str, Any], kind: str) -> dict[str, Any]:
        result = dict(value)
        result["__loom_record_kind"] = kind
        return result

    expanded = [tagged(record, "root")]
    planning = record.get("planning")
    if not isinstance(planning, dict):
        return expanded
    expanded.append(tagged(planning, "planning"))
    inventory = planning.get("candidate_inventory")
    if isinstance(inventory, list):
        expanded.extend(
            tagged(candidate, "candidate")
            for candidate in inventory
            if isinstance(candidate, dict)
        )
    return expanded


LEDGER_FIELDS = (
    "limit",
    "planned",
    "consumed",
    "reserved",
    "rejected",
    "cancelled",
    "elapsed_nanoseconds",
)


def collect_work_entries(
    value: Any,
    path: str,
    stage: Any,
    event: Any,
    output: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    """Collect nested invocation ledgers without treating reports as owners."""
    if not isinstance(value, dict):
        return
    numeric = {
        name: parsed
        for name in LEDGER_FIELDS
        if (parsed := integer(value.get(name))) is not None
    }
    required = {"planned", "consumed", "reserved", "rejected", "cancelled"}
    if required.issubset(numeric):
        entry = {
            "name": path,
            "stage": stage,
            "event": event,
            "values": numeric,
        }
        output.append(entry)
        valid = (
            numeric["reserved"] == numeric["planned"]
            and numeric["consumed"] <= numeric["reserved"]
            and numeric["consumed"]
            + numeric["rejected"]
            + numeric["cancelled"]
            == numeric["reserved"]
        )
        if not valid:
            errors.append(entry)
    for key, child in value.items():
        if isinstance(child, dict):
            collect_work_entries(
                child, f"{path}.{key}" if path else key,
                stage, event, output, errors
            )


def validate_portfolio_pair(decision: dict[str, Any]) -> dict[str, Any] | None:
    selection = decision.get("portfolio_input")
    if not isinstance(selection, dict):
        return None
    application = selection.get("application_identity")
    input_name = selection.get("input_name")
    reasons: list[str] = []
    typed_reasons: list[str] = []
    if not isinstance(application, str) or not isinstance(input_name, str):
        reasons.append("invalid_selection")
        typed_reasons.append("invalid_selection")
    dispositions = {
        "verified_acceleration",
        "verified_feasible_but_not_beneficial",
        "no_promising_candidate",
        "exact_hardware_incompatible",
        "mapping_proof_not_established",
        "cancelled_or_timeout",
        "budget_exhausted",
        "unsupported_semantic",
        "implementation_failure",
        "hardware_dse_alternative",
    }
    disposition = decision.get("disposition")
    if disposition not in dispositions:
        typed_reasons.append("invalid_disposition")
    join_status = decision.get("invocation_manifest_join_status")
    if join_status not in {
        "exact",
        "owner_scoped_planning_closure",
        "owner_verified_pre_admission",
    }:
        typed_reasons.append("manifest_join_unverified")
    if join_status == "owner_verified_pre_admission":
        if decision.get("manifest_join_owner_verified") is not True:
            typed_reasons.append("manifest_join_owner_unverified")
        if not isinstance(decision.get("manifest_join_owner"), str):
            typed_reasons.append("manifest_join_owner_missing")
        if not isinstance(decision.get("manifest_join_contract"), str):
            typed_reasons.append("manifest_join_contract_missing")
    else:
        run_key = decision.get("invocation_manifest_run_key")
        if not isinstance(run_key, str) or re.fullmatch(r"[0-9a-f]{64}", run_key) is None:
            typed_reasons.append("manifest_run_key_missing")
    if not isinstance(decision.get("fabric"), str):
        typed_reasons.append("fabric_missing")
    causal_dispositions = {
        "no_promising_candidate",
        "exact_hardware_incompatible",
        "mapping_proof_not_established",
        "cancelled_or_timeout",
        "budget_exhausted",
        "unsupported_semantic",
        "implementation_failure",
    }
    if disposition in causal_dispositions and not isinstance(
        decision.get("detail"), str
    ):
        typed_reasons.append("typed_failure_detail_missing")
    if (
        selection.get("execution_binding") != "canonical_simulation_and_oracle"
        or selection.get("execution_binding_established") is not True
    ):
        reasons.append("execution_binding_incomplete")
    if decision.get("host_only_baseline_complete") is not True:
        reasons.append("host_baseline_incomplete")
    if decision.get("final_application_qor_complete") is not True:
        reasons.append("application_qor_incomplete")
    if decision.get("invocation_manifest_join_status") != "exact":
        reasons.append("invocation_manifest_join_incomplete")

    candidates = decision.get("candidates")
    selected = []
    if isinstance(candidates, list):
        selected = [
            candidate
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("selected") is True
        ]
    if len(selected) != 1:
        reasons.append("selected_candidate_not_unique")
    else:
        observations = selected[0].get("mapping_observations")
        completed = (
            [
                observation
                for observation in observations
                if isinstance(observation, dict)
                and observation.get("runtime_disposition") == "completed"
                and isinstance(observation.get("system_mappings"), list)
                and observation["system_mappings"]
                and isinstance(observation.get("oracle_evidence"), list)
                and observation["oracle_evidence"]
            ]
            if isinstance(observations, list)
            else []
        )
        if not completed:
            reasons.append("selected_mapping_evidence_incomplete")

    objective_vectors = [decision.get("host_only_baseline")]
    objective_vectors.append(decision.get("selected_objective"))
    if isinstance(candidates, list):
        objective_vectors.extend(
            candidate.get("objective")
            for candidate in candidates
            if isinstance(candidate, dict)
        )
    for vector in objective_vectors:
        if not isinstance(vector, list):
            continue
        if any(
            isinstance(observation, dict)
            and observation.get("evidence") == "unsupported"
            and observation.get("value") is not None
            for observation in vector
        ):
            reasons.append("unsupported_objective_has_value")
            break
    if disposition in {
        "verified_acceleration",
        "verified_feasible_but_not_beneficial",
        "hardware_dse_alternative",
    }:
        typed_reasons.extend(f"success_{reason}" for reason in reasons)
    return {
        "application_identity": application,
        "input_name": input_name,
        "disposition": disposition,
        "typed_complete": not typed_reasons,
        "typed_incomplete_reasons": typed_reasons,
        "complete": not reasons,
        "incomplete_reasons": reasons,
    }


def validate_portfolio_host_run(
    report: dict[str, Any], expected: dict[str, Any] | None
) -> dict[str, Any]:
    selection = report.get("selection")
    application = (
        selection.get("application_identity") if isinstance(selection, dict) else None
    )
    input_name = selection.get("input_name") if isinstance(selection, dict) else None
    reasons: list[str] = []
    if (
        report.get("schema") != "loom.application_host_run"
        or report.get("version") != "1.0"
    ):
        reasons.append("invalid_report_schema")
    if not isinstance(application, str) or not isinstance(input_name, str):
        reasons.append("invalid_selection")
    if expected is None:
        reasons.append("manifest_selection_missing")
    else:
        expected_selection = {
            key: expected[key]
            for key in (
                "application_identity",
                "input_name",
                "source",
                "build",
                "workload",
                "runtime_input",
                "cached_inputs",
                "oracle",
            )
        }
        if selection != expected_selection:
            reasons.append("manifest_selection_mismatch")
        if report.get("profile") != expected["profile"]:
            reasons.append("manifest_profile_mismatch")

    source_admission = report.get("source_admission")
    if (
        not isinstance(source_admission, dict)
        or source_admission.get("status") != "admitted"
    ):
        reasons.append("source_not_admitted")
    compilation = report.get("compile")
    if (
        not isinstance(compilation, dict)
        or compilation.get("status") != "succeeded"
        or compilation.get("exit_status") != 0
        or not isinstance(compilation.get("compiler"), str)
    ):
        reasons.append("host_compile_incomplete")
    execution = report.get("execution")
    wall_time = (
        execution.get("host_wall_time_nanoseconds")
        if isinstance(execution, dict)
        else None
    )
    if (
        not isinstance(execution, dict)
        or execution.get("status") != "succeeded"
        or execution.get("exit_status") != 0
        or integer(wall_time) is None
        or wall_time <= 0
    ):
        reasons.append("host_execution_incomplete")
    elif expected is not None:
        profile = expected.get("profile")
        deadline = (
            profile.get("deadline_milliseconds")
            if isinstance(profile, dict)
            else None
        )
        if integer(deadline) is None or wall_time > deadline * 1_000_000:
            reasons.append("host_deadline_exceeded")
    oracle = report.get("oracle_result")
    if not isinstance(oracle, dict) or oracle.get("status") != "matched":
        reasons.append("host_oracle_incomplete")
    if report.get("outcome") != "succeeded":
        reasons.append("host_outcome_incomplete")
    return {
        "application_identity": application,
        "input_name": input_name,
        "outcome": report.get("outcome"),
        "host_wall_time_nanoseconds": wall_time,
        "complete": not reasons,
        "incomplete_reasons": reasons,
    }


def portfolio_selection_key(report: dict[str, Any]) -> tuple[Any, Any]:
    selection = report.get("selection")
    if not isinstance(selection, dict):
        return (None, None)
    return (selection.get("application_identity"), selection.get("input_name"))


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def repository_facts(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "head": run("rev-parse", "HEAD"),
        "worktree_clean": status == "",
        "status_available": status is not None,
    }


def collect_facts(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    work: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    identities: dict[str, set[str]] = {}
    completeness: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    application_outcomes: list[dict[str, Any]] = []
    application_pair_decisions: list[dict[str, Any]] = []
    portfolio_inventory: list[dict[str, Any]] = []
    portfolio_host_runs: list[dict[str, Any]] = []
    quality_summaries: list[dict[str, Any]] = []
    quality_observations: list[dict[str, Any]] = []
    funnel_summaries: list[dict[str, Any]] = []
    resource_time_funnels: list[dict[str, Any]] = []
    evaluation_timings: list[dict[str, Any]] = []
    mapping_observations: list[dict[str, Any]] = []
    ledger_errors: list[dict[str, Any]] = []
    cache_observations: list[dict[str, Any]] = []
    runtime_failures: list[dict[str, Any]] = []
    simulation_cycle_breakdowns: list[dict[str, Any]] = []
    simulation_cycle_comparisons: list[dict[str, Any]] = []
    incomplete_observations: list[dict[str, Any]] = []
    resource_time_terminal_observations: list[dict[str, Any]] = []
    resource_time_evaluations: list[dict[str, Any]] = []
    execution_matrix_observations: list[dict[str, Any]] = []
    migration_observations: list[dict[str, Any]] = []
    record_kinds: dict[str, int] = {}
    for record in records:
        record_kind = record.get("__loom_record_kind", "root")
        if not isinstance(record_kind, str):
            record_kind = "root"
        record_kinds[record_kind] = record_kinds.get(record_kind, 0) + 1
        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = record
        if payload.get("schema") == "loom.execution_matrix_workspace.1.2":
            runs = payload.get("runs")
            if isinstance(runs, list):
                for run in runs:
                    if not isinstance(run, dict):
                        continue
                    observation = {
                        key: run[key]
                        for key in (
                            "scope",
                            "engine",
                            "invocation_ordinal",
                            "dispatch_target_ordinal",
                            "entry_tick",
                            "exit_tick",
                            "terminal_tick",
                            "request",
                            "evidence",
                            "execution",
                            "dataflow",
                            "acc_core_ref",
                            "execution_context_key",
                        )
                        if key in run
                    }
                    execution_matrix_observations.append(observation)
        if payload.get("schema") == "loom.application_portfolio":
            applications = payload.get("applications")
            if isinstance(applications, list):
                for application in applications:
                    if not isinstance(application, dict):
                        continue
                    identity = application.get("identity")
                    selections = application.get("selections")
                    inputs = application.get("inputs")
                    cached_inputs = application.get("cached_inputs")
                    if not isinstance(inputs, list):
                        continue
                    cached_input_rows = (
                        cached_inputs if isinstance(cached_inputs, list) else []
                    )
                    cached_by_name = {
                        cached.get("logical_name"): cached
                        for cached in cached_input_rows
                        if isinstance(cached, dict)
                        and isinstance(cached.get("logical_name"), str)
                    }
                    for selected_input in inputs:
                        if not isinstance(selected_input, dict):
                            continue
                        selected_cached_names = selected_input.get("cached_inputs")
                        selected_cached = (
                            [cached_by_name.get(name) for name in selected_cached_names]
                            if isinstance(selected_cached_names, list)
                            else []
                        )
                        portfolio_inventory.append(
                            {
                                "application_identity": identity,
                                "input_name": selected_input.get("name"),
                                "source": application.get("source"),
                                "build": application.get("build"),
                                "workload": selected_input.get("workload"),
                                "runtime_input": selected_input.get("runtime_input"),
                                "cached_inputs": selected_cached,
                                "oracle": selected_input.get("oracle"),
                                "profile": selected_input.get("profile"),
                                "selections": selections,
                            }
                        )
        if payload.get("schema") == "loom.application_host_run":
            portfolio_host_runs.append(payload)
        # Nested planning/candidate records project one root event. Counting
        # their statuses again would fabricate outcomes and work.
        if record_kind in {"root", "standalone"}:
            for key in ("disposition", "status", "outcome", "quality_disposition"):
                value = payload.get(key)
                if isinstance(value, str):
                    statuses[value] = statuses.get(value, 0) + 1
        for key, value in payload.items():
            if key in {
                "source_program",
                "fabric",
                "workload",
                "runtime_input",
                "dataflow",
                "system",
                "candidate_identity",
                "policy_digest",
                "frontier_policy_digest",
            } and isinstance(value, str):
                identities.setdefault(key, set()).add(value)
            if record_kind in {"root", "planning", "standalone"} and isinstance(value, dict) and (
                key.endswith("_work")
                or key in {"work_accounting", "checkpoint_work_accounting"}
            ):
                collect_work_entries(
                    value,
                    key,
                    record.get("stage"),
                    record.get("event"),
                    work,
                    ledger_errors,
                )
            if record_kind in {"root", "planning", "standalone"} and key in {"evaluation_cache", "cache", "cache_statistics"} and isinstance(
                value, dict
            ):
                cache_observations.append(
                    {
                        "name": key,
                        "stage": record.get("stage"),
                        "event": record.get("event"),
                        "values": value,
                    }
                )
            if record_kind in {"root", "planning", "standalone"} and key in {
                "duration_ns",
                "elapsed_nanoseconds",
                "wall_seconds",
                "max_rss_kb",
                "observed_wall_ns",
                "requested_wall_time_limit_ms",
                "ticks",
                "simulated_ticks",
                "dfg_ticks",
                "cgra_ticks",
                "accelerator_cycles",
                "spatial_cycles",
                "temporal_cycles",
                "peak_resident_bytes",
            }:
                parsed = value if isinstance(value, (int, float)) else None
                if parsed is not None and not isinstance(parsed, bool):
                    timings.append(
                        {
                            "stage": record.get("stage"),
                            "event": record.get("event"),
                            "operation": payload.get("operation"),
                            "metric": key,
                            "value": parsed,
                        }
                    )
        if record_kind in {"root", "standalone"} and "search_complete" in payload:
            name = "search_complete" if payload["search_complete"] else "search_incomplete"
            statuses[name] = statuses.get(name, 0) + 1
        if record_kind in {"root", "planning", "standalone"} and any(
            key in payload
            for key in (
                "domain_complete",
                "budget_complete",
                "provider_complete",
                "evidence_complete",
                "selection_complete",
            )
        ):
            completeness.append(
                {
                    key: payload[key]
                    for key in (
                        "domain_complete",
                        "budget_complete",
                        "provider_complete",
                        "evidence_complete",
                        "selection_complete",
                    )
                    if key in payload
                }
            )
        nested_completeness = (
            payload.get("completeness")
            if record_kind in {"root", "planning", "standalone"}
            else None
        )
        if isinstance(nested_completeness, dict):
            completeness.append(
                {
                    key: nested_completeness[key]
                    for key in (
                        "domain_complete",
                        "budget_complete",
                        "provider_complete",
                        "evidence_complete",
                        "selection_complete",
                        "exact_complete",
                    )
                    if key in nested_completeness
                }
            )
        if (
            "pre_mapping_candidate_record_ordinal" in payload
            or "planning_record_ordinal" in payload
            or "seed_kinds" in payload
        ):
            candidates.append(
                {
                    key: payload[key]
                    for key in (
                        "pre_mapping_candidate_record_ordinal",
                        "planning_record_ordinal",
                        "candidate_identity",
                        "structured_program",
                        "canonical_dataflow",
                        "seed_kinds",
                        "disposition",
                        "preference_rank",
                        "logical_domain_fact",
                    )
                    if key in payload
                }
            )
        if record_kind in {"root", "planning", "standalone"} and isinstance(
            payload.get("funnel"), dict
        ):
            funnel_summaries.append(payload["funnel"])
        if record_kind in {"root", "planning", "standalone"} and isinstance(
            payload.get("resource_time_funnel"), dict
        ):
            resource_time_funnels.append(
                {
                    key: payload["resource_time_funnel"][key]
                    for key in (
                        "generated_candidates",
                        "screened_candidates",
                        "detailed_frontier_candidates",
                        "successive_halving_deferred_candidates",
                        "sound_gate_rejected_candidates",
                        "estimated_candidates",
                        "incomplete_candidates",
                        "mapping_eligible_schedule_hints",
                        "analytic_shadow_compared_candidates",
                        "analytic_shadow_exact_feasible_candidates",
                        "analytic_shadow_admissible_candidates",
                        "analytic_shadow_feasible_intersection",
                        "analytic_shadow_best_rank_matches",
                        "analytic_shadow_out_of_domain_candidates",
                        "analytic_shadow_maximum_lower_bound_gap_picoseconds",
                        "mapping_finalists",
                        "functional_replay_candidates",
                        "dataflow_projection_requests",
                        "dataflow_projection_cache_hits",
                        "dataflow_projection_cache_misses",
                        "dataflow_projection_cache_capacity_bypasses",
                        "dataflow_projection_cache_entries",
                        "dataflow_projection_cache_retained_bytes",
                        "dataflow_projection_elapsed_nanoseconds",
                        "dataflow_materialized_candidates",
                        "mapping_plan_candidates",
                        "unsupported_before_mapping_candidates",
                        "unsupported_before_mapping_schedule_hints",
                        "application_promotion_accounting_complete",
                        "mapping_calls_avoided_by_sound_gate",
                        "mapping_plan_constructions_avoided_by_exact_memo",
                        "mapping_calls_deferred_by_model",
                        "mapping_calls_withheld_by_incomplete",
                        "incremental_lower_bound_updates",
                        "exact_invocation_memo_hits",
                        "exact_invocation_memo_misses",
                        "exact_invocation_memo_single_flight_waits",
                        "exact_invocation_memo_coalesced_uncached_results",
                        "exact_invocation_memo_cancelled_waits",
                        "exact_invocation_memo_capacity_bypasses",
                        "exact_invocation_memo_entries",
                        "exact_invocation_memo_retained_bytes",
                        "frontier_work",
                        "elapsed_nanoseconds",
                        "truncated",
                    )
                    if key in payload["resource_time_funnel"]
                }
            )
            rows = payload.get("resource_time_evaluations")
            if isinstance(rows, list):
                resource_time_evaluations.extend(
                    {
                        key: row[key]
                        for key in (
                            "candidate_identity",
                            "disposition",
                            "screening_lower_bound_picoseconds",
                            "screening_feature_score",
                            "screening_support",
                            "screening_confidence",
                            "detailed_frontier_evaluated",
                            "minimum_peak_concurrent_regions",
                            "maximum_peak_concurrent_regions",
                            "concurrency_bound_support",
                            "estimated_makespan_picoseconds",
                            "peak_concurrent_regions",
                            "estimate_support",
                            "retained_mapping_schedule_count",
                            "retained_for_mapping",
                        )
                        if key in row
                    }
                    for row in rows
                    if isinstance(row, dict)
                )
        if record_kind in {"root", "planning", "standalone"} and isinstance(
            payload.get("evaluation_timing"), dict
        ):
            evaluation_timings.append(payload["evaluation_timing"])
        if payload.get("domain") in {
            "application_mapping_join",
            "application_pair_decision",
        }:
            if isinstance(payload.get("pair_decision"), dict):
                application_pair_decisions.append(payload["pair_decision"])
        if payload.get("operation") == "simulation_cycle_breakdown":
            simulation_cycle_breakdowns.append(
                {
                    key: payload[key]
                    for key in (
                        "engine",
                        "measurement_kind",
                        "direct",
                        "derived",
                        "request",
                        "cycle_count",
                        "wavefront_steps",
                        "event_count",
                        "dynamic_work_items",
                        "dynamic_operation_fires",
                        "operation_kind_count",
                        "operation_fire_counts",
                        "compute_operation_count",
                        "control_operation_count",
                        "memory_operation_count",
                        "recurrence_carrier_count",
                        "stream_actor_count",
                        "sync_actor_count",
                        "load_count",
                        "store_count",
                        "atomic_memory_operation_count",
                        "fence_count",
                        "modeled_library_call_count",
                        "modeled_instruction_ipc",
                        "modeled_instruction_cpi",
                        "cycles_per_dynamic_work_item",
                        "recurrence_or_ii",
                        "event_frame_count",
                        "empty_event_frame_count",
                        "compute_source_frame_count",
                        "memory_source_frame_count",
                        "transport_source_frame_count",
                        "physical_source_frame_count",
                        "actor_commit_count",
                        "actor_firing_count",
                        "actor_retirement_count",
                        "token_publication_count",
                        "memory_linearization_count",
                        "physical_request_count",
                        "physical_grant_count",
                        "physical_retirement_count",
                        "request_grant_gap",
                        "grant_retirement_gap",
                        "actor_ipc",
                        "actor_cpi",
                        "physical_action_rate",
                        "cycles_per_physical_action",
                        "cycles_per_actor_retirement",
                        "event_frames_per_cycle",
                        "transport_frames_per_cycle",
                        "physical_frames_per_cycle",
                        "memory_load_store_split",
                    )
                    if key in payload
                }
            )
        if payload.get("operation") == "simulation_cycle_comparison":
            simulation_cycle_comparisons.append(
                {
                    key: payload[key]
                    for key in (
                        "measurement_kind",
                        "direct",
                        "derived",
                        "dataflow",
                        "spatial_mapping",
                        "dfg_request",
                        "cgra_request",
                        "dfg_cycles",
                        "cgra_cycles",
                        "cycle_delta",
                        "cgra_to_dfg_ratio",
                        "cgra_is_slower",
                    )
                    if key in payload
                }
            )
        if payload.get("domain") == "application_mapping_join":
            mapping_observations.append(
                {
                    key: payload[key]
                    for key in (
                        "system_pnr_dispatch_count",
                        "actual_system_pnr_attempt_count",
                        "mapping_pair_slots_consumed",
                        "resource_time_schedule_hint_digest",
                        "cold_reopen_wall_time_ns",
                        "incremental_reopen_wall_time_ns",
                        "preserved_tech_mappings",
                        "preserved_spatial_mappings",
                        "repaired_tech_mappings",
                        "repaired_spatial_mappings",
                        "invalidated_tech_mappings",
                        "invalidated_spatial_mappings",
                        "parent_thread_binding_count",
                        "preserved_thread_binding_count",
                        "reopened_thread_binding_count",
                        "parent_graph_binding_count",
                        "preserved_graph_binding_count",
                        "reopened_graph_binding_count",
                        "parent_resource_use_count",
                        "preserved_resource_use_count",
                        "reopened_resource_use_count",
                        "parent_service_realization_count",
                        "preserved_service_realization_count",
                        "reopened_service_realization_count",
                        "mapping_rebase_work",
                        "system_mapping_rebase_work",
                        "joined_max_temporal_outcome_count",
                        "joined_max_spatial_outcome_count",
                        "joined_intermediate_outcome_count",
                        "verified_mapping_max_temporal_count",
                        "verified_mapping_max_spatial_count",
                        "verified_mapping_intermediate_count",
                        "resource_time_generated_candidates",
                        "resource_time_screened_candidates",
                        "resource_time_detailed_frontier_candidates",
                        "resource_time_successive_halving_deferred_candidates",
                        "resource_time_mapping_finalists",
                        "resource_time_mapping_eligible_schedule_hints",
                        "resource_time_analytic_shadow_compared_candidates",
                        "resource_time_analytic_shadow_exact_feasible_candidates",
                        "resource_time_analytic_shadow_admissible_candidates",
                        "resource_time_analytic_shadow_feasible_intersection",
                        "resource_time_analytic_shadow_best_rank_matches",
                        "resource_time_analytic_shadow_out_of_domain_candidates",
                        "resource_time_analytic_shadow_maximum_lower_bound_gap_picoseconds",
                        "application_incremental_mapping_transitions",
                        "resource_time_actual_tech_mapping_dispatch_count",
                        "resource_time_actual_spatial_pnr_dispatch_count",
                        "resource_time_actual_system_pnr_dispatch_count",
                        "tech_mapping_invocation_count",
                        "spatial_pnr_invocation_count",
                        "system_pnr_invocation_count",
                        "tech_mapping_journal_replay_count",
                        "spatial_pnr_journal_replay_count",
                        "system_pnr_journal_replay_count",
                        "resource_time_functional_replay_candidates",
                        "resource_time_dataflow_materialized_candidates",
                        "resource_time_mapping_plan_candidates",
                        "resource_time_mapping_plan_constructions_avoided_by_exact_memo",
                        "resource_time_unsupported_before_mapping_schedule_hints",
                        "resource_time_application_promotion_accounting_complete",
                        "resource_time_mapping_calls_avoided_by_sound_gate",
                        "resource_time_mapping_calls_deferred_by_model",
                        "resource_time_mapping_calls_withheld_by_incomplete",
                        "resource_time_exact_invocation_memo_hits",
                        "resource_time_exact_invocation_memo_misses",
                        "resource_time_exact_invocation_memo_single_flight_waits",
                        "resource_time_exact_invocation_memo_coalesced_uncached_results",
                        "resource_time_exact_invocation_memo_cancelled_waits",
                        "resource_time_exact_invocation_memo_capacity_bypasses",
                        "resource_time_exact_invocation_memo_entries",
                        "resource_time_exact_invocation_memo_retained_bytes",
                        "resource_time_frontier_work",
                        "hardware_parent_promotions",
                        "hardware_reopens_deferred_by_quality",
                        "hardware_reopens_withheld_without_exact_feedback",
                        "hardware_repair_work",
                        "time_to_first_feasible_wall_time_ns",
                        "time_to_best_wall_time_ns",
                    )
                    if key in payload
                }
            )
            if "quality_objective_dimension_labels" in payload:
                quality_summaries.append(
                    {
                        key: payload[key]
                        for key in (
                            "quality_disposition",
                            "quality_objective_dimension_count",
                            "quality_objective_dimension_labels",
                            "quality_complete_observation_count",
                            "quality_incomplete_observation_count",
                        )
                        if key in payload
                    }
                )
            if "quality_observations" in payload and isinstance(
                payload["quality_observations"], list
            ):
                for observation in payload["quality_observations"]:
                    if isinstance(observation, dict):
                        quality_observations.append(
                            {
                                key: observation[key]
                                for key in (
                                    "candidate",
                                    "objective_codes",
                                    "incomplete_reason",
                                )
                                if key in observation
                            }
                        )
            if "plan_ordinal" in payload or "runtime_disposition" in payload:
                application_outcomes.append(
                    {
                        key: payload[key]
                        for key in (
                            "planning_record_ordinal",
                            "candidate_identity",
                            "plan_ordinal",
                            "dataflow",
                            "system",
                            "disposition",
                            "runtime_disposition",
                            "runtime_evidence",
                            "dfg_cycles",
                            "cgra_cycles",
                            "resource_core_cost",
                            "quality_objective_codes",
                            "system_binding_partitions",
                            "resource_time_verified_scenarios",
                            "resource_time_hint_candidates",
                            "resource_time_independent_mapping_imports",
                            "resource_time_mapping_import_requests",
                            "resource_time_mapping_import_cache_hits",
                            "resource_time_mapping_import_cache_misses",
                            "resource_time_mapping_import_retained_bytes",
                            "resource_time_mapping_progress_qualified",
                            "resource_time_mapping_progress_proof_not_established",
                            "resource_time_matching_mapping_checks",
                            "resource_time_materialized_scenarios",
                            "resource_time_unmatched_hints",
                            "resource_time_transition_unsupported_hints",
                            "resource_time_verification_incomplete_reason",
                            "system_mappings",
                            "incomplete_reason",
                        )
                        if key in payload
                    }
                )
        if payload.get("failure_scope") in {
            "application_runtime_validation",
            "application_resource_time_preflight",
            "cgra_simulation_adapter",
            "cgra_simulation_session",
        }:
            runtime_failures.append(
                {
                    key: payload[key]
                    for key in (
                        "failure_scope",
                        "model",
                        "outcome",
                        "reason",
                        "diagnostic",
                        "session_state",
                        "pending_actor_firings",
                        "pending_transfers",
                        "pending_physical_actions",
                        "closed_wait_actors",
                        "closed_wait_transfers",
                        "closed_wait_physical_actions",
                        "operand_queue_group_count",
                        "operand_queue_potentially_blocking_group_count",
                        "operand_queue_unknown_pairing_group_count",
                        "operand_queue_distinct_ingress_count",
                        "operand_queue_pairing_key_count",
                        "operand_queue_progress_status",
                        "operand_queue_progress_support",
                        "operand_queue_projection_digest",
                        "closed_wait_operand_queue_heads",
                        "closed_wait_owners",
                        "closed_wait_transfer_cycle",
                        "closed_wait_actor_cycle",
                    )
                    if key in payload
                }
            )
        if payload.get("operation") in {
            "rebase_mapping_frontier",
            "mapping_rebase_cold_fallback",
            "mapping_rebase_fallback",
            "hardware_reopen_mapping_attempt",
            "resource_time_adjacent_mapping_repair",
            "spatial_operand_queue_runtime_feedback",
            "spatial_fifo_runtime_feedback",
            "spatial_fifo_hardware_repair",
        }:
            migration_observations.append(
                {
                    key: payload[key]
                    for key in (
                        "operation",
                        "parent_mapping",
                        "spatial_mapping",
                        "seed_source",
                        "fallback_reason",
                        "typed_impact_projection_present",
                        "typed_impact_locality",
                        "typed_impact_tech_kind",
                        "typed_impact_spatial_kind",
                        "mapping_reuse_disposition",
                        "hardware_mutation_family",
                        "hardware_mutation_locality",
                        "hardware_tech_impact",
                        "hardware_spatial_impact",
                        "hardware_system_impact",
                        "reopened_root_count",
                        "parent_tech_mappings",
                        "parent_spatial_mappings",
                        "preserved_tech_mappings",
                        "preserved_spatial_mappings",
                        "repaired_tech_mappings",
                        "repaired_spatial_mappings",
                        "invalidated_tech_mappings",
                        "invalidated_spatial_mappings",
                        "mapping_rebase_work",
                        "acc_core_count",
                        "system_mapping_count",
                        "candidate_ordinal",
                        "wall_time_ns",
                        "disposition",
                        "reason",
                        "fifo",
                        "occupancy",
                        "capacity",
                        "minimum_candidate_depth",
                        "bypass_capable",
                        "transfer_cycle_edge_count",
                        "actor_cycle_edge_count",
                        "queue_wait_edge_count",
                        "observed_head_count",
                        "exact_head_count",
                        "matched_pairing_key_count",
                        "unmatched_pairing_key_count",
                        "mismatched_head_count",
                        "full_queue_count",
                        "runtime_projection_digest",
                        "causal_actor",
                        "causal_action",
                        "causal_occurrence",
                        "hardware_child_count",
                        "parent_system",
                        "child_system",
                        "candidate_depth",
                        "rebase_failure_count",
                        "rebase_failures",
                        "liveness",
                        "ii_support",
                        "throughput_support",
                        "latency_support",
                        "timing_fmax_support",
                        "area_support",
                        "power_energy_support",
                        "reconfiguration_support",
                        "bypass_alternative",
                    )
                    if key in payload
                }
            )
        if payload.get("domain") in {
            "pre_mapping_incomplete",
            "resource_time_incomplete",
        }:
            incomplete_observations.append(
                {
                    key: payload[key]
                    for key in (
                        "domain",
                        "reason",
                        "incomplete_reason",
                        "plan_node_ordinal",
                        "checkpoint_boundary",
                        "checkpoint_retained_candidate_count",
                        "checkpoint_work_accounting",
                    )
                    if key in payload
                }
            )
        if payload.get("context_kind") == "resource_time_application_funnel":
            terminal_funnel = payload.get("resource_time_funnel")
            if not isinstance(terminal_funnel, dict):
                terminal_funnel = payload
            resource_time_terminal_observations.append(
                {**({"status": payload["status"]} if "status" in payload else {}), **{
                    key: terminal_funnel[key]
                    for key in (
                        "generated_candidates",
                        "screened_candidates",
                        "detailed_frontier_candidates",
                        "successive_halving_deferred_candidates",
                        "estimated_candidates",
                        "mapping_eligible_schedule_hints",
                        "analytic_shadow_compared_candidates",
                        "analytic_shadow_exact_feasible_candidates",
                        "analytic_shadow_admissible_candidates",
                        "analytic_shadow_feasible_intersection",
                        "analytic_shadow_best_rank_matches",
                        "analytic_shadow_out_of_domain_candidates",
                        "analytic_shadow_maximum_lower_bound_gap_picoseconds",
                        "mapping_finalists",
                        "dataflow_materialized_candidates",
                        "mapping_plan_candidates",
                        "unsupported_before_mapping_candidates",
                        "unsupported_before_mapping_schedule_hints",
                        "mapping_plan_constructions_avoided_by_exact_memo",
                        "incremental_lower_bound_updates",
                        "dataflow_projection_requests",
                        "dataflow_projection_cache_hits",
                        "dataflow_projection_cache_misses",
                        "dataflow_projection_cache_capacity_bypasses",
                        "dataflow_projection_cache_entries",
                        "dataflow_projection_cache_retained_bytes",
                        "dataflow_projection_elapsed_nanoseconds",
                        "mapping_calls_deferred_by_model",
                        "exact_invocation_memo_hits",
                        "exact_invocation_memo_misses",
                        "exact_invocation_memo_single_flight_waits",
                        "exact_invocation_memo_coalesced_uncached_results",
                        "exact_invocation_memo_cancelled_waits",
                        "exact_invocation_memo_capacity_bypasses",
                        "exact_invocation_memo_entries",
                        "exact_invocation_memo_retained_bytes",
                        "frontier_work",
                        "truncated",
                        "incomplete_reason",
                    )
                    if key in terminal_funnel
                }}
            )
    disposition_counts: dict[str, int] = {}
    for candidate in candidates:
        disposition = candidate.get("disposition")
        if isinstance(disposition, str):
            disposition_counts[disposition] = disposition_counts.get(disposition, 0) + 1
    portfolio_pairs = [
        validated
        for decision in application_pair_decisions
        if (validated := validate_portfolio_pair(decision)) is not None
    ]
    portfolio_by_key = {
        (row.get("application_identity"), row.get("input_name")): row
        for row in portfolio_inventory
    }
    portfolio_host_evaluations = [
        validate_portfolio_host_run(
            report,
            portfolio_by_key.get(portfolio_selection_key(report)),
        )
        for report in portfolio_host_runs
    ]
    required_smoke = {
        (row.get("application_identity"), row.get("input_name"))
        for row in portfolio_inventory
        if isinstance(row.get("selections"), list) and "smoke" in row["selections"]
    }
    pairs_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in portfolio_pairs:
        key = (row.get("application_identity"), row.get("input_name"))
        pairs_by_key.setdefault(key, []).append(row)
    hosts_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in portfolio_host_evaluations:
        key = (row.get("application_identity"), row.get("input_name"))
        hosts_by_key.setdefault(key, []).append(row)
    member_evaluations = []
    for application, input_name in sorted(
        required_smoke, key=lambda pair: (str(pair[0]), str(pair[1]))
    ):
        key = (application, input_name)
        pair_rows = pairs_by_key.get(key, [])
        host_rows = hosts_by_key.get(key, [])
        host_complete = bool(host_rows) and all(row["complete"] for row in host_rows)
        typed_pair_complete = bool(pair_rows) and all(
            row["typed_complete"] for row in pair_rows
        )
        canonical_qor_complete = any(row["complete"] for row in pair_rows)
        member_evaluations.append(
            {
                "application_identity": application,
                "input_name": input_name,
                "host_run_count": len(host_rows),
                "pair_decision_count": len(pair_rows),
                "typed_dispositions": sorted(
                    {
                        row["disposition"]
                        for row in pair_rows
                        if isinstance(row.get("disposition"), str)
                    }
                ),
                "host_complete": host_complete,
                "typed_pair_complete": typed_pair_complete,
                "canonical_qor_complete": canonical_qor_complete,
                "member_complete": host_complete and typed_pair_complete,
            }
        )
    missing_canonical_smoke = [
        (row["application_identity"], row["input_name"])
        for row in member_evaluations
        if not row["canonical_qor_complete"]
    ]
    missing_typed_smoke = [
        (row["application_identity"], row["input_name"])
        for row in member_evaluations
        if not row["typed_pair_complete"]
    ]
    missing_host_smoke = [
        (row["application_identity"], row["input_name"])
        for row in member_evaluations
        if not row["host_complete"]
    ]
    return {
        "event_count": len(records),
        "record_kinds": dict(sorted(record_kinds.items())),
        "identities": {key: sorted(values) for key, values in sorted(identities.items())},
        "statuses": dict(sorted(statuses.items())),
        "work": work,
        "ledger": {
            "entries_checked": len(work),
            "invalid_entries": ledger_errors,
            "additive_invariants_hold": not ledger_errors,
        },
        "completeness_observations": completeness,
        "candidate_observations": candidates,
        "candidate_dispositions": dict(sorted(disposition_counts.items())),
        "application_mapping_outcomes": application_outcomes,
        "application_pair_decisions": application_pair_decisions,
        "portfolio": {
            "inventory": portfolio_inventory,
            "host_run_evaluations": portfolio_host_evaluations,
            "pair_evaluations": portfolio_pairs,
            "member_evaluations": member_evaluations,
            "required_smoke_pairs": [
                [row["application_identity"], row["input_name"]]
                for row in member_evaluations
            ],
            "missing_host_smoke_pairs": [list(pair) for pair in missing_host_smoke],
            "missing_typed_smoke_pairs": [list(pair) for pair in missing_typed_smoke],
            "missing_canonical_qor_smoke_pairs": [
                list(pair) for pair in missing_canonical_smoke
            ],
            "host_gates_hold": bool(required_smoke) and not missing_host_smoke,
            "typed_pair_gates_hold": bool(required_smoke) and not missing_typed_smoke,
            "canonical_qor_gates_hold": bool(required_smoke)
            and not missing_canonical_smoke,
            "member_gates_hold": bool(required_smoke)
            and all(row["member_complete"] for row in member_evaluations),
        },
        "quality_summaries": quality_summaries,
        "quality_observations": quality_observations,
        "funnel_summaries": funnel_summaries,
        "resource_time_funnels": resource_time_funnels,
        "evaluation_timings": evaluation_timings,
        "mapping_observations": mapping_observations,
        "timings": timings,
        "cache_observations": cache_observations,
        "runtime_failures": runtime_failures,
        "simulation_cycle_breakdowns": simulation_cycle_breakdowns,
        "simulation_cycle_comparisons": simulation_cycle_comparisons,
        "incomplete_observations": incomplete_observations,
        "resource_time_terminal_observations": resource_time_terminal_observations,
        "resource_time_evaluations": resource_time_evaluations,
        "execution_matrix_observations": execution_matrix_observations,
        "migration_observations": migration_observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    inputs: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    for specification in args.input:
        if "=" not in specification:
            parser.error("--input must be NAME=PATH")
        name, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
        inputs.append(
            {
                "name": name,
                "path": relative_path(path, args.repo_root),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        for record in parse_json_lines(path):
            records.extend(expand_evidence_records(record))

    manifest = {
        "schema": "loom.evidence.manifest.2",
        "repository": repository_facts(args.repo_root),
        "inputs": sorted(inputs, key=lambda item: item["name"]),
        "commands": list(args.command),
        "facts": collect_facts(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
