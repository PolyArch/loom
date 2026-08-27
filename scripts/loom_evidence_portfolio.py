#!/usr/bin/env python3
"""Derive portfolio closure from exact manifest, host, and pair projections."""

from __future__ import annotations

import re
from typing import Any


EXECUTION_SELECTIONS = ("smoke", "validation", "scale_eda")
APPLICATION_OBJECTIVE_DIMENSIONS = (
    "host_only_work",
    "dfg_cycles",
    "cgra_cycles",
    "host_residual_work",
    "cut_transfer_work",
    "launch_synchronization_work",
    "resource_core_cost",
    "mapping_work",
    "area",
    "power",
    "energy",
)
OBJECTIVE_EVIDENCE = {
    "exact",
    "sound_bound",
    "analytic",
    "calibrated",
    "runtime_measured",
    "unsupported",
}
TINYML_TYPED_FALLBACK_DISPOSITION = "unsupported_semantic"
PAIR_DISPOSITIONS = {
    "verified_acceleration",
    "verified_feasible_but_not_beneficial",
    "no_promising_candidate",
    "exact_hardware_incompatible",
    "mapping_proof_not_established",
    "cancelled_or_timeout",
    "budget_exhausted",
    TINYML_TYPED_FALLBACK_DISPOSITION,
    "implementation_failure",
    "hardware_dse_alternative",
}
SUCCESS_DISPOSITIONS = {
    "verified_acceleration",
    "verified_feasible_but_not_beneficial",
    "hardware_dse_alternative",
}
CAUSAL_DISPOSITIONS = PAIR_DISPOSITIONS - SUCCESS_DISPOSITIONS
CANONICAL_QOR_APPLICATIONS = (
    "gapbs-pagerank",
    "llama2c-kernels",
    "loom-multisensor-attention",
    "vecadd-memory",
)
TINYML_APPLICATION = "mlperf-tiny-anomaly-detection"
PRE_ADMISSION_OWNER = "application_build"
PRE_ADMISSION_CONTRACT = "pre_mapping_owner_verified_v1"
PAIR_DECISION_SCHEMA = "loom.application_pair_decision"
PAIR_DECISION_VERSION = "1.0"
PAIR_EVIDENCE_SCHEMA = "loom.application_pair_evidence"
PAIR_DISPOSITION_SCHEMA = "loom.application_pair_disposition"
PAIR_EVIDENCE_VERSION = "1.0"


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _artifact_root(value: Any) -> bool:
    if not isinstance(value, str) or len(value) % 2 != 0:
        return False
    try:
        encoded = bytes.fromhex(value)
    except ValueError:
        return False
    if len(encoded) < 4 + 1 + 4 + 4 + 32:
        return False
    schema_length = int.from_bytes(encoded[:4], "big")
    expected_size = 4 + schema_length + 4 + 4 + 32
    if schema_length == 0 or len(encoded) != expected_size:
        return False
    schema = encoded[4 : 4 + schema_length]
    return all(0x21 <= byte <= 0x7E for byte in schema)


def _artifact_root_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_artifact_root(item) for item in value)
    )


def _pair_decision(evidence: Any) -> dict[str, Any] | None:
    if not isinstance(evidence, dict):
        return None
    decision = evidence.get("pair_decision")
    return decision if isinstance(decision, dict) else None


def _selection_key(selection: Any) -> tuple[Any, Any]:
    if not isinstance(selection, dict):
        return (None, None)
    application = selection.get("application_identity")
    input_name = selection.get("input_name")
    if not isinstance(application, str) or not isinstance(input_name, str):
        return (None, None)
    return (application, input_name)


def portfolio_host_key(report: dict[str, Any]) -> tuple[Any, Any]:
    return _selection_key(report.get("selection"))


def portfolio_pair_key(evidence: dict[str, Any]) -> tuple[Any, Any]:
    decision = _pair_decision(evidence)
    return _selection_key(
        decision.get("portfolio_input") if decision is not None else None
    )


def collect_portfolio_inventory(
    report: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate rows emitted after the canonical C++ manifest parser."""
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if (
        report.get("schema") != "loom.application_portfolio_inventory"
        or report.get("version") != "1.0"
        or report.get("manifest_schema") != "loom.application_portfolio"
        or report.get("manifest_version") != "3.0"
    ):
        return rows, ["unsupported_manifest_inventory_schema"]
    inventory = report.get("rows")
    if not isinstance(inventory, list) or not inventory:
        return rows, ["manifest_inventory_rows_missing"]

    expected_fields = {
        "application_identity",
        "input_name",
        "source",
        "build",
        "workload",
        "runtime_input",
        "input_compiler_options",
        "cached_inputs",
        "oracle",
        "profile",
        "execution_selections",
    }
    previous_key: tuple[str, str] | None = None
    for ordinal, row in enumerate(inventory):
        context = f"row[{ordinal}]"
        error_count = len(errors)
        if not isinstance(row, dict):
            errors.append(f"{context}:malformed")
            continue
        if set(row) != expected_fields:
            errors.append(f"{context}:fields_invalid")
        application = row.get("application_identity")
        input_name = row.get("input_name")
        if not isinstance(application, str) or not isinstance(input_name, str):
            errors.append(f"{context}:identity_invalid")
            continue
        key = (application, input_name)
        if previous_key is not None and key <= previous_key:
            errors.append(f"{context}:order_or_identity_invalid")
        previous_key = key

        selections = row.get("execution_selections")
        if (
            not isinstance(selections, list)
            or not selections
            or any(not isinstance(selection, str) for selection in selections)
        ):
            errors.append(f"{context}:execution_selections_invalid")
        elif selections != [
            selection for selection in EXECUTION_SELECTIONS if selection in selections
        ]:
            errors.append(f"{context}:execution_selections_not_canonical")

        source = row.get("source")
        build = row.get("build")
        oracle = row.get("oracle")
        cached_inputs = row.get("cached_inputs")
        input_compiler_options = row.get("input_compiler_options")
        profile = row.get("profile")
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("kind"), str)
            or not isinstance(source.get("root"), str)
        ):
            errors.append(f"{context}:source_invalid")
        if (
            not isinstance(build, dict)
            or not isinstance(build.get("entry"), str)
            or not isinstance(build.get("language"), str)
            or any(
                not isinstance(build.get(field), list)
                or any(not isinstance(value, str) for value in build[field])
                for field in (
                    "sources",
                    "compiler_options",
                    "link_options",
                    "operator_protocol_symbols",
                )
            )
        ):
            errors.append(f"{context}:build_invalid")
        if not isinstance(row.get("workload"), str) or not isinstance(
            row.get("runtime_input"), str
        ):
            errors.append(f"{context}:workload_invalid")
        if not isinstance(cached_inputs, list) or any(
            not isinstance(cached, dict) for cached in cached_inputs
        ):
            errors.append(f"{context}:cached_inputs_invalid")
        if not isinstance(input_compiler_options, list) or any(
            not isinstance(option, str) for option in input_compiler_options
        ):
            errors.append(f"{context}:input_compiler_options_invalid")
        if (
            not isinstance(oracle, dict)
            or not isinstance(oracle.get("kind"), str)
            or not isinstance(oracle.get("entry"), str)
        ):
            errors.append(f"{context}:oracle_invalid")
        warmup = profile.get("warmup_samples") if isinstance(profile, dict) else None
        measured = (
            profile.get("measured_samples") if isinstance(profile, dict) else None
        )
        deadline = (
            profile.get("deadline_milliseconds") if isinstance(profile, dict) else None
        )
        if (
            _integer(warmup) is None
            or warmup < 0
            or _integer(measured) is None
            or measured <= 0
            or _integer(deadline) is None
            or deadline <= 0
            or not isinstance(profile, dict)
            or profile.get("oracle_coverage") != "all_measured_samples"
        ):
            errors.append(f"{context}:profile_invalid")
        if len(errors) == error_count:
            rows.append(dict(row))
    return rows, errors


def _expected_host_selection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "application_identity": row["application_identity"],
        "input_name": row["input_name"],
        "source": row["source"],
        "build": row["build"],
        "workload": row["workload"],
        "runtime_input": row["runtime_input"],
        "cached_inputs": row["cached_inputs"],
        "input_compiler_options": row["input_compiler_options"],
        "oracle": row["oracle"],
    }


def _expected_pair_selection(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source")
    build = row.get("build")
    profile = row.get("profile")
    if not isinstance(source, dict) or not isinstance(build, dict):
        return {}
    expected_profile = dict(profile) if isinstance(profile, dict) else {}
    warmup = _integer(expected_profile.get("warmup_samples"))
    measured = _integer(expected_profile.get("measured_samples"))
    if warmup is not None and measured is not None:
        expected_profile["total_samples"] = warmup + measured
    return {
        "application_identity": row["application_identity"],
        "input_name": row["input_name"],
        "source_kind": source.get("kind"),
        "source_root": source.get("root"),
        "build_entry": build.get("entry"),
        "language": build.get("language"),
        "sources": build.get("sources"),
        "compiler_options": build.get("compiler_options"),
        "input_compiler_options": row.get("input_compiler_options"),
        "link_options": build.get("link_options"),
        "operator_protocol_symbols": build.get("operator_protocol_symbols"),
        "declared_workload": row["workload"],
        "declared_runtime_input": row["runtime_input"],
        "cached_inputs": row["cached_inputs"],
        "declared_oracle": row["oracle"],
        "declared_profile": expected_profile,
    }


def validate_portfolio_host_run(
    report: dict[str, Any], expected: dict[str, Any] | None
) -> dict[str, Any]:
    selection = report.get("selection")
    application, input_name = _selection_key(selection)
    execution_selection = report.get("__portfolio_execution_selection")
    reasons: list[str] = []
    if (
        report.get("schema") != "loom.application_host_run"
        or report.get("version") != "1.0"
    ):
        reasons.append("invalid_report_schema")
    if not isinstance(application, str) or not isinstance(input_name, str):
        reasons.append("invalid_selection")
    if (
        execution_selection is not None
        and execution_selection not in EXECUTION_SELECTIONS
    ):
        reasons.append("invalid_execution_selection")
    if expected is None:
        reasons.append("manifest_selection_missing")
    else:
        if selection != _expected_host_selection(expected):
            reasons.append("manifest_selection_mismatch")
        if report.get("profile") != expected.get("profile"):
            reasons.append("manifest_profile_mismatch")
        if execution_selection is not None and execution_selection not in expected.get(
            "execution_selections", []
        ):
            reasons.append("manifest_execution_selection_mismatch")

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
        or _integer(wall_time) is None
        or wall_time <= 0
    ):
        reasons.append("host_execution_incomplete")
    elif expected is not None:
        profile = expected.get("profile")
        deadline = (
            profile.get("deadline_milliseconds") if isinstance(profile, dict) else None
        )
        if _integer(deadline) is None or wall_time > deadline * 1_000_000:
            reasons.append("host_deadline_exceeded")
    oracle = report.get("oracle_result")
    if not isinstance(oracle, dict) or oracle.get("status") != "matched":
        reasons.append("host_oracle_incomplete")
    if report.get("outcome") != "succeeded":
        reasons.append("host_outcome_incomplete")
    return {
        "application_identity": application,
        "input_name": input_name,
        "execution_selection": execution_selection,
        "outcome": report.get("outcome"),
        "host_wall_time_nanoseconds": wall_time,
        "complete": not reasons,
        "incomplete_reasons": reasons,
    }


def _validate_objective_vector(value: Any, context: str) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    unsupported: list[str] = []
    if not isinstance(value, list):
        return [f"{context}_missing"], unsupported
    dimensions = [
        observation.get("dimension") if isinstance(observation, dict) else None
        for observation in value
    ]
    if dimensions != list(APPLICATION_OBJECTIVE_DIMENSIONS):
        reasons.append(f"{context}_dimensions_invalid")
    for observation in value:
        if not isinstance(observation, dict):
            reasons.append(f"{context}_observation_invalid")
            continue
        dimension = observation.get("dimension")
        evidence = observation.get("evidence")
        measured = observation.get("value")
        confidence = _integer(observation.get("confidence_permille"))
        if evidence not in OBJECTIVE_EVIDENCE:
            reasons.append(f"{context}_{dimension}_evidence_invalid")
            continue
        if confidence is None or confidence < 0 or confidence > 1000:
            reasons.append(f"{context}_{dimension}_confidence_invalid")
        if not isinstance(observation.get("out_of_distribution"), bool):
            reasons.append(f"{context}_{dimension}_ood_invalid")
        if evidence == "unsupported":
            unsupported.append(str(dimension))
            if measured is not None or confidence != 0:
                reasons.append(f"{context}_{dimension}_unsupported_has_value")
        elif _integer(measured) is None or measured < 0 or confidence == 0:
            reasons.append(f"{context}_{dimension}_measurement_invalid")
        elif evidence in {"exact", "runtime_measured"} and confidence != 1000:
            reasons.append(f"{context}_{dimension}_confidence_invalid")
    return reasons, unsupported


def _validate_work_ledger(value: Any, context: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{context}_missing"]
    names = ("limit", "planned", "reserved", "consumed", "rejected", "cancelled")
    parsed = {name: _integer(value.get(name)) for name in names}
    if any(number is None or number < 0 for number in parsed.values()):
        return [f"{context}_invalid"]
    if (
        parsed["reserved"] != parsed["planned"]
        or parsed["consumed"] + parsed["rejected"] + parsed["cancelled"]
        != parsed["reserved"]
        or parsed["planned"] > parsed["limit"]
    ):
        return [f"{context}_invalid"]
    return []


def _validate_counter_object(
    value: Any, fields: tuple[str, ...], context: str
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{context}_missing"]
    if any(_integer(value.get(field)) is None or value[field] < 0 for field in fields):
        return [f"{context}_invalid"]
    return []


def _validate_pair_evidence_envelope(
    evidence: dict[str, Any], decision: dict[str, Any]
) -> tuple[list[str], bool]:
    reasons: list[str] = []
    schema = evidence.get("schema")
    version = evidence.get("version")
    if version != PAIR_EVIDENCE_VERSION:
        reasons.append("pair_evidence_version_invalid")
    if schema == PAIR_DISPOSITION_SCHEMA:
        if evidence.get("domain") != "application_pair_decision":
            reasons.append("pair_disposition_domain_invalid")
        return reasons, False
    if schema != PAIR_EVIDENCE_SCHEMA:
        reasons.append("pair_evidence_schema_invalid")
        return reasons, False
    if evidence.get("domain") != "application_mapping_join":
        reasons.append("pair_evidence_domain_invalid")

    for field in ("source_program", "fabric", "workload", "runtime_input"):
        if evidence.get(field) != decision.get(field):
            reasons.append(f"pair_evidence_{field}_mismatch")
    selected_mapping = evidence.get("selected_mapping")
    if not _artifact_root(selected_mapping) or selected_mapping != decision.get(
        "selected_system_mapping"
    ):
        reasons.append("selected_mapping_checkpoint_mismatch")
    selected_plan = _integer(evidence.get("selected_plan_ordinal"))
    if selected_plan is None or selected_plan < 0:
        reasons.append("selected_plan_checkpoint_missing")

    for stage in ("tech_mapping", "spatial_pnr", "system_pnr"):
        invocations = _integer(evidence.get(f"{stage}_invocation_count"))
        dispatches = _integer(evidence.get(f"{stage}_dispatch_count"))
        replays = _integer(evidence.get(f"{stage}_journal_replay_count"))
        if (
            invocations is None
            or invocations <= 0
            or dispatches is None
            or dispatches < 0
            or replays is None
            or replays < 0
            or dispatches + replays < invocations
        ):
            reasons.append(f"{stage}_work_missing")

    eligible = _integer(evidence.get("eligible_joint_pair_count"))
    evaluated = _integer(evidence.get("analytic_evaluated_joint_pair_count"))
    retained = _integer(evidence.get("retained_joint_pair_count"))
    if (
        eligible is None
        or evaluated is None
        or retained is None
        or eligible <= 0
        or evaluated < retained
        or eligible < evaluated
        or retained <= 0
    ):
        reasons.append("candidate_gate_counts_invalid")
    analytics = evidence.get("retained_joint_pair_analytics")
    if not isinstance(analytics, list) or not analytics:
        reasons.append("candidate_estimates_missing")
    else:
        for observation in analytics:
            if (
                not isinstance(observation, dict)
                or not _artifact_root(observation.get("dataflow"))
                or not _artifact_root(observation.get("system"))
                or _integer(observation.get("estimated_work_units")) is None
                or not isinstance(observation.get("confidence"), str)
            ):
                reasons.append("candidate_estimate_invalid")
                break

    reasons.extend(
        _validate_work_ledger(
            evidence.get("hardware_repair_work"), "hardware_repair_work"
        )
    )
    reasons.extend(
        _validate_work_ledger(
            evidence.get("spatial_mapping_repair_work"),
            "spatial_mapping_repair_work",
        )
    )
    reasons.extend(
        _validate_counter_object(
            evidence.get("mapping_rebase_work"),
            (
                "parent_tech_decisions",
                "parent_spatial_decisions",
                "preserved_tech_decisions",
                "preserved_spatial_decisions",
                "reopened_tech_decisions",
                "reopened_spatial_decisions",
                "repaired_tech_decisions",
                "repaired_spatial_decisions",
                "invalidation_root_count",
                "invalidation_cone_decision_count",
            ),
            "mapping_failure_cone",
        )
    )
    reasons.extend(
        _validate_counter_object(
            evidence.get("system_mapping_rebase_work"),
            (
                "parent_thread_binding_count",
                "preserved_thread_binding_count",
                "reopened_thread_binding_count",
                "parent_graph_binding_count",
                "preserved_graph_binding_count",
                "reopened_graph_binding_count",
            ),
            "system_mapping_failure_cone",
        )
    )
    verified = _integer(evidence.get("verified_alternatives"))
    joined = _integer(evidence.get("joined_candidate_identity_count"))
    outcomes = _integer(evidence.get("outcome_count"))
    if verified is None or verified <= 0 or joined is None or joined <= 0:
        reasons.append("verified_mapping_inventory_missing")
    if outcomes is None or outcomes < joined:
        reasons.append("mapping_outcome_inventory_invalid")
    if (
        evidence.get("resource_time_application_promotion_accounting_complete")
        is not True
    ):
        reasons.append("hardware_promotion_accounting_incomplete")
    for field in (
        "domain_complete",
        "budget_complete",
        "provider_complete",
        "evidence_complete",
        "selection_complete",
        "joint_frontier_truncated",
        "declared_work_exhausted",
    ):
        if not isinstance(evidence.get(field), bool):
            reasons.append(f"{field}_missing")
    return reasons, True


def validate_portfolio_pair(
    evidence: dict[str, Any], expected: dict[str, Any] | None
) -> dict[str, Any] | None:
    decision = _pair_decision(evidence)
    if decision is None:
        return None
    selection = decision.get("portfolio_input")
    if not isinstance(selection, dict):
        return None
    application, input_name = _selection_key(selection)
    typed_reasons: list[str] = []
    closure_reasons: list[str] = []
    if (
        decision.get("schema") != PAIR_DECISION_SCHEMA
        or decision.get("version") != PAIR_DECISION_VERSION
    ):
        typed_reasons.append("pair_decision_schema_invalid")
    envelope_reasons, has_mapping_evidence = _validate_pair_evidence_envelope(
        evidence, decision
    )
    typed_reasons.extend(envelope_reasons)
    if not isinstance(application, str) or not isinstance(input_name, str):
        typed_reasons.append("invalid_selection")
    if expected is None:
        typed_reasons.append("manifest_selection_missing")
    elif selection != {
        **_expected_pair_selection(expected),
        "execution_binding": selection.get("execution_binding"),
        "execution_binding_established": selection.get("execution_binding_established"),
    }:
        typed_reasons.append("manifest_selection_mismatch")

    disposition = decision.get("disposition")
    if disposition not in PAIR_DISPOSITIONS:
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
        if decision.get("manifest_join_owner") != PRE_ADMISSION_OWNER:
            typed_reasons.append("manifest_join_owner_invalid")
        if decision.get("manifest_join_contract") != PRE_ADMISSION_CONTRACT:
            typed_reasons.append("manifest_join_contract_invalid")
    else:
        run_key = decision.get("invocation_manifest_run_key")
        if not _digest(run_key):
            typed_reasons.append("manifest_run_key_missing")
    if not _artifact_root(decision.get("fabric")):
        typed_reasons.append("fabric_invalid")
    if disposition in CAUSAL_DISPOSITIONS and (
        not isinstance(decision.get("detail"), str) or not decision["detail"]
    ):
        typed_reasons.append("typed_failure_detail_missing")

    if (
        selection.get("execution_binding") != "canonical_simulation_and_oracle"
        or selection.get("execution_binding_established") is not True
    ):
        closure_reasons.append("execution_binding_incomplete")
    if decision.get("host_only_baseline_complete") is not True:
        closure_reasons.append("host_baseline_incomplete")
    if decision.get("final_application_qor_complete") is not True:
        closure_reasons.append("application_qor_incomplete")
    if join_status not in {"exact", "owner_scoped_planning_closure"}:
        closure_reasons.append("invocation_manifest_join_incomplete")

    baseline_reasons, _ = _validate_objective_vector(
        decision.get("host_only_baseline"), "host_baseline"
    )
    selected_reasons, unsupported_dimensions = _validate_objective_vector(
        decision.get("selected_objective"), "selected_objective"
    )
    typed_reasons.extend(baseline_reasons)
    typed_reasons.extend(selected_reasons)
    baseline = decision.get("host_only_baseline")
    if isinstance(baseline, list) and baseline:
        host_work = baseline[0]
        if (
            not isinstance(host_work, dict)
            or _integer(host_work.get("value")) is None
            or host_work.get("evidence") == "unsupported"
        ):
            closure_reasons.append("host_work_observation_missing")
    selected_objective = decision.get("selected_objective")
    if isinstance(selected_objective, list) and len(selected_objective) >= 3:
        for ordinal, dimension in ((1, "dfg_cycles"), (2, "cgra_cycles")):
            observation = selected_objective[ordinal]
            if (
                not isinstance(observation, dict)
                or observation.get("dimension") != dimension
                or _integer(observation.get("value")) is None
                or observation.get("evidence") != "runtime_measured"
            ):
                closure_reasons.append(f"{dimension}_measurement_missing")
    if isinstance(selected_objective, list) and len(selected_objective) >= 8:
        mapping_work = selected_objective[7]
        if (
            not isinstance(mapping_work, dict)
            or mapping_work.get("dimension") != "mapping_work"
            or _integer(mapping_work.get("value")) is None
            or mapping_work.get("evidence") != "runtime_measured"
        ):
            closure_reasons.append("mapping_work_measurement_missing")

    candidates = decision.get("candidates")
    selected_candidates: list[dict[str, Any]] = []
    if not isinstance(candidates, list):
        typed_reasons.append("candidate_inventory_missing")
        candidates = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            typed_reasons.append("candidate_record_invalid")
            continue
        candidate_reasons, _ = _validate_objective_vector(
            candidate.get("objective"), "candidate_objective"
        )
        typed_reasons.extend(candidate_reasons)
        if candidate.get("selected") is True:
            selected_candidates.append(candidate)
    planning_count = _integer(decision.get("planning_record_count"))
    noncandidate_count = _integer(decision.get("non_candidate_planning_record_count"))
    if (
        planning_count is None
        or noncandidate_count is None
        or planning_count < len(candidates)
        or noncandidate_count != planning_count - len(candidates)
    ):
        typed_reasons.append("candidate_inventory_count_mismatch")
    if len(selected_candidates) != 1:
        closure_reasons.append("selected_candidate_not_unique")
    else:
        selected_candidate = selected_candidates[0]
        if not _digest(
            selected_candidate.get("candidate_identity")
        ) or selected_candidate.get("candidate_identity") != decision.get(
            "selected_candidate_identity"
        ):
            typed_reasons.append("selected_candidate_identity_mismatch")
        observations = selected_candidate.get("mapping_observations")
        selected_plan = _integer(evidence.get("selected_plan_ordinal"))
        selected_mapping = decision.get("selected_system_mapping")
        selected_observations = (
            [
                observation
                for observation in observations
                if isinstance(observation, dict)
                and observation.get("plan_ordinal") == selected_plan
                and isinstance(observation.get("system_mappings"), list)
                and selected_mapping in observation["system_mappings"]
            ]
            if isinstance(observations, list)
            else []
        )
        if selected_candidate.get("plan_ordinal") != selected_plan:
            typed_reasons.append("selected_candidate_plan_mismatch")
        if len(selected_observations) != 1:
            typed_reasons.append("selected_mapping_observation_not_unique")
            closure_reasons.append("selected_mapping_evidence_incomplete")
        else:
            selected_observation = selected_observations[0]
            if (
                selected_observation.get("mapping_disposition") != "verified"
                or selected_observation.get("runtime_disposition") != "completed"
                or not _artifact_root(selected_observation.get("system"))
                or not _digest(selected_observation.get("schedule_hint_digest"))
                or not _artifact_root_list(selected_observation.get("system_mappings"))
                or not _artifact_root_list(selected_observation.get("runtime_evidence"))
                or not _artifact_root_list(selected_observation.get("oracle_evidence"))
            ):
                closure_reasons.append("selected_mapping_evidence_incomplete")
            if decision.get("selected_system") != selected_observation.get("system"):
                typed_reasons.append("selected_system_mismatch")
            if not _artifact_root(selected_mapping):
                typed_reasons.append("selected_mapping_mismatch")
            if disposition in SUCCESS_DISPOSITIONS:
                objective_by_dimension = {
                    observation.get("dimension"): observation
                    for observation in (
                        selected_objective
                        if isinstance(selected_objective, list)
                        else []
                    )
                    if isinstance(observation, dict)
                }
                for dimension, field in (
                    ("dfg_cycles", "dfg_cycles"),
                    ("cgra_cycles", "cgra_cycles"),
                    ("resource_core_cost", "resource_core_cost"),
                ):
                    measured = selected_observation.get(field)
                    objective = objective_by_dimension.get(dimension)
                    if (
                        _integer(measured) is None
                        or not isinstance(objective, dict)
                        or objective.get("evidence") != "runtime_measured"
                        or objective.get("value") != measured
                    ):
                        typed_reasons.append(f"selected_{dimension}_join_mismatch")

    if disposition in SUCCESS_DISPOSITIONS:
        if not has_mapping_evidence:
            typed_reasons.append("success_mapping_evidence_missing")
        for field in ("pair_identity", "selected_candidate_identity"):
            if not _digest(decision.get(field)):
                typed_reasons.append(f"success_{field}_missing")
        for field in (
            "source_program",
            "workload",
            "runtime_input",
            "selected_system",
            "selected_system_mapping",
        ):
            if not _artifact_root(decision.get(field)):
                typed_reasons.append(f"success_{field}_missing")
        typed_reasons.extend(f"success_{reason}" for reason in closure_reasons)
    typed_reasons = list(dict.fromkeys(typed_reasons))
    closure_reasons = list(dict.fromkeys(closure_reasons))
    return {
        "application_identity": application,
        "input_name": input_name,
        "disposition": disposition,
        "typed_complete": not typed_reasons,
        "typed_incomplete_reasons": typed_reasons,
        "canonical_qor_complete": disposition in SUCCESS_DISPOSITIONS
        and not typed_reasons
        and not closure_reasons,
        "closure_residuals": closure_reasons,
        "unsupported_objective_dimensions": unsupported_dimensions,
    }


def evaluate_portfolio(
    inventory: list[dict[str, Any]],
    host_runs: list[dict[str, Any]],
    pair_evidence: list[dict[str, Any]],
    manifest_errors: list[str],
) -> dict[str, Any]:
    inventory_by_key = {
        (row.get("application_identity"), row.get("input_name")): row
        for row in inventory
    }
    host_evaluations = [
        validate_portfolio_host_run(
            report, inventory_by_key.get(portfolio_host_key(report))
        )
        for report in host_runs
    ]
    pair_evaluations = [
        validated
        for evidence in pair_evidence
        if (
            validated := validate_portfolio_pair(
                evidence, inventory_by_key.get(portfolio_pair_key(evidence))
            )
        )
        is not None
    ]
    hosts_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in host_evaluations:
        hosts_by_key.setdefault(
            (row.get("application_identity"), row.get("input_name")), []
        ).append(row)
    pairs_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in pair_evaluations:
        pairs_by_key.setdefault(
            (row.get("application_identity"), row.get("input_name")), []
        ).append(row)

    member_evaluations: list[dict[str, Any]] = []
    for row in inventory:
        key = (row.get("application_identity"), row.get("input_name"))
        hosts = hosts_by_key.get(key, [])
        pairs = pairs_by_key.get(key, [])
        host_selection_complete = {
            selection: bool(
                selected_hosts := [
                    host
                    for host in hosts
                    if host.get("execution_selection") == selection
                ]
            )
            and all(host["complete"] for host in selected_hosts)
            for selection in row.get("execution_selections", [])
        }
        host_conformance_complete = bool(host_selection_complete) and all(
            host_selection_complete.values()
        )
        typed_pair_complete = bool(pairs) and all(
            pair["typed_complete"] for pair in pairs
        )
        canonical_application_qor_complete = any(
            pair["canonical_qor_complete"] for pair in pairs
        )
        application = key[0]
        tinyml_typed_fallback_complete = (
            application == TINYML_APPLICATION
            and typed_pair_complete
            and not canonical_application_qor_complete
            and all(
                pair.get("disposition") == TINYML_TYPED_FALLBACK_DISPOSITION
                and set(pair.get("unsupported_objective_dimensions", []))
                == set(APPLICATION_OBJECTIVE_DIMENSIONS)
                for pair in pairs
            )
        )
        member_evaluations.append(
            {
                "application_identity": application,
                "input_name": key[1],
                "execution_selections": row.get("execution_selections", []),
                "host_run_count": len(hosts),
                "unscoped_host_run_count": sum(
                    host.get("execution_selection") is None for host in hosts
                ),
                "host_selection_complete": host_selection_complete,
                "pair_decision_count": len(pairs),
                "typed_dispositions": sorted(
                    {
                        pair["disposition"]
                        for pair in pairs
                        if isinstance(pair.get("disposition"), str)
                    }
                ),
                "host_conformance_complete": host_conformance_complete,
                "accelerator_disposition_complete": typed_pair_complete,
                "canonical_application_qor_complete": (
                    canonical_application_qor_complete
                ),
                "tinyml_typed_fallback_complete": (tinyml_typed_fallback_complete),
                "typed_residuals": sorted(
                    {
                        residual
                        for pair in pairs
                        for residual in pair["closure_residuals"]
                    }
                ),
                "unsupported_objective_dimensions": sorted(
                    {
                        dimension
                        for pair in pairs
                        for dimension in pair["unsupported_objective_dimensions"]
                    }
                ),
                "portfolio_requirement_complete": host_conformance_complete
                and typed_pair_complete
                and (
                    tinyml_typed_fallback_complete or canonical_application_qor_complete
                ),
            }
        )

    selection_evaluations: list[dict[str, Any]] = []
    for selection in EXECUTION_SELECTIONS:
        required = [
            row
            for row in member_evaluations
            if selection in row["execution_selections"]
        ]
        missing_host = [
            [row["application_identity"], row["input_name"]]
            for row in required
            if not row["host_selection_complete"].get(selection, False)
        ]
        missing_typed = [
            [row["application_identity"], row["input_name"]]
            for row in required
            if not row["accelerator_disposition_complete"]
        ]
        qor_required = [
            row for row in required if row["application_identity"] != TINYML_APPLICATION
        ]
        missing_qor = [
            [row["application_identity"], row["input_name"]]
            for row in qor_required
            if not row["canonical_application_qor_complete"]
        ]
        selection_evaluations.append(
            {
                "selection": selection,
                "required_pairs": [
                    [row["application_identity"], row["input_name"]] for row in required
                ],
                "missing_host_pairs": missing_host,
                "missing_typed_pair_decisions": missing_typed,
                "canonical_qor_required_pairs": [
                    [row["application_identity"], row["input_name"]]
                    for row in qor_required
                ],
                "missing_canonical_qor_pairs": missing_qor,
                "host_conformance_gate_holds": bool(required) and not missing_host,
                "accelerator_disposition_gate_holds": bool(required)
                and not missing_typed,
                "canonical_application_qor_gate_holds": bool(qor_required)
                and not missing_qor,
                "portfolio_requirement_gate_holds": bool(required)
                and all(row["portfolio_requirement_complete"] for row in required),
            }
        )

    members_by_application: dict[str, list[dict[str, Any]]] = {}
    for row in member_evaluations:
        application = row.get("application_identity")
        if isinstance(application, str):
            members_by_application.setdefault(application, []).append(row)
    canonical_witnesses = {
        application: any(
            row["canonical_application_qor_complete"]
            for row in members_by_application.get(application, [])
        )
        for application in CANONICAL_QOR_APPLICATIONS
    }
    tinyml_rows = members_by_application.get(TINYML_APPLICATION, [])
    tinyml_profile_holds = bool(tinyml_rows) and any(
        row["host_conformance_complete"] and row["tinyml_typed_fallback_complete"]
        for row in tinyml_rows
    )
    all_selection_gates_hold = all(
        row["portfolio_requirement_gate_holds"] for row in selection_evaluations
    )
    acceptance = {
        "canonical_qor_witnesses": canonical_witnesses,
        "canonical_qor_witnesses_hold": all(canonical_witnesses.values()),
        "tinyml_host_conformance_and_typed_fallback_hold": tinyml_profile_holds,
        "all_execution_selection_gates_hold": all_selection_gates_hold,
        "portfolio_acceptance_holds": not manifest_errors
        and all_selection_gates_hold
        and all(canonical_witnesses.values())
        and tinyml_profile_holds,
    }
    return {
        "manifest_errors": manifest_errors,
        "inventory": inventory,
        "host_run_evaluations": host_evaluations,
        "pair_evaluations": pair_evaluations,
        "member_evaluations": member_evaluations,
        "selection_evaluations": selection_evaluations,
        "acceptance": acceptance,
        "host_conformance_gates_hold": bool(member_evaluations)
        and all(row["host_conformance_complete"] for row in member_evaluations),
        "accelerator_disposition_gates_hold": bool(member_evaluations)
        and all(row["accelerator_disposition_complete"] for row in member_evaluations),
        "canonical_application_qor_gates_hold": all(canonical_witnesses.values()),
        "portfolio_requirement_gates_hold": bool(member_evaluations)
        and all(row["portfolio_requirement_complete"] for row in member_evaluations),
    }
