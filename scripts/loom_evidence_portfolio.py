#!/usr/bin/env python3
"""Derive portfolio closure from exact manifest, host, and pair projections."""

from __future__ import annotations

import re
from pathlib import Path
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
UNSUPPORTED_SEMANTIC_DISPOSITION = "unsupported_semantic"
PAIR_DISPOSITIONS = {
    "verified_acceleration",
    "verified_feasible_but_not_beneficial",
    "no_promising_candidate",
    "exact_hardware_incompatible",
    "mapping_proof_not_established",
    "cancelled_or_timeout",
    "budget_exhausted",
    UNSUPPORTED_SEMANTIC_DISPOSITION,
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
# Join-status spellings are owned by ApplicationPairManifestJoinStatus in
# lib/Application/PairDecision.cpp; the third owner value, "missing", never
# closes a join.
MANIFEST_JOIN_COMPLETE = "owner_scoped_planning_closure"
MANIFEST_JOIN_PRE_ADMISSION = "owner_verified_pre_admission"

_ROOT = Path(__file__).resolve().parents[1]
_PAIR_DIAGNOSTIC_OWNER = (
    _ROOT / "lib/Application/BuildDiagnosticsInternal.h"
).read_text(encoding="utf-8")


def _owned_projection_literal(name: str) -> str:
    value = re.search(
        rf'\b{re.escape(name)}\s*=\s*"([^"]+)"',
        _PAIR_DIAGNOSTIC_OWNER,
    )
    if value is None:
        raise RuntimeError("application pair diagnostic ABI owner is malformed")
    return value.group(1)


PAIR_DECISION_SCHEMA = _owned_projection_literal(
    "applicationPairDecisionSchemaIdentity"
)
PAIR_DECISION_VERSION = _owned_projection_literal(
    "applicationPairDecisionSchemaVersion"
)
PAIR_EVIDENCE_SCHEMA = _owned_projection_literal(
    "applicationPairEvidenceSchemaIdentity"
)
PAIR_EVIDENCE_VERSION = _owned_projection_literal(
    "applicationPairEvidenceSchemaVersion"
)
PAIR_DISPOSITION_SCHEMA = _owned_projection_literal(
    "applicationPairDispositionSchemaIdentity"
)
PAIR_DISPOSITION_VERSION = _owned_projection_literal(
    "applicationPairDispositionSchemaVersion"
)
RUNTIME_BINDING_SCHEMA = _owned_projection_literal(
    "applicationRuntimeBindingSchemaIdentity"
)
RUNTIME_BINDING_VERSION = _owned_projection_literal(
    "applicationRuntimeBindingSchemaVersion"
)
_runtime_manifest_schema = re.search(
    r'\bapplicationRuntimeManifestSchema\s*\{\s*"([^"]+)",\s*'
    r'SchemaVersion\{(\d+),\s*(\d+)\}\s*\}',
    (_ROOT / "include/Application/RuntimeManifest.h").read_text(encoding="utf-8"),
)
if _runtime_manifest_schema is None:
    raise RuntimeError("application runtime manifest schema owner is malformed")
RUNTIME_MANIFEST_SCHEMA = _runtime_manifest_schema[1]
RUNTIME_MANIFEST_VERSION = (
    f"{_runtime_manifest_schema[2]}.{_runtime_manifest_schema[3]}"
)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def decode_artifact_root_hex(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or len(value) % 2 != 0:
        return None
    try:
        encoded = bytes.fromhex(value)
    except ValueError:
        return None
    if len(encoded) < 4 + 1 + 4 + 4 + 32:
        return None
    schema_length = int.from_bytes(encoded[:4], "big")
    expected_size = 4 + schema_length + 4 + 4 + 32
    if schema_length == 0 or len(encoded) != expected_size:
        return None
    schema = encoded[4 : 4 + schema_length]
    if not all(0x21 <= byte <= 0x7E for byte in schema):
        return None
    return {
        "schema": schema.decode("ascii"),
        "schema_version": (
            f"{int.from_bytes(encoded[4 + schema_length : 8 + schema_length], 'big')}."
            f"{int.from_bytes(encoded[8 + schema_length : 12 + schema_length], 'big')}"
        ),
        "artifact": encoded[12 + schema_length :].hex(),
    }


def _artifact_root(value: Any) -> bool:
    return decode_artifact_root_hex(value) is not None


def _selection_spectrum_valid(spectrum: Any) -> bool:
    if spectrum is None:
        return True
    if not isinstance(spectrum, dict):
        return False
    disposition = spectrum.get("disposition")
    scenarios = spectrum.get("scenarios")
    if disposition == "verified":
        return (
            _artifact_root(spectrum.get("dataflow"))
            and _artifact_root(spectrum.get("fabric"))
            and isinstance(scenarios, list)
            and bool(scenarios)
            and all(
                isinstance(scenario, dict)
                and scenario.get("spectrum_class")
                in {"max_temporal", "max_spatial", "intermediate"}
                and isinstance(scenario.get("system_mappings"), list)
                and bool(scenario["system_mappings"])
                and all(_artifact_root(mapping) for mapping in scenario["system_mappings"])
                for scenario in scenarios
            )
        )
    return (
        disposition
        in {"unsupported", "proof_not_established", "cancelled_or_timeout"}
        and isinstance(spectrum.get("diagnostic"), str)
        and bool(spectrum["diagnostic"])
        and scenarios == []
        and spectrum.get("dataflow") is None
        and spectrum.get("fabric") is None
    )


def _selection_spectrum_admits(
    spectrum: Any, mapping: str, endpoint: str
) -> bool:
    if not isinstance(spectrum, dict) or spectrum.get("disposition") != "verified":
        return False
    scenarios = spectrum.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        return False
    requested_class = None if endpoint == "automatic" else endpoint
    return any(
        isinstance(scenario, dict)
        and isinstance(scenario.get("system_mappings"), list)
        and mapping in scenario["system_mappings"]
        and (
            requested_class is None
            or scenario.get("spectrum_class") == requested_class
        )
        for scenario in scenarios
    )


def validate_resource_time_mapping_repair_transition(
    transition: Any,
) -> tuple[bool, list[str]]:
    """Derive one adjacent Mapping-repair status from its exact side evidence."""
    errors: list[str] = []
    if not isinstance(transition, dict):
        return False, ["not_object"]
    endpoint = transition.get("spectrum_endpoint")
    if endpoint not in {"automatic", "max_temporal", "max_spatial", "intermediate"}:
        errors.append("spectrum_endpoint_invalid")
        endpoint = "automatic"
    root_arrays = (
        "cold_mapping_candidates",
        "incremental_mapping_candidates",
        "cold_eligible_mappings",
        "incremental_eligible_mappings",
        "cold_runtime_evidence",
        "cold_oracle_evidence",
        "incremental_runtime_evidence",
        "incremental_oracle_evidence",
    )
    reason_arrays = (
        "cold_execution_incomplete_reasons",
        "incremental_execution_incomplete_reasons",
    )
    for field in root_arrays:
        values = transition.get(field)
        if (
            not isinstance(values, list)
            or any(not _artifact_root(value) for value in values)
            or len(values) != len(set(values))
        ):
            errors.append(f"{field}_invalid")
    for field in reason_arrays:
        values = transition.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            errors.append(f"{field}_invalid")

    reopened_roots = transition.get("reopened_roots")
    reopened_keys: list[tuple[str, int]] = []
    if isinstance(reopened_roots, list):
        for root in reopened_roots:
            if (
                not isinstance(root, dict)
                or not _digest(root.get("artifact"))
                or _integer(root.get("entity")) is None
                or root["entity"] < 0
            ):
                errors.append("reopened_root_invalid")
                continue
            reopened_keys.append((root["artifact"], root["entity"]))
    if (
        not isinstance(reopened_roots, list)
        or not reopened_roots
        or transition.get("reopened_root_count") != len(reopened_roots)
        or len(reopened_keys) != len(set(reopened_keys))
    ):
        errors.append("reopened_roots_invalid")

    runtime_dispositions = {
        "not_requested",
        "completed",
        "unsupported",
        "proof_not_established",
        "execution_failed",
        "cancelled_or_timeout",
    }
    side_status: list[bool] = []
    for mode, mapping_field in (("cold", "cold_mapping"),
                                ("incremental", "child_mapping")):
        candidates = transition.get(f"{mode}_mapping_candidates")
        eligible = transition.get(f"{mode}_eligible_mappings")
        reasons = transition.get(f"{mode}_execution_incomplete_reasons")
        runtime = transition.get(f"{mode}_runtime_evidence")
        oracle = transition.get(f"{mode}_oracle_evidence")
        spectrum = transition.get(f"{mode}_selection_spectrum")
        mapping = transition.get(mapping_field)
        disposition = transition.get(f"{mode}_runtime_disposition")
        if disposition not in runtime_dispositions:
            errors.append(f"{mode}_runtime_disposition_invalid")
        if not _selection_spectrum_valid(spectrum):
            errors.append(f"{mode}_selection_spectrum_invalid")
        if isinstance(candidates, list) and isinstance(eligible, list) and any(
            value not in candidates for value in eligible
        ):
            errors.append(f"{mode}_eligible_mapping_foreign")
        if isinstance(runtime, list) and isinstance(oracle, list) and any(
            value not in runtime for value in oracle
        ):
            errors.append(f"{mode}_oracle_evidence_foreign")
        if isinstance(runtime, list):
            for evidence in runtime:
                decoded = decode_artifact_root_hex(evidence)
                if (
                    decoded is not None
                    and (
                        decoded["schema"] != "evaluation.evidence"
                        or decoded["schema_version"] != "1.0"
                    )
                ):
                    errors.append(f"{mode}_runtime_evidence_schema_invalid")
        if mapping is not None and not _artifact_root(mapping):
            errors.append(f"{mode}_mapping_invalid")
        if isinstance(reasons, list) and reasons:
            if (
                mapping is not None
                or spectrum is not None
                or disposition != "not_requested"
                or runtime
                or oracle
            ):
                errors.append(f"{mode}_incomplete_plan_has_post_plan_evidence")
        elif mapping is not None and (
            not isinstance(candidates, list)
            or mapping not in candidates
            or not isinstance(eligible, list)
            or mapping not in eligible
            or not _selection_spectrum_admits(spectrum, mapping, endpoint)
        ):
            errors.append(f"{mode}_selected_mapping_unproven")
        side_status.append(
            isinstance(reasons, list)
            and not reasons
            and isinstance(candidates, list)
            and mapping in candidates
            and isinstance(eligible, list)
            and mapping in eligible
            and disposition == "completed"
            and isinstance(runtime, list)
            and bool(runtime)
            and isinstance(oracle, list)
            and bool(oracle)
            and _selection_spectrum_admits(spectrum, mapping, endpoint)
        )
    derived_verified = len(side_status) == 2 and all(side_status)
    if transition.get("verified") is not derived_verified:
        errors.append("verified_flag_mismatch")
    if derived_verified:
        if (
            transition.get("disposition") != "verified"
            or transition.get("incomplete_reason") is not None
        ):
            errors.append("verified_disposition_invalid")
    elif (
        transition.get("disposition") != "incomplete"
        or not isinstance(transition.get("incomplete_reason"), str)
        or not transition["incomplete_reason"]
    ):
        errors.append("incomplete_disposition_invalid")
    return derived_verified, errors


def _root_reference(
    value: Any, schema: str | None = None, version: str | None = None
) -> dict[str, str] | None:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "schema_version",
        "artifact",
    }:
        return None
    if not isinstance(value.get("schema"), str) or not isinstance(
        value.get("schema_version"), str
    ):
        return None
    artifact = value.get("artifact")
    if not _digest(artifact):
        return None
    if schema is not None and value["schema"] != schema:
        return None
    if version is not None and value["schema_version"] != version:
        return None
    return dict(value)


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


FUNNEL_COMPARISON_COUNTS = (
    "mapped_candidates",
    "predicted_feasible_candidates",
    "verified_candidates",
    "measured_candidates",
    "out_of_distribution_candidates",
    "prediction_error_candidates",
)


def _funnel_exact_comparison(decision: dict[str, Any]) -> dict[str, Any] | None:
    """The pair decision's exact funnel comparison, validated field by field.

    Returns None when the object or any count is missing or malformed; the
    optional members stay None when the decision has no shared time basis.
    """
    comparison = decision.get("funnel_exact_comparison")
    if not isinstance(comparison, dict):
        return None
    result: dict[str, Any] = {}
    for field in FUNNEL_COMPARISON_COUNTS:
        value = _integer(comparison.get(field))
        if value is None or value < 0:
            return None
        result[field] = value
    ranking = comparison.get("best_ranking_match")
    if ranking is not None and not isinstance(ranking, bool):
        return None
    result["best_ranking_match"] = ranking
    for field in ("analytic_clock_period_picoseconds", "maximum_prediction_error_ppm"):
        value = comparison.get(field)
        if value is not None and (_integer(value) is None or _integer(value) < 0):
            return None
        result[field] = _integer(value) if value is not None else None
    return result


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
        or report.get("version") != "2.0"
        or report.get("manifest_schema") != "loom.application_portfolio"
        or report.get("manifest_version") != "4.0"
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
            or set(build)
            != {
                "entry",
                "language",
                "sources",
                "compiler_options",
                "link_options",
                "operator_protocol_symbols",
                "product_execution",
            }
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
        else:
            product = build.get("product_execution")
            if product is not None and (
                not isinstance(product, dict)
                or set(product)
                != {"entry_symbol", "measured_output_bytes_per_sample"}
                or not isinstance(product.get("entry_symbol"), str)
                or not product["entry_symbol"]
                or _integer(product.get("measured_output_bytes_per_sample"))
                is None
                or product["measured_output_bytes_per_sample"] <= 0
            ):
                errors.append(f"{context}:product_execution_invalid")
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
            or not _digest(oracle.get("sha256"))
            or oracle.get("encoding") not in {"utf8", "hex_sample_lines"}
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
        if isinstance(build, dict):
            product = build.get("product_execution")
            if product is None and isinstance(oracle, dict):
                if oracle.get("encoding") != "utf8":
                    errors.append(f"{context}:host_oracle_encoding_invalid")
            elif isinstance(product, dict):
                if (
                    not isinstance(build.get("operator_protocol_symbols"), list)
                    or not build["operator_protocol_symbols"]
                    or not isinstance(oracle, dict)
                    or oracle.get("kind") != "exact"
                    or oracle.get("encoding") != "hex_sample_lines"
                ):
                    errors.append(f"{context}:product_oracle_contract_invalid")
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
    if schema == PAIR_DISPOSITION_SCHEMA:
        if version != PAIR_DISPOSITION_VERSION:
            reasons.append("pair_disposition_version_invalid")
        if evidence.get("domain") != "application_pair_decision":
            reasons.append("pair_disposition_domain_invalid")
        return reasons, False
    if version != PAIR_EVIDENCE_VERSION:
        reasons.append("pair_evidence_version_invalid")
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
    repair_records = evidence.get("hardware_mutation_repair_records")
    if has_mapping_evidence and (
        not isinstance(repair_records, list)
        or any(not _artifact_root(record) for record in repair_records)
        or repair_records != sorted(set(repair_records))
    ):
        typed_reasons.append("hardware_mutation_repair_inventory_invalid")
        repair_records = []
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
    if join_status not in {MANIFEST_JOIN_COMPLETE, MANIFEST_JOIN_PRE_ADMISSION}:
        typed_reasons.append("manifest_join_unverified")
    if join_status == MANIFEST_JOIN_PRE_ADMISSION:
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
    repair_attempts = _integer(
        decision.get("resource_time_mapping_repair_attempt_count")
    )
    verified_repairs = _integer(
        decision.get("resource_time_mapping_repair_verified_count")
    )
    repair_incomplete = decision.get(
        "resource_time_mapping_repair_incomplete_reason"
    )
    if (
        repair_attempts is None
        or repair_attempts < 0
        or verified_repairs is None
        or verified_repairs < 0
        or verified_repairs > repair_attempts
        or (
            verified_repairs == repair_attempts
            and repair_incomplete is not None
        )
        or (
            verified_repairs < repair_attempts
            and (not isinstance(repair_incomplete, str) or not repair_incomplete)
        )
    ):
        typed_reasons.append("resource_time_mapping_repair_summary_invalid")
    repair_transitions = evidence.get("application_incremental_mapping_transitions")
    if has_mapping_evidence:
        if not isinstance(repair_transitions, list):
            typed_reasons.append("resource_time_mapping_repair_rows_missing")
        else:
            row_results = [
                validate_resource_time_mapping_repair_transition(transition)
                for transition in repair_transitions
            ]
            if any(errors for _, errors in row_results):
                typed_reasons.append("resource_time_mapping_repair_row_invalid")
            if (
                repair_attempts != len(repair_transitions)
                or verified_repairs
                != sum(1 for verified, _ in row_results if verified)
            ):
                typed_reasons.append("resource_time_mapping_repair_rows_mismatch")
    elif repair_attempts != 0 or verified_repairs != 0:
        typed_reasons.append("resource_time_mapping_repair_rows_unowned")

    if (
        selection.get("execution_binding") != "canonical_simulation_and_oracle"
        or selection.get("execution_binding_established") is not True
    ):
        closure_reasons.append("execution_binding_incomplete")
    if decision.get("host_only_baseline_complete") is not True:
        closure_reasons.append("host_baseline_incomplete")
    if decision.get("final_application_qor_complete") is not True:
        closure_reasons.append("application_qor_incomplete")
    if join_status != MANIFEST_JOIN_COMPLETE:
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
        observations = candidate.get("mapping_observations")
        if candidate.get("entered_mapping") is True and (
            not isinstance(observations, list) or not observations
        ):
            typed_reasons.append("mapping_observation_inventory_missing")
        if isinstance(observations, list):
            for observation in observations:
                if not isinstance(observation, dict):
                    typed_reasons.append("mapping_observation_invalid")
                    continue
                runtime_mapping = observation.get("runtime_mapping")
                runtime_disposition = observation.get("runtime_disposition")
                system_mappings = observation.get("system_mappings")
                if not (
                    runtime_disposition == "not_requested"
                    and runtime_mapping is None
                ) and not (
                    runtime_disposition != "not_requested"
                    and _artifact_root(runtime_mapping)
                    and isinstance(system_mappings, list)
                    and runtime_mapping in system_mappings
                ):
                    typed_reasons.append("mapping_runtime_owner_invalid")
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
        # Equivalent schedule hints share one Mapping plan and are each
        # verified against the same Mapping; the finalist digest named by the
        # decision is what identifies the one selected observation.
        selected_hint = decision.get("selected_schedule_hint_digest")
        if not _digest(selected_hint):
            typed_reasons.append("selected_schedule_hint_missing")
        selected_observations = (
            [
                observation
                for observation in observations
                if isinstance(observation, dict)
                and observation.get("plan_ordinal") == selected_plan
                and observation.get("schedule_hint_digest") == selected_hint
                and isinstance(observation.get("system_mappings"), list)
                and observation.get("runtime_mapping") == selected_mapping
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
                or selected_mapping not in selected_observation["system_mappings"]
                or not _artifact_root(selected_observation.get("runtime_mapping"))
                or not _artifact_root_list(selected_observation.get("runtime_evidence"))
                or not _artifact_root_list(selected_observation.get("oracle_evidence"))
            ):
                closure_reasons.append("selected_mapping_evidence_incomplete")
            if decision.get("selected_system") != selected_observation.get("system"):
                typed_reasons.append("selected_system_mismatch")
            if not _artifact_root(selected_mapping):
                typed_reasons.append("selected_mapping_mismatch")
            selected_repair = selected_observation.get(
                "hardware_mutation_repair_record"
            )
            if selected_repair is not None:
                if (
                    disposition != "hardware_dse_alternative"
                    or not _artifact_root(selected_repair)
                    or selected_repair not in repair_records
                ):
                    typed_reasons.append("selected_hardware_repair_mismatch")
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
    funnel_comparison = _funnel_exact_comparison(decision)
    if funnel_comparison is None:
        typed_reasons.append("funnel_exact_comparison_invalid")
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
        "funnel_exact_comparison": funnel_comparison,
    }


PRODUCT_ENTRY_ABI = "cached_inputs_profile_output_v1"
PRODUCT_PROFILE_FIELDS = {
    "entry_abi",
    "entry_symbol",
    "warmup_samples",
    "measured_samples",
    "measured_output_bytes_per_sample",
    "expected_output_sha256",
    "output_interface_ordinal",
}


def validate_portfolio_product_execution(
    pair_evidence: dict[str, Any],
    expected: dict[str, Any],
    runtime_manifest_bindings: list[dict[str, Any]],
    execution_workspaces: list[dict[str, Any]],
) -> dict[str, Any]:
    """Join one product row to its exact runtime manifest and oracle Evidence."""
    reasons: list[str] = []
    decision = _pair_decision(pair_evidence)
    build = expected.get("build")
    product = build.get("product_execution") if isinstance(build, dict) else None
    profile = expected.get("profile")
    cached_inputs = expected.get("cached_inputs")
    if (
        decision is None
        or not isinstance(product, dict)
        or not isinstance(profile, dict)
        or not isinstance(cached_inputs, list)
    ):
        return {
            "complete": False,
            "incomplete_reasons": ["product_manifest_contract_missing"],
            "runtime_manifest": None,
            "workspace_count": 0,
            "oracle_evidence": [],
        }

    candidate_bindings = [
        binding
        for binding in runtime_manifest_bindings
        if isinstance(binding, dict)
        and binding.get("pair_identity") == decision.get("pair_identity")
        and binding.get("invocation_manifest_run_key")
        == decision.get("invocation_manifest_run_key")
    ]
    if not candidate_bindings:
        reasons.append("runtime_manifest_binding_missing")

    runtime_roots: list[dict[str, str]] = []
    for binding in candidate_bindings:
        if (
            binding.get("schema")
            != RUNTIME_BINDING_SCHEMA
            or binding.get("version") != RUNTIME_BINDING_VERSION
            or binding.get("domain") != "application_runtime_manifest"
        ):
            reasons.append("runtime_manifest_binding_schema_invalid")
            continue
        for binding_field, decision_field in (
            ("source_program", "source_program"),
            ("fabric", "fabric"),
            ("workload", "workload"),
            ("runtime_input", "runtime_input"),
            ("selected_system", "selected_system"),
            ("selected_mapping", "selected_system_mapping"),
        ):
            if binding.get(binding_field) != decision.get(decision_field):
                reasons.append(f"runtime_manifest_{binding_field}_mismatch")
        decoded = decode_artifact_root_hex(binding.get("runtime_manifest"))
        if (
            decoded is None
            or decoded.get("schema") != RUNTIME_MANIFEST_SCHEMA
            or decoded.get("schema_version") != RUNTIME_MANIFEST_VERSION
        ):
            reasons.append("runtime_manifest_root_invalid")
            continue
        runtime_roots.append(decoded)

    unique_runtime_roots = {
        (root["schema"], root["schema_version"], root["artifact"])
        for root in runtime_roots
    }
    if len(unique_runtime_roots) != 1:
        reasons.append("runtime_manifest_binding_not_unique")
        runtime_root = None
    else:
        runtime_root = runtime_roots[0]

    matching_workspaces = (
        [
            workspace
            for workspace in execution_workspaces
            if isinstance(workspace, dict)
            and workspace.get("application_runtime_manifest") == runtime_root
        ]
        if runtime_root is not None
        else []
    )
    if not matching_workspaces:
        reasons.append("product_execution_workspace_missing")

    oracle_evidence: list[dict[str, str]] = []
    for workspace in matching_workspaces:
        if workspace.get("schema") != "loom.execution_matrix_workspace.2.0":
            reasons.append("product_execution_workspace_schema_invalid")
        if candidate_bindings:
            binding = candidate_bindings[0]
            for workspace_field, binding_field in (
                ("deployment", "deployment"),
                ("workload", "activation_workload"),
                ("runtime_input", "activation_runtime_input"),
            ):
                if workspace.get(workspace_field) != decode_artifact_root_hex(
                    binding.get(binding_field)
                ):
                    reasons.append(f"product_execution_{workspace_field}_mismatch")

        actual_profile = workspace.get("product_profile")
        if not isinstance(actual_profile, dict) or set(actual_profile) != (
            PRODUCT_PROFILE_FIELDS
        ):
            reasons.append("product_execution_profile_invalid")
        else:
            expected_profile = {
                "entry_abi": PRODUCT_ENTRY_ABI,
                "entry_symbol": product.get("entry_symbol"),
                "warmup_samples": profile.get("warmup_samples"),
                "measured_samples": profile.get("measured_samples"),
                "measured_output_bytes_per_sample": product.get(
                    "measured_output_bytes_per_sample"
                ),
                "output_interface_ordinal": len(cached_inputs),
            }
            for field, value in expected_profile.items():
                if actual_profile.get(field) != value:
                    reasons.append(f"product_execution_{field}_mismatch")
            if not _digest(actual_profile.get("expected_output_sha256")):
                reasons.append("product_execution_expected_output_invalid")

        if workspace.get("value_results") != [["0"]]:
            reasons.append("product_execution_result_invalid")
        runs = workspace.get("runs")
        system_runs = (
            [
                run
                for run in runs
                if isinstance(run, dict) and run.get("scope") == "system"
            ]
            if isinstance(runs, list)
            else []
        )
        if len(system_runs) != 2 or {
            run.get("engine") for run in system_runs
        } != {"dfg", "cgra"}:
            reasons.append("product_execution_system_matrix_incomplete")
            continue
        for run in system_runs:
            request = _root_reference(
                run.get("product_oracle_request"), "evaluation.request"
            )
            evidence = _root_reference(
                run.get("product_oracle_evidence"), "evaluation.evidence"
            )
            if request is None:
                reasons.append("product_oracle_request_invalid")
            if evidence is None:
                reasons.append("product_oracle_evidence_invalid")
            else:
                oracle_evidence.append(evidence)

    reasons = list(dict.fromkeys(reasons))
    return {
        "complete": not reasons,
        "incomplete_reasons": reasons,
        "runtime_manifest": runtime_root,
        "workspace_count": len(matching_workspaces),
        "oracle_evidence": oracle_evidence,
    }


def evaluate_portfolio(
    inventory: list[dict[str, Any]],
    host_runs: list[dict[str, Any]],
    pair_evidence: list[dict[str, Any]],
    manifest_errors: list[str],
    runtime_manifest_bindings: list[dict[str, Any]] | None = None,
    execution_workspaces: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    runtime_manifest_bindings = runtime_manifest_bindings or []
    execution_workspaces = execution_workspaces or []
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
    pair_records_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for evidence in pair_evidence:
        pair_records_by_key.setdefault(portfolio_pair_key(evidence), []).append(
            evidence
        )

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
        build = row.get("build")
        product_required = isinstance(build, dict) and isinstance(
            build.get("product_execution"), dict
        )
        product_executions = [
            validate_portfolio_product_execution(
                evidence,
                row,
                runtime_manifest_bindings,
                execution_workspaces,
            )
            for evidence in pair_records_by_key.get(key, [])
            if (
                (validated := validate_portfolio_pair(evidence, row)) is not None
                and validated["canonical_qor_complete"]
            )
        ]
        product_execution_complete = not product_required or (
            bool(product_executions)
            and all(execution["complete"] for execution in product_executions)
        )
        application = key[0]
        funnel_comparisons = [
            pair["funnel_exact_comparison"]
            for pair in pairs
            if isinstance(pair.get("funnel_exact_comparison"), dict)
        ]
        ranking_matches = [
            comparison["best_ranking_match"]
            for comparison in funnel_comparisons
            if comparison["best_ranking_match"] is not None
        ]
        prediction_errors = [
            comparison["maximum_prediction_error_ppm"]
            for comparison in funnel_comparisons
            if comparison["maximum_prediction_error_ppm"] is not None
        ]
        funnel_exact_comparison = {
            **{
                field: sum(comparison[field] for comparison in funnel_comparisons)
                for field in FUNNEL_COMPARISON_COUNTS
            },
            "best_ranking_match_holds": bool(ranking_matches)
            and all(ranking_matches),
            "maximum_prediction_error_ppm": (
                max(prediction_errors) if prediction_errors else None
            ),
        }
        funnel_exact_comparison["exact_sample_complete"] = (
            funnel_exact_comparison["measured_candidates"] >= 1
            and funnel_exact_comparison["prediction_error_candidates"] >= 1
        )
        member_evaluations.append(
            {
                "application_identity": application,
                "input_name": key[1],
                "funnel_exact_comparison": funnel_exact_comparison,
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
                "product_execution_required": product_required,
                "product_execution_complete": product_execution_complete,
                "product_executions": product_executions,
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
                and canonical_application_qor_complete
                and product_execution_complete,
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
        qor_required = required
        missing_qor = [
            [row["application_identity"], row["input_name"]]
            for row in qor_required
            if not row["canonical_application_qor_complete"]
        ]
        missing_product_execution = [
            [row["application_identity"], row["input_name"]]
            for row in required
            if row["product_execution_required"]
            and not row["product_execution_complete"]
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
                "missing_product_execution_pairs": missing_product_execution,
                "host_conformance_gate_holds": bool(required) and not missing_host,
                "accelerator_disposition_gate_holds": bool(required)
                and not missing_typed,
                "canonical_application_qor_gate_holds": bool(qor_required)
                and not missing_qor,
                "product_execution_gate_holds": not missing_product_execution,
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
    tinyml_profile_holds = bool(tinyml_rows) and all(
        row["host_conformance_complete"]
        and row["canonical_application_qor_complete"]
        and row["product_execution_complete"]
        for row in tinyml_rows
    )
    all_selection_gates_hold = all(
        row["portfolio_requirement_gate_holds"] for row in selection_evaluations
    )
    canonical_qor_rows = [
        row
        for row in member_evaluations
        if row["application_identity"] in CANONICAL_QOR_APPLICATIONS
        and row["canonical_application_qor_complete"]
    ]
    funnel_exact_comparison_sample_holds = bool(canonical_qor_rows) and all(
        row["funnel_exact_comparison"]["exact_sample_complete"]
        for row in canonical_qor_rows
    )
    acceptance = {
        "canonical_qor_witnesses": canonical_witnesses,
        "canonical_qor_witnesses_hold": all(canonical_witnesses.values()),
        "tinyml_product_profiles_hold": tinyml_profile_holds,
        "all_execution_selection_gates_hold": all_selection_gates_hold,
        "funnel_exact_comparison_sample_holds": funnel_exact_comparison_sample_holds,
        "portfolio_acceptance_holds": not manifest_errors
        and all_selection_gates_hold
        and all(canonical_witnesses.values())
        and tinyml_profile_holds
        and funnel_exact_comparison_sample_holds,
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
