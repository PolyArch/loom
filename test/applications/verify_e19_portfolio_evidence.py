#!/usr/bin/env python3
"""Protect the exact E19 promotion and weighted-OOD diagnostic joins."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from loom_evidence_portfolio import (  # noqa: E402
    EVALUATION_EVIDENCE_SCHEMA,
    PAIR_DECISION_SCHEMA,
    PAIR_DECISION_VERSION,
    PAIR_EVIDENCE_SCHEMA,
    PAIR_EVIDENCE_VERSION,
    validate_hardware_promotion_witnesses,
    validate_weighted_ood_witnesses,
)


U64_MAX = (1 << 64) - 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def artifact_root(schema: str, version: str, ordinal: int) -> str:
    encoded_schema = schema.encode("ascii")
    major, minor = (int(component) for component in version.split("."))
    return (
        len(encoded_schema).to_bytes(4, "big")
        + encoded_schema
        + major.to_bytes(4, "big")
        + minor.to_bytes(4, "big")
        + ordinal.to_bytes(32, "big")
    ).hex()


def integer(value: int) -> dict[str, Any]:
    return {"kind": "integer", "negative": False, "magnitude": value}


def decimal(coefficient: int, exponent: int) -> dict[str, Any]:
    return {
        "kind": "decimal",
        "coefficient": coefficient,
        "base10_exponent": exponent,
    }


def objective(dimension: str, value: int) -> dict[str, Any]:
    return {
        "dimension": dimension,
        "value": value,
        "evidence": "calibrated",
        "confidence_permille": 500,
        "out_of_distribution": False,
    }


def promotion_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_schema, evidence_version = EVALUATION_EVIDENCE_SCHEMA
    fpa_evidence = artifact_root(evidence_schema, evidence_version, 1)
    parent = artifact_root("loom.fabric", "7.1", 2)
    child = artifact_root("loom.fabric", "7.1", 3)
    mapping = artifact_root("loom.mapping", "1.0", 4)
    raw_fpa = [
        decimal(100, 0),
        decimal(2, -6),
        decimal(3, -3),
        decimal(4, -4),
    ]
    fpa_codes = [U64_MAX - 100, 2_000_000, 3_000, 400]
    raw_runtime = [integer(7), integer(11), integer(2)]
    runtime_codes = [7, 11, 2, *fpa_codes]
    selected_mapping_observation = {
        "plan_ordinal": 1,
        "schedule_hint_digest": "11" * 32,
        "system": child,
        "mapping_disposition": "verified",
        "system_mappings": [mapping],
        "dfg_cycles": 7,
        "cgra_cycles": 11,
        "resource_core_cost": 2,
    }
    promotion = {
        "plan_ordinal": 1,
        "system": parent,
        "objective_codes": fpa_codes,
        "provenance": {
            "raw_measures": raw_fpa,
            "runtime_completion": "not_established",
            "calibrated_model_support": "in_domain",
        },
        "incomplete_reason": None,
        "evidence": fpa_evidence,
        "promoted_to_exact_mapping": True,
    }
    decision = {
        "schema": PAIR_DECISION_SCHEMA,
        "version": PAIR_DECISION_VERSION,
        "hardware_promotion_objective_dimension_labels": [
            "limiting_clock_frequency",
            "total_area",
            "dynamic_power",
            "leakage_power",
        ],
        "hardware_promotion_observations": [promotion],
        "selected_system": child,
        "selected_system_mapping": mapping,
        "selected_schedule_hint_digest": "11" * 32,
        "candidates": [
            {
                "selected": True,
                "mapping_observations": [selected_mapping_observation],
            }
        ],
        "quality_observations": [
            {
                "system_mapping": mapping,
                "objective_codes": runtime_codes,
                "incomplete_reason": None,
                "evidence": fpa_evidence,
                "provenance": {
                    "raw_measures": [*raw_runtime, *raw_fpa],
                    "runtime_completion": "completed",
                    "calibrated_model_support": "in_domain",
                },
            }
        ],
        "selected_objective": [
            objective("area", 2_000_000),
            objective("power", 3_400),
            objective("energy", 374_000_000),
        ],
    }
    evidence = {
        "schema": PAIR_EVIDENCE_SCHEMA,
        "version": PAIR_EVIDENCE_VERSION,
        "stopping_policy": "bounded_quality",
        "hardware_parent_promotions": 1,
        "hardware_promotion_observations": [deepcopy(promotion)],
        "selected_plan_ordinal": 1,
        "joint_design_attempts": [
            {
                "plan_ordinal": 1,
                "system": child,
                "disposition": "verified",
                "system_mappings": [mapping],
                "hardware_promotion_parent_system": parent,
            }
        ],
    }
    return evidence, decision


def verify_promotion_join() -> None:
    evidence, decision = promotion_fixture()
    witnesses, reasons = validate_hardware_promotion_witnesses(evidence, decision)
    require(len(witnesses) == 1 and not reasons, "valid promotion was rejected")

    foreign_parent = deepcopy(evidence)
    foreign_parent["joint_design_attempts"][0][
        "hardware_promotion_parent_system"
    ] = foreign_parent["joint_design_attempts"][0]["system"]
    witnesses, reasons = validate_hardware_promotion_witnesses(
        foreign_parent, decision
    )
    require(
        not witnesses and "hardware_promotion_not_observed" in reasons,
        "foreign promotion parent was accepted",
    )

    wrong_code = deepcopy(decision)
    wrong_code["hardware_promotion_observations"][0]["objective_codes"][1] += 1
    witnesses, reasons = validate_hardware_promotion_witnesses(evidence, wrong_code)
    require(
        not witnesses and "hardware_promotion_observation_invalid" in reasons,
        "unreproduced promotion objective was accepted",
    )

    wrong_area = deepcopy(decision)
    wrong_area["selected_objective"][0]["value"] += 1
    witnesses, reasons = validate_hardware_promotion_witnesses(evidence, wrong_area)
    require(
        not witnesses and "hardware_promotion_selected_area_invalid" in reasons,
        "unreproduced selected physical objective was accepted",
    )

    wrong_count = deepcopy(evidence)
    wrong_count["hardware_parent_promotions"] = 2
    _, reasons = validate_hardware_promotion_witnesses(wrong_count, decision)
    require(
        "hardware_promotion_count_mismatch" in reasons,
        "promotion count mismatch was accepted",
    )


def ood_fixture() -> dict[str, Any]:
    evidence_schema, evidence_version = EVALUATION_EVIDENCE_SCHEMA
    system = artifact_root("loom.fabric", "7.1", 5)
    mapping = artifact_root("loom.mapping", "1.0", 6)
    runtime_evidence = artifact_root(evidence_schema, evidence_version, 7)
    oracle_evidence = artifact_root(evidence_schema, evidence_version, 8)
    fpa_evidence = artifact_root(evidence_schema, evidence_version, 9)
    clock = 10
    cgra_cycles = 11
    measured = clock * cgra_cycles
    predicted = 121
    error = abs(predicted - measured) * 1_000_000 // measured
    decision = {
        "schema": PAIR_DECISION_SCHEMA,
        "version": PAIR_DECISION_VERSION,
        "portfolio_input": {
            "application_identity": "gapbs-pagerank",
            "input_name": "validation-scale-eda",
        },
        "funnel_exact_comparison": {
            "mapped_candidates": 1,
            "predicted_feasible_candidates": 1,
            "verified_candidates": 1,
            "measured_candidates": 1,
            "out_of_distribution_candidates": 1,
            "prediction_error_candidates": 1,
            "best_ranking_match": True,
            "analytic_clock_period_picoseconds": clock,
            "maximum_prediction_error_ppm": error,
        },
        "candidates": [
            {
                "candidate_identity": "22" * 32,
                "mapping_observations": [
                    {
                        "physical_model_support": "out_of_domain",
                        "mapping_disposition": "verified",
                        "runtime_disposition": "completed",
                        "system": system,
                        "schedule_hint_digest": "33" * 32,
                        "system_mappings": [mapping],
                        "runtime_evidence": [runtime_evidence],
                        "oracle_evidence": [oracle_evidence],
                        "dfg_cycles": 7,
                        "cgra_cycles": cgra_cycles,
                        "resource_core_cost": 2,
                        "predicted_makespan_picoseconds": predicted,
                        "predicted_support": "out_of_domain",
                        "measured_makespan_picoseconds": measured,
                        "prediction_error_ppm": error,
                    }
                ],
            }
        ],
        "quality_observations": [
            {
                "system_mapping": mapping,
                "incomplete_reason": "unsupported",
                "evidence": fpa_evidence,
                "provenance": {
                    "raw_measures": [integer(7), integer(cgra_cycles), integer(2)],
                    "runtime_completion": "completed",
                    "calibrated_model_support": "out_of_domain",
                },
            }
        ],
    }
    return {
        "schema": PAIR_EVIDENCE_SCHEMA,
        "version": PAIR_EVIDENCE_VERSION,
        "pair_decision": decision,
    }


def verify_weighted_ood_join() -> None:
    evidence = ood_fixture()
    witnesses, reasons = validate_weighted_ood_witnesses(evidence)
    require(len(witnesses) == 1 and not reasons, "valid weighted OOD was rejected")

    unowned_runtime = deepcopy(evidence)
    unowned_runtime["pair_decision"]["quality_observations"][0]["provenance"][
        "runtime_completion"
    ] = "not_established"
    witnesses, reasons = validate_weighted_ood_witnesses(unowned_runtime)
    require(
        not witnesses and "weighted_ood_observation_invalid" in reasons,
        "ownerless runtime completion was accepted",
    )

    stale_schema = deepcopy(evidence)
    stale_schema["version"] = "1.0"
    witnesses, reasons = validate_weighted_ood_witnesses(stale_schema)
    require(
        not witnesses and reasons == ["weighted_ood_schema_invalid"],
        "stale pair ABI was accepted",
    )


if __name__ == "__main__":
    verify_promotion_join()
    verify_weighted_ood_join()
