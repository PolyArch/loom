#!/usr/bin/env python3
"""Shared report metric record helpers."""

from __future__ import annotations


def metric_record(
    *,
    metric_id: str,
    metric_class: str,
    value: float | int,
    unit: str,
    fidelity_level: str,
    evidence_source_artifact_id: str,
    producer_component: str,
    derivation_kind: str,
    diagnostics: list[str] | None = None,
    input_metric_ids: list[str] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "metric_id": metric_id,
        "metric_class": metric_class,
        "value": value,
        "unit": unit,
        "fidelity_level": fidelity_level,
        "evidence_source_artifact_id": evidence_source_artifact_id,
        "producer_component": producer_component,
        "derivation_kind": derivation_kind,
        "diagnostics": diagnostics or [],
    }
    if input_metric_ids is not None:
        record["input_metric_ids"] = input_metric_ids
    return record
