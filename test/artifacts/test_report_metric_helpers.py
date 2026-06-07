#!/usr/bin/env python3
"""Regression test for shared report metric record helpers."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import report_metric_helpers  # noqa: E402


def main() -> int:
    base = report_metric_helpers.metric_record(
        metric_id="metric::vecsum::cgra_sim_cycles",
        metric_class="hardware_cycles",
        value=589,
        unit="cycles",
        fidelity_level="cgra_mapped",
        evidence_source_artifact_id="vecsum-cgra-sim-report",
        producer_component="workload-report-bundle",
        derivation_kind="cgra_sim_report",
    )
    expected_base = {
        "metric_id": "metric::vecsum::cgra_sim_cycles",
        "metric_class": "hardware_cycles",
        "value": 589,
        "unit": "cycles",
        "fidelity_level": "cgra_mapped",
        "evidence_source_artifact_id": "vecsum-cgra-sim-report",
        "producer_component": "workload-report-bundle",
        "derivation_kind": "cgra_sim_report",
        "diagnostics": [],
    }
    if base != expected_base:
        raise AssertionError(f"unexpected base metric record: {base}")
    derived = report_metric_helpers.metric_record(
        metric_id="metric::vecsum::estimated_runtime_us",
        metric_class="estimated_runtime",
        value=2.356,
        unit="us",
        fidelity_level="analytic",
        evidence_source_artifact_id="rtl-fpa-summary",
        producer_component="workload-report-bundle",
        derivation_kind="cycle_frequency_runtime",
        diagnostics=["derived from cycle and frequency metrics"],
        input_metric_ids=[
            "metric::vecsum::cgra_sim_cycles",
            "metric::shared_reduction_adg::frequency_mhz",
        ],
    )
    if derived.get("diagnostics") != ["derived from cycle and frequency metrics"]:
        raise AssertionError(f"derived metric diagnostics drifted: {derived}")
    if derived.get("input_metric_ids") != [
        "metric::vecsum::cgra_sim_cycles",
        "metric::shared_reduction_adg::frequency_mhz",
    ]:
        raise AssertionError(f"derived metric inputs drifted: {derived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
