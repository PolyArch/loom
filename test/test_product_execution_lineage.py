#!/usr/bin/env python3
"""Anchor tests for selected product schedule lineage."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "test" / "system" / "verify_product_execution.py"
SPEC = importlib.util.spec_from_file_location("verify_product_execution", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load product execution verifier")
verify_product_execution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_product_execution)


SCHEDULE_PROGRAM = "a" * 64
SCHEDULE_DATAFLOW = "c" * 64
OTHER_DATAFLOW = "d" * 64
SCHEDULE_CANDIDATE = "e" * 64
SCHEDULE_HINT = "1" * 64
PLANNING_RECORD = 5
SYSTEM = "system"
MAPPING = "mapping"


def _events(
    *,
    dataflow: str = SCHEDULE_DATAFLOW,
    planning_record: int = PLANNING_RECORD,
) -> list[dict[str, Any]]:
    first_parent = "schedule-parent"
    first_child = "schedule-child"
    second_child = "parallel-child"
    decisions: list[dict[str, Any]] = [
        {
            "kind": "polyhedral_schedule",
            "factor": 2,
            "parent": first_parent,
            "child": first_child,
        },
        {
            "kind": "parallelize",
            "factor": 0,
            "parent": first_child,
            "child": second_child,
        },
    ]
    proposals: list[dict[str, Any]] = [
        {
            "stage": "dataflow_lowering",
            "event": "candidate",
            "payload": {
                "operation": "structured_schedule_candidate",
                "decision_kind": "polyhedral_schedule",
                "factor": 2,
                "parent": first_parent,
                "child": first_child,
            },
        },
        {
            "stage": "dataflow_lowering",
            "event": "candidate",
            "payload": {
                "operation": "structured_schedule_candidate",
                "decision_kind": "parallelize",
                "factor": 0,
                "parent": first_child,
                "child": second_child,
            },
        },
    ]
    proposals.append(
        {
            "stage": "dataflow_lowering",
            "event": "candidate",
            "payload": {
                "operation": "selected_candidate_lineage",
                "planning_record_ordinal": planning_record,
                "structured_program": SCHEDULE_PROGRAM,
                "canonical_dataflow": dataflow,
                "schedule_decisions": decisions,
            },
        }
    )
    return proposals


def _candidate() -> dict[str, Any]:
    return {
        "structured_program": SCHEDULE_PROGRAM,
        "canonical_dataflow": SCHEDULE_DATAFLOW,
        "planning_record_ordinal": PLANNING_RECORD,
        "candidate_identity": SCHEDULE_CANDIDATE,
        "plan_ordinal": 7,
        "entered_mapping": True,
        "selected": True,
        "mapping_observations": [
            {
                "plan_ordinal": 7,
                "schedule_hint_digest": SCHEDULE_HINT,
                "system": SYSTEM,
                "system_mappings": [MAPPING],
                "mapping_disposition": "verified",
                "runtime_disposition": "completed",
                "runtime_evidence": ["runtime"],
                "oracle_evidence": ["oracle"],
            }
        ],
    }


def _decision(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidates": [candidate],
        "selected_candidate_identity": candidate["candidate_identity"],
        "selected_schedule_hint_digest": SCHEDULE_HINT,
        "selected_system": SYSTEM,
        "selected_system_mapping": MAPPING,
    }


class ProductScheduleLineageTest(unittest.TestCase):
    def test_selected_lineage_joins_the_exact_planning_record(self) -> None:
        candidate = _candidate()
        result = verify_product_execution.validate_schedule_lineage(
            _events(),
            _decision(candidate),
            [("polyhedral_schedule", 2), ("parallelize", 0)],
        )
        self.assertTrue(result["selected"])
        self.assertEqual(result["verified_observation_count"], 1)

    def test_schedule_path_rejects_a_foreign_planning_record(self) -> None:
        candidate = _candidate()
        with self.assertRaisesRegex(ValueError, "no selected compilation"):
            verify_product_execution.validate_schedule_lineage(
                _events(planning_record=PLANNING_RECORD + 1),
                _decision(candidate),
                [("polyhedral_schedule", 2), ("parallelize", 0)],
            )

    def test_schedule_path_rejects_a_foreign_dataflow(self) -> None:
        candidate = _candidate()
        with self.assertRaisesRegex(ValueError, "no selected compilation"):
            verify_product_execution.validate_schedule_lineage(
                _events(dataflow=OTHER_DATAFLOW),
                _decision(candidate),
                [("polyhedral_schedule", 2), ("parallelize", 0)],
            )

    def test_selected_path_requires_completed_runtime_and_oracle_evidence(
        self,
    ) -> None:
        candidate = _candidate()
        observation = candidate["mapping_observations"][0]
        observation["runtime_evidence"] = []
        with self.assertRaisesRegex(ValueError, "completed runtime and oracle"):
            verify_product_execution.validate_schedule_lineage(
                _events(),
                _decision(candidate),
                [("polyhedral_schedule", 2), ("parallelize", 0)],
            )


if __name__ == "__main__":
    unittest.main()
