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
OTHER_PROGRAM = "b" * 64
SCHEDULE_CANDIDATE = "c" * 64
OTHER_CANDIDATE = "d" * 64
SCHEDULE_HINT = "e" * 64
SYSTEM = "system"
MAPPING = "mapping"


def _events(*, with_reduction: bool = False) -> list[dict[str, Any]]:
    first_parent = "schedule-parent"
    first_child = "schedule-child"
    second_child = "parallel-child"
    reduction_child = "reduction-child"
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
    if with_reduction:
        decisions.append(
            {
                "kind": "vectorize",
                "factor": 0,
                "parent": second_child,
                "child": reduction_child,
            }
        )
        proposals.append(
            {
                "stage": "dataflow_lowering",
                "event": "candidate",
                "payload": {
                    "operation": "structured_schedule_candidate",
                    "decision_kind": "vectorize",
                    "factor": 0,
                    "parent": second_child,
                    "child": reduction_child,
                },
            }
        )
    proposals.append(
        {
            "stage": "dataflow_lowering",
            "event": "candidate",
            "payload": {
                "operation": "selected_candidate_lineage",
                "structured_program": SCHEDULE_PROGRAM,
                "schedule_decisions": decisions,
            },
        }
    )
    return proposals


def _candidate(
    program: str, identity: str, *, selected: bool
) -> dict[str, Any]:
    return {
        "structured_program": program,
        "candidate_identity": identity,
        "plan_ordinal": 7,
        "entered_mapping": True,
        "selected": selected,
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


def _decision(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selected = next(candidate for candidate in candidates if candidate["selected"])
    return {
        "candidates": candidates,
        "selected_candidate_identity": selected["candidate_identity"],
        "selected_schedule_hint_digest": SCHEDULE_HINT,
        "selected_system": SYSTEM,
        "selected_system_mapping": MAPPING,
    }


class ProductScheduleLineageTest(unittest.TestCase):
    def test_selected_lineage_can_join_the_existing_reduction_coordinate(
        self,
    ) -> None:
        candidate = _candidate(
            SCHEDULE_PROGRAM, SCHEDULE_CANDIDATE, selected=True
        )
        result = verify_product_execution.validate_schedule_lineage(
            _events(with_reduction=True),
            _decision([candidate]),
            [
                ("polyhedral_schedule", 2),
                ("parallelize", 0),
                ("vectorize", 0),
            ],
        )
        self.assertTrue(result["selected"])
        self.assertEqual(result["verified_observation_count"], 1)

    def test_schedule_path_must_belong_to_the_selected_candidate(self) -> None:
        decision = _decision(
            [
                _candidate(
                    SCHEDULE_PROGRAM, SCHEDULE_CANDIDATE, selected=False
                ),
                _candidate(OTHER_PROGRAM, OTHER_CANDIDATE, selected=True),
            ]
        )
        with self.assertRaisesRegex(ValueError, "uniquely schedule-derived"):
            verify_product_execution.validate_schedule_lineage(
                _events(),
                decision,
                [("polyhedral_schedule", 2), ("parallelize", 0)],
            )

    def test_selected_path_requires_completed_runtime_and_oracle_evidence(
        self,
    ) -> None:
        candidate = _candidate(
            SCHEDULE_PROGRAM, SCHEDULE_CANDIDATE, selected=True
        )
        observation = candidate["mapping_observations"][0]
        observation["runtime_evidence"] = []
        with self.assertRaisesRegex(ValueError, "completed runtime and oracle"):
            verify_product_execution.validate_schedule_lineage(
                _events(),
                _decision([candidate]),
                [("polyhedral_schedule", 2), ("parallelize", 0)],
            )


if __name__ == "__main__":
    unittest.main()
