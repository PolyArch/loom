#!/usr/bin/env python3
"""Anchor tests for paired Spatial/System simulation conformance policy."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))

import simulation_conformance  # noqa: E402


class PairedSimulationBudgetTest(unittest.TestCase):
    def test_warmed_median_and_floor_define_the_system_budget(self) -> None:
        tiny = simulation_conformance.paired_system_budget(
            [0.002, 0.003, 0.004],
            spatial_absolute_budget_seconds=(
                simulation_conformance.DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS
            ),
        )
        ordinary = simulation_conformance.paired_system_budget(
            [0.8, 1.0, 1.2],
            spatial_absolute_budget_seconds=(
                simulation_conformance.DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS
            ),
        )
        absolute_budget = (
            simulation_conformance.DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS
        )
        capped = simulation_conformance.paired_system_budget(
            [absolute_budget + 1.0] * 3,
            spatial_absolute_budget_seconds=absolute_budget,
        )

        self.assertEqual(tiny.spatial_reference_seconds, 0.1)
        self.assertAlmostEqual(tiny.system_budget_seconds, 0.3)
        self.assertEqual(ordinary.spatial_reference_seconds, 1.0)
        self.assertEqual(ordinary.system_budget_seconds, 3.0)
        self.assertEqual(
            capped.system_budget_seconds,
            simulation_conformance.SYSTEM_BUDGET_MULTIPLIER * absolute_budget,
        )

    def test_paired_result_keeps_budget_rate_and_hard_ratio_distinct(self) -> None:
        budget = simulation_conformance.paired_system_budget(
            [1.0, 1.0, 1.0],
            spatial_absolute_budget_seconds=(
                simulation_conformance.DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS
            ),
        )
        within = simulation_conformance.evaluate_paired_execution(
            budget,
            simulation_conformance.ActiveExecutionTiming(
                active_wall_seconds=2.5,
                reference_cycles=500_000,
                event_count=17,
                activation_count=3,
                peak_resident_bytes=4096,
            ),
        )
        slow = simulation_conformance.evaluate_paired_execution(
            budget,
            simulation_conformance.ActiveExecutionTiming(
                active_wall_seconds=10.0,
                reference_cycles=500_000,
            ),
        )

        self.assertTrue(within.within_system_budget)
        self.assertFalse(within.hard_ratio_failure)
        self.assertEqual(within.reference_cycles_per_second, 200_000.0)
        self.assertTrue(within.meets_reference_rate_target)
        self.assertEqual(within.event_count, 17)
        self.assertEqual(within.activation_count, 3)
        self.assertEqual(within.peak_resident_bytes, 4096)

        self.assertFalse(slow.within_system_budget)
        self.assertTrue(slow.hard_ratio_failure)
        self.assertFalse(slow.meets_reference_rate_target)

    def test_invalid_measurements_fail_closed(self) -> None:
        for samples in ([], [0.0], [-1.0], [math.inf], [math.nan]):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    simulation_conformance.paired_system_budget(
                        samples,
                        simulation_conformance.DFG_SPATIAL_ABSOLUTE_BUDGET_SECONDS,
                    )

        with self.assertRaises(ValueError):
            simulation_conformance.ActiveExecutionTiming(0.0, 1)
        with self.assertRaises(ValueError):
            simulation_conformance.ActiveExecutionTiming(1.0, -1)

    def test_outer_worker_limit_reserves_cpus_and_obeys_memory_limit(self) -> None:
        self.assertEqual(
            simulation_conformance.outer_worker_limit(
                cpu_count=32, memory_derived_limit=128
            ),
            28,
        )
        self.assertEqual(
            simulation_conformance.outer_worker_limit(
                cpu_count=256, memory_derived_limit=80
            ),
            80,
        )
        self.assertEqual(
            simulation_conformance.outer_worker_limit(
                cpu_count=256, memory_derived_limit=256
            ),
            120,
        )
        self.assertEqual(
            simulation_conformance.outer_worker_limit(
                cpu_count=4, memory_derived_limit=128
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
