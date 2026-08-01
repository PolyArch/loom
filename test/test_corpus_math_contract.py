#!/usr/bin/env python3
"""Anchors exact corpus compilation for source-visible math semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))

import corpus_inventory  # noqa: E402


_MATH_SOURCE_BY_CASE = {
    "arm-entropy-f16": "StatisticsFunctions/arm_entropy_f16.c",
    "arm-jensenshannon-distance-f16": (
        "DistanceFunctions/arm_jensenshannon_distance_f16.c"
    ),
    "arm-jensenshannon-distance-f32": (
        "DistanceFunctions/arm_jensenshannon_distance_f32.c"
    ),
    "arm-kullback-leibler-f16": (
        "StatisticsFunctions/arm_kullback_leibler_f16.c"
    ),
    "arm-minkowski-distance-f16": "DistanceFunctions/arm_minkowski_distance_f16.c",
    "arm-minkowski-distance-f32": "DistanceFunctions/arm_minkowski_distance_f32.c",
    "arm-rotation2quaternion-f32": (
        "QuaternionMathFunctions/arm_rotation2quaternion_f32.c"
    ),
    "arm-sqrt-f16": None,
    "arm-vexp-f16": "FastMathFunctions/arm_vexp_f16.c",
}


class CorpusMathContractTest(unittest.TestCase):
    def test_math_workloads_bind_exact_semantic_compilation(self) -> None:
        workloads = {
            workload.case: workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in _MATH_SOURCE_BY_CASE
        }
        self.assertEqual(set(workloads), set(_MATH_SOURCE_BY_CASE))

        source_prefix = "externals/cmsis-dsp/Source/"
        for case, relative_source in _MATH_SOURCE_BY_CASE.items():
            with self.subTest(case=case):
                workload = workloads[case]
                self.assertEqual(workload.compiler_flags, ("-fno-math-errno",))
                expected_sources = (
                    ()
                    if relative_source is None
                    else (source_prefix + relative_source,)
                )
                self.assertEqual(workload.sources, expected_sources)


if __name__ == "__main__":
    unittest.main()
