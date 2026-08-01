#!/usr/bin/env python3
"""Anchor tests for direct CMSIS-DSP fast-math protocols."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))

import corpus_inventory  # noqa: E402
import corpus_workload_provider  # noqa: E402


DIVIDE_CASES = {"arm-divide-q15", "arm-divide-q31"}


class CmsisDspFastMathProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="loom-dsp-fast-math-")
        self.work = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_fixed_point_division_uses_one_direct_typed_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in DIVIDE_CASES
        )
        self.assertEqual({workload.case for workload in workloads}, DIVIDE_CASES)

        harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "harness",
        )
        cmake = (harness.source_dir / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("Testing/testmain.cpp", cmake)
        self.assertNotIn("loom_cmsis_dsp_framework", cmake)

        for workload in workloads:
            with self.subTest(case=workload.case):
                source_path = harness.protocol_source(workload.executable)
                source = source_path.read_text(encoding="utf-8")
                protocol, oracle = source.split("int main()", maxsplit=1)
                symbol = workload.protocol[0].symbol
                self.assertEqual(protocol.count(f"{symbol}("), 1)
                self.assertNotIn(f"{symbol}(", oracle)
                self.assertIn("kExpectedQuotient", oracle)
                self.assertIn("kExpectedShift", oracle)
                self.assertIn("ARM_MATH_SUCCESS", oracle)


if __name__ == "__main__":
    unittest.main()
