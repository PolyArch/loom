#!/usr/bin/env python3
"""Anchor tests for direct CMSIS-DSP convolution protocols."""

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


PARTIAL_CONVOLUTION_CASES = {
    "arm-conv-partial-f32",
    "arm-conv-partial-fast-opt-q15",
    "arm-conv-partial-fast-q15",
    "arm-conv-partial-fast-q31",
    "arm-conv-partial-opt-q15",
    "arm-conv-partial-opt-q7",
    "arm-conv-partial-q15",
    "arm-conv-partial-q31",
    "arm-conv-partial-q7",
}


class CmsisDspConvolutionProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="loom-dsp-convolution-")
        self.work = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_partial_convolution_uses_one_direct_typed_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in PARTIAL_CONVOLUTION_CASES
        )
        self.assertEqual(
            {workload.case for workload in workloads}, PARTIAL_CONVOLUTION_CASES
        )

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
                source_path, authoritative_owner = harness.protocol_source_owner(
                    workload.executable
                )
                self.assertEqual(authoritative_owner.name, "filtering_functions.h")
                source = source_path.read_text(encoding="utf-8")
                protocol, oracle = source.split("int main()", maxsplit=1)
                symbol = workload.protocol[0].symbol
                self.assertEqual(protocol.count(f"{symbol}("), 1)
                self.assertNotIn(f"{symbol}(", oracle)
                self.assertIn("ARM_MATH_SUCCESS", oracle)
                self.assertIn("output_matches_expected", oracle)


if __name__ == "__main__":
    unittest.main()
