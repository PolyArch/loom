#!/usr/bin/env python3
"""Anchor tests for direct CMSIS-DSP distance protocols."""

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


DISTANCE_CASES = {
    "arm-correlation-distance-f16",
    "arm-correlation-distance-f32",
    "arm-cosine-distance-f16",
    "arm-cosine-distance-f32",
    "arm-cosine-distance-f64",
}


class CmsisDspDistanceProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="loom-dsp-distance-")
        self.work = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_distance_protocols_use_one_direct_typed_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in DISTANCE_CASES
        )
        self.assertEqual({workload.case for workload in workloads}, DISTANCE_CASES)

        for profile in {workload.target_profile for workload in workloads}:
            selected = tuple(
                workload for workload in workloads if workload.target_profile == profile
            )
            harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
                selected,
                corpus_inventory.resolve_externals_root(ROOT),
                self.work / profile,
            )
            cmake = (harness.source_dir / "CMakeLists.txt").read_text(encoding="utf-8")
            self.assertNotIn("Testing/testmain.cpp", cmake)
            self.assertNotIn("loom_cmsis_dsp_framework", cmake)

            for workload in selected:
                with self.subTest(case=workload.case):
                    source_path, authoritative_owner = harness.protocol_source_owner(
                        workload.executable
                    )
                    self.assertIn(
                        authoritative_owner.name,
                        {"distance_functions.h", "distance_functions_f16.h"},
                    )
                    source = source_path.read_text(encoding="utf-8")
                    protocol, oracle = source.split("int main()", maxsplit=1)
                    symbol = workload.protocol[0].symbol
                    self.assertEqual(protocol.count(f"{symbol}("), 1)
                    self.assertNotIn(f"{symbol}(", oracle)
                    self.assertIn("output_matches_expected", oracle)


if __name__ == "__main__":
    unittest.main()
