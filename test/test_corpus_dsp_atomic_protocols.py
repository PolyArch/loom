#!/usr/bin/env python3
"""Anchor tests for atomic multi-call CMSIS-DSP protocols."""

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


ATOMIC_CASES = {
    "arm-dtw-init-window-q7",
    "arm-mat-mult-fast-q15",
    "arm-mat-mult-q7",
    "arm-merge-sort-f32",
    "arm-sort-f32",
    "arm-spline-f32",
}

TRANSFORM_CASES = {
    "arm-cfft-f16",
    "arm-cfft-f32",
    "arm-cfft-f64",
    "arm-cfft-q15",
    "arm-cfft-q31",
    "arm-mfcc-f16",
    "arm-mfcc-f32",
    "arm-mfcc-q15",
    "arm-mfcc-q31",
    "arm-rfft-fast-f16",
    "arm-rfft-fast-f32",
    "arm-rfft-fast-f64",
    "arm-rfft-q15",
    "arm-rfft-q31",
}


class CmsisDspAtomicProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="loom-dsp-atomic-")
        self.work = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_multi_call_workloads_use_one_atomic_operator_wrapper(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in ATOMIC_CASES
        )
        self.assertEqual({workload.case for workload in workloads}, ATOMIC_CASES)

        harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "harness",
        )

        for workload in workloads:
            with self.subTest(case=workload.case):
                self.assertEqual(
                    harness.protocol_symbols(workload.executable),
                    ("loom_corpus_operator_protocol",),
                )
                self.assertEqual(harness.expected_entry_result(workload.executable), 0)
                compiled_owner = harness.protocol_source(workload.executable)
                self.assertEqual(compiled_owner.name, "OperatorProtocol.cpp")
                source = compiled_owner.read_text(encoding="utf-8")
                protocol = source.split("int main()", maxsplit=1)[0]
                self.assertNotIn(" instance;", protocol)
                self.assertNotIn(" instance{}", protocol)
                self.assertNotIn("costs_data[", protocol)
                self.assertNotIn("coefficients[", protocol)
                positions = []
                for call in workload.protocol:
                    self.assertEqual(protocol.count(f"{call.symbol}("), 1)
                    positions.append(protocol.index(f"{call.symbol}("))
                self.assertEqual(positions, sorted(positions))
                self.assertNotIn("testmain", source)
                self.assertNotIn("Testing::", source)

    def test_transform_lifecycles_use_one_atomic_operator_wrapper(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in TRANSFORM_CASES
            and workload.target_profile
            in {
                corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
                corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE,
            }
        )
        self.assertEqual({workload.case for workload in workloads}, TRANSFORM_CASES)

        for profile_ordinal, profile in enumerate(
            sorted({workload.target_profile for workload in workloads})
        ):
            selected = tuple(
                workload for workload in workloads if workload.target_profile == profile
            )
            harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
                selected,
                corpus_inventory.resolve_externals_root(ROOT),
                self.work / f"transform-harness-{profile_ordinal}",
            )
            for workload in selected:
                with self.subTest(case=workload.case):
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        ("loom_corpus_operator_protocol",),
                    )
                    compiled_owner = harness.protocol_source(workload.executable)
                    self.assertEqual(compiled_owner.name, "OperatorProtocol.cpp")
                    source = compiled_owner.read_text(encoding="utf-8")
                    protocol = source.split("int main()", maxsplit=1)[0]
                    self.assertNotIn(" instance;", protocol)
                    self.assertNotIn("mutable_input[", protocol)
                    self.assertNotIn("scratch[", protocol)
                    self.assertNotIn("for (", protocol)
                    positions = []
                    for call in workload.protocol:
                        self.assertEqual(protocol.count(f"{call.symbol}("), 1)
                        positions.append(protocol.index(f"{call.symbol}("))
                    self.assertEqual(positions, sorted(positions))
                    self.assertNotIn("testmain", source)
                    self.assertNotIn("Testing::", source)


if __name__ == "__main__":
    unittest.main()
