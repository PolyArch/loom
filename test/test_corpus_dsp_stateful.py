#!/usr/bin/env python3
"""Anchor tests for stateful CMSIS-DSP operator protocols."""

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


class CmsisDspStatefulProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="loom-dsp-stateful-")
        self.work = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_fir_variants_use_one_atomic_init_execute_wrapper(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in {"arm-fir-f16", "arm-fir-q15"}
        )
        self.assertEqual(len(workloads), 2)

        for ordinal, workload in enumerate(workloads):
            with self.subTest(case=workload.case):
                harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
                    (workload,),
                    corpus_inventory.resolve_externals_root(ROOT),
                    self.work / f"harness-{ordinal}",
                )
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
                self.assertNotIn("scratch_output[", protocol)
                suffix = workload.case.removeprefix("arm-fir-")
                self.assertEqual(protocol.count(f"arm_fir_init_{suffix}("), 1)
                self.assertEqual(protocol.count(f"arm_fir_{suffix}("), 2)
                self.assertIn("kExpected", source)
                self.assertIn("return oracle_matches(output) ? 0 : 1;", source)
                if workload.case.endswith("q15"):
                    self.assertIn("(void)arm_fir_init_q15", source)

    def test_svm_variants_keep_model_init_and_predictions_atomic(self) -> None:
        cases = {
            "arm-svm-linear-predict-f32",
            "arm-svm-polynomial-predict-f16",
        }
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in cases
        )
        self.assertEqual(len(workloads), len(cases))

        for ordinal, workload in enumerate(workloads):
            with self.subTest(case=workload.case):
                harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
                    (workload,),
                    corpus_inventory.resolve_externals_root(ROOT),
                    self.work / f"svm-harness-{ordinal}",
                )
                self.assertEqual(
                    harness.protocol_symbols(workload.executable),
                    ("loom_corpus_operator_protocol",),
                )
                self.assertEqual(harness.expected_entry_result(workload.executable), 0)
                compiled_owner = harness.protocol_source(workload.executable)
                source = compiled_owner.read_text(encoding="utf-8")
                kernel = workload.case.removeprefix("arm-svm-").split("-predict-")[0]
                suffix = workload.case.rsplit("-", maxsplit=1)[1]
                protocol = source.split("int main()", maxsplit=1)[0]
                self.assertEqual(protocol.count(f"arm_svm_{kernel}_init_{suffix}("), 1)
                self.assertEqual(
                    protocol.count(f"arm_svm_{kernel}_predict_{suffix}("), 1
                )
                self.assertIn(
                    "for (std::size_t sample = 0; sample < kSampleCount; ++sample)",
                    protocol,
                )
                self.assertNotIn("instance{}", protocol)
                self.assertIn("return oracle_matches(output) ? 0 : 1;", source)
                if kernel == "polynomial":
                    self.assertIn("kDegree", protocol)

    def test_biquad_variants_keep_init_and_filtering_atomic(self) -> None:
        cases = {
            "arm-biquad-cas-df1-32x64-q31",
            "arm-biquad-cascade-df1-f32",
            "arm-biquad-cascade-stereo-df2t-f16",
        }
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in cases
        )
        self.assertEqual(len(workloads), len(cases))

        for ordinal, workload in enumerate(workloads):
            with self.subTest(case=workload.case):
                harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
                    (workload,),
                    corpus_inventory.resolve_externals_root(ROOT),
                    self.work / f"biquad-harness-{ordinal}",
                )
                self.assertEqual(
                    harness.protocol_symbols(workload.executable),
                    ("loom_corpus_operator_protocol",),
                )
                compiled_owner = harness.protocol_source(workload.executable)
                source = compiled_owner.read_text(encoding="utf-8")
                protocol = source.split("int main()", maxsplit=1)[0]
                self.assertEqual(protocol.count(f"{workload.protocol[0].symbol}("), 1)
                self.assertEqual(protocol.count(f"{workload.protocol[1].symbol}("), 1)
                self.assertIn("return oracle_matches(output) ? 0 : 1;", source)

    def test_rate_conversion_keeps_init_and_processing_atomic(self) -> None:
        cases = {
            "arm-fir-decimate-f64",
            "arm-fir-interpolate-q15",
        }
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in cases
        )
        self.assertEqual(len(workloads), len(cases))

        for ordinal, workload in enumerate(workloads):
            with self.subTest(case=workload.case):
                harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
                    (workload,),
                    corpus_inventory.resolve_externals_root(ROOT),
                    self.work / f"rate-conversion-harness-{ordinal}",
                )
                self.assertEqual(
                    harness.protocol_symbols(workload.executable),
                    ("loom_corpus_operator_protocol",),
                )
                compiled_owner = harness.protocol_source(workload.executable)
                source = compiled_owner.read_text(encoding="utf-8")
                protocol = source.split("int main()", maxsplit=1)[0]
                self.assertEqual(protocol.count(f"{workload.protocol[0].symbol}("), 1)
                self.assertEqual(protocol.count(f"{workload.protocol[1].symbol}("), 1)
                self.assertIn("return oracle_matches(output) ? 0 : 1;", source)


if __name__ == "__main__":
    unittest.main()
