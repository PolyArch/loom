#!/usr/bin/env python3
"""Anchor tests for generated CMSIS-DSP operator protocols."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))

import corpus_inventory  # noqa: E402
import corpus_dsp_generated  # noqa: E402
import corpus_workload_provider  # noqa: E402


TRANSFORM_QUERY_CASES = {
    "arm-cfft-output-buffer-size",
    "arm-cfft-tmp-buffer-size",
    "arm-cifft-output-buffer-size",
    "arm-mfcc-tmp-buffer-size",
    "arm-rfft-output-buffer-size",
    "arm-rfft-tmp-buffer-size",
    "arm-rifft-input-buffer-size",
}
LIFECYCLE_CASES = {
    "arm-mat-init-f16",
    "arm-mat-init-f32",
    "arm-mat-init-f64",
    "arm-mat-init-q15",
    "arm-mat-init-q31",
    "arm-mat-init-q7",
    "arm-pid-reset-f32",
    "arm-pid-reset-q15",
    "arm-pid-reset-q31",
}


class CmsisDspGeneratedProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="loom-dsp-generated-")
        self.work = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_transform_queries_have_one_typed_generated_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in TRANSFORM_QUERY_CASES
        )

        self.assertEqual(len(workloads), len(TRANSFORM_QUERY_CASES))
        for workload in workloads:
            with self.subTest(case=workload.case):
                self.assertIsInstance(
                    workload.producer,
                    corpus_inventory.CmsisDspGeneratedWorkloadProducer,
                )
                self.assertEqual(workload.producer.selector_kind, "transform-query")
                self.assertTrue(
                    corpus_workload_provider.supports_cmsis_dsp_harness(workload)
                )

    def test_transform_query_provider_keeps_exact_calls_and_oracles(self) -> None:
        selected_cases = {
            "arm-cfft-output-buffer-size",
            "arm-rfft-tmp-buffer-size",
            "arm-mfcc-tmp-buffer-size",
        }
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in selected_cases
        )
        self.assertEqual(len(workloads), len(selected_cases))

        harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "harness",
        )

        by_case = {workload.case: workload for workload in workloads}
        for workload in workloads:
            with self.subTest(case=workload.case):
                self.assertEqual(
                    harness.protocol_symbols(workload.executable),
                    ("loom_corpus_operator_protocol",),
                )
                self.assertEqual(harness.expected_entry_result(workload.executable), 0)
                compiled_owner, authoritative_owner = harness.protocol_source_owner(
                    workload.executable
                )
                self.assertEqual(compiled_owner.name, "OperatorProtocol.cpp")
                self.assertEqual(authoritative_owner.name, "transform_functions.h")
                source = compiled_owner.read_text(encoding="utf-8")
                protocol = source.split("int main()", maxsplit=1)[0]
                self.assertIn(f"{workload.protocol[0].symbol}(", protocol)
                self.assertIn("int main()", source)
                self.assertNotIn("int main(int", source)

        cfft_source = harness.protocol_source_owner(
            by_case["arm-cfft-output-buffer-size"].executable
        )[0].read_text(encoding="utf-8")
        self.assertIn("ARM_MATH_F32", cfft_source)
        self.assertIn("2 * sample_count", cfft_source)

        rfft_tmp_source = harness.protocol_source_owner(
            by_case["arm-rfft-tmp-buffer-size"].executable
        )[0].read_text(encoding="utf-8")
        self.assertIn("ARM_MATH_SCALAR_ARCH", rfft_tmp_source)
        self.assertIn("!= 0", rfft_tmp_source)

        mfcc_source = harness.protocol_source_owner(
            by_case["arm-mfcc-tmp-buffer-size"].executable
        )[0].read_text(encoding="utf-8")
        self.assertIn("use_cfft", mfcc_source)
        self.assertIn("buf_id", mfcc_source)
        self.assertIn("2 * sample_count", mfcc_source)

    def test_lifecycle_protocols_have_one_typed_generated_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in LIFECYCLE_CASES
        )

        self.assertEqual(len(workloads), len(LIFECYCLE_CASES))
        for workload in workloads:
            with self.subTest(case=workload.case):
                self.assertIsInstance(
                    workload.producer,
                    corpus_inventory.CmsisDspGeneratedWorkloadProducer,
                )
                self.assertEqual(
                    workload.producer.selector_kind, "lifecycle-completion"
                )
                self.assertTrue(
                    corpus_workload_provider.supports_cmsis_dsp_harness(workload)
                )
                if workload.case == "arm-mat-init-f16":
                    protocol = corpus_dsp_generated.lifecycle_protocol(workload)
                    self.assertIsNotNone(protocol)
                    self.assertEqual(protocol.owner_header, "matrix_functions_f16.h")

    def test_lifecycle_provider_observes_structural_effects(self) -> None:
        selected_cases = {"arm-mat-init-f32", "arm-pid-reset-q15"}
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in selected_cases
        )
        self.assertEqual(len(workloads), len(selected_cases))

        harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "lifecycle-harness",
        )
        by_case = {workload.case: workload for workload in workloads}

        matrix = by_case["arm-mat-init-f32"]
        matrix_source, matrix_owner = harness.protocol_source_owner(matrix.executable)
        self.assertEqual(matrix_owner.name, "matrix_functions.h")
        rendered_matrix = matrix_source.read_text(encoding="utf-8")
        self.assertIn(
            "arm_mat_init_f32(&instance, rows, columns, data)", rendered_matrix
        )
        self.assertIn("instance.numRows != rows", rendered_matrix)
        self.assertIn("instance.numCols != columns", rendered_matrix)
        self.assertIn("instance.pData != data", rendered_matrix)
        self.assertIn("data[6];", rendered_matrix)
        self.assertNotIn("data[6]{}", rendered_matrix)

        reset = by_case["arm-pid-reset-q15"]
        reset_source, reset_owner = harness.protocol_source_owner(reset.executable)
        self.assertEqual(reset_owner.name, "controller_functions.h")
        rendered_reset = reset_source.read_text(encoding="utf-8")
        self.assertIn("arm_pid_reset_q15(&instance)", rendered_reset)
        self.assertIn("instance.state[index] != 0", rendered_reset)
        self.assertIn("int main()", rendered_reset)


if __name__ == "__main__":
    unittest.main()
