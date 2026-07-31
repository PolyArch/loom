#!/usr/bin/env python3
"""Anchor tests for generated CMSIS-NN operator workloads."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))

import corpus_inventory  # noqa: E402
import corpus_workload_provider  # noqa: E402
from corpus_workload_errors import WorkloadProviderError  # noqa: E402


def _workload(case: str) -> corpus_inventory.ProgramWorkload:
    return next(
        workload
        for workload in corpus_inventory.load_workload_inventory(ROOT)
        if workload.suite == "cmsis-nn"
        and workload.case == case
        and workload.target_profile == corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
    )


class GeneratedCmsisNnProtocolTest(unittest.TestCase):
    def test_inventory_preserves_generated_public_provider_identity(self) -> None:
        workload = _workload("arm-relu-q7")

        self.assertIsInstance(
            workload.producer,
            corpus_inventory.CmsisNnGeneratedWorkloadProducer,
        )
        self.assertEqual(workload.producer.public_symbol, "arm_relu_q7")
        self.assertEqual(
            workload.producer.definitions,
            (
                "externals/cmsis-nn/Include/arm_nnfunctions.h",
                "externals/cmsis-nn/Include/arm_nnsupportfunctions.h",
            ),
        )

    def test_generated_activation_uses_public_protocol_and_independent_oracle(
        self,
    ) -> None:
        workload = _workload("arm-relu-q7")
        external_root = corpus_inventory.resolve_externals_root(ROOT)

        with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
            harness = corpus_workload_provider.materialize_cmsis_nn_harness(
                (workload,),
                external_root,
                Path(directory) / "harness",
            )

            self.assertEqual(
                harness.protocol_symbols(workload.executable),
                ("arm_relu_q7",),
            )
            compiled_owner, authoritative_owner = harness.protocol_source_owner(
                workload.executable
            )
            self.assertEqual(
                compiled_owner,
                external_root
                / "cmsis-nn"
                / "Source"
                / "ActivationFunctions"
                / "arm_relu_q7.c",
            )
            self.assertEqual(
                authoritative_owner,
                external_root / "cmsis-nn" / "Include" / "arm_nnfunctions.h",
            )

            source = (
                harness.source_dir
                / "generated"
                / "targets"
                / workload.executable
                / "OperatorProtocol.c"
            ).read_text()
            self.assertEqual(source.count("arm_relu_q7(data, kElementCount);"), 1)
            self.assertIn("kInput[index] < 0 ? 0 : kInput[index]", source)
            self.assertNotIn("LOOM_UNITY_SOURCE", source)

    def test_generated_protocol_rejects_signature_drift(self) -> None:
        workload = _workload("arm-relu-q7")
        malformed = dataclasses.replace(
            workload,
            protocol=(
                corpus_inventory.OperatorProtocolCall(
                    symbol="arm_relu_q7",
                    signature="void (int8_t *, uint32_t)",
                ),
            ),
        )

        with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
            with self.assertRaisesRegex(
                WorkloadProviderError,
                "generated CMSIS-NN protocol is unsupported",
            ):
                corpus_workload_provider.materialize_cmsis_nn_harness(
                    (malformed,),
                    corpus_inventory.resolve_externals_root(ROOT),
                    Path(directory) / "harness",
                )

    def test_generated_contiguous_array_protocol_families(self) -> None:
        expectations = {
            "arm-relu-q15": (
                "arm_relu_q15(data, kElementCount);",
                "kInput[index] < 0 ? 0 : kInput[index]",
            ),
            "arm-relu6-s8": (
                "arm_relu6_s8(data, kElementCount);",
                "kInput[index] > 6 ? 6 : kInput[index]",
            ),
            "arm-reshape-s8": (
                "arm_reshape_s8(kInput, output, kElementCount);",
                "output[index] != kInput[index]",
            ),
            "arm-q7-to-q15-with-offset": (
                "arm_q7_to_q15_with_offset(",
                "(int16_t)kInput[index] + kOffset",
            ),
        }
        external_root = corpus_inventory.resolve_externals_root(ROOT)

        for case, snippets in expectations.items():
            with self.subTest(case=case):
                workload = _workload(case)
                with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
                    harness = corpus_workload_provider.materialize_cmsis_nn_harness(
                        (workload,),
                        external_root,
                        Path(directory) / "harness",
                    )
                    source = (
                        harness.source_dir
                        / "generated"
                        / "targets"
                        / workload.executable
                        / "OperatorProtocol.c"
                    ).read_text()
                    for snippet in snippets:
                        self.assertIn(snippet, source)
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        (workload.protocol[0].symbol,),
                    )

    def test_unsupported_generated_protocol_does_not_admit_a_provider(self) -> None:
        self.assertTrue(
            corpus_workload_provider.supports_cmsis_nn_harness(_workload("arm-relu-q7"))
        )
        self.assertFalse(
            corpus_workload_provider.supports_cmsis_nn_harness(
                _workload("arm-softmax-u8")
            )
        )

    def test_header_defined_memory_protocols_use_a_mechanical_wrapper(self) -> None:
        expectations = {
            "arm-memcpy-s8": "arm_memcpy_s8(output, input, byte_count);",
            "arm-memcpy-q15": "arm_memcpy_q15(output, input, byte_count);",
            "arm-memset-s8": "arm_memset_s8(output, value, byte_count);",
        }
        external_root = corpus_inventory.resolve_externals_root(ROOT)

        for case, invocation in expectations.items():
            with self.subTest(case=case):
                workload = _workload(case)
                with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
                    harness = corpus_workload_provider.materialize_cmsis_nn_harness(
                        (workload,),
                        external_root,
                        Path(directory) / "harness",
                    )
                    source_path = (
                        harness.source_dir
                        / "generated"
                        / "targets"
                        / workload.executable
                        / "OperatorProtocol.c"
                    )
                    source = source_path.read_text()
                    self.assertEqual(source.count(invocation), 1)
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        ("loom_corpus_operator_protocol",),
                    )
                    self.assertEqual(
                        harness.protocol_source_owner(workload.executable),
                        (
                            source_path,
                            external_root
                            / "cmsis-nn"
                            / "Include"
                            / "arm_nnsupportfunctions.h",
                        ),
                    )

    def test_generated_concatenation_protocols_cover_each_axis(self) -> None:
        expected_calls = {
            "arm-concatenation-s8-w": "arm_concatenation_s8_w(",
            "arm-concatenation-s8-x": "arm_concatenation_s8_x(",
            "arm-concatenation-s8-y": "arm_concatenation_s8_y(",
            "arm-concatenation-s8-z": "arm_concatenation_s8_z(",
        }
        external_root = corpus_inventory.resolve_externals_root(ROOT)

        for case, invocation in expected_calls.items():
            with self.subTest(case=case):
                workload = _workload(case)
                with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
                    harness = corpus_workload_provider.materialize_cmsis_nn_harness(
                        (workload,),
                        external_root,
                        Path(directory) / "harness",
                    )
                    source = (
                        harness.source_dir
                        / "generated"
                        / "targets"
                        / workload.executable
                        / "OperatorProtocol.c"
                    ).read_text()
                    self.assertEqual(source.count(invocation), 1)
                    self.assertIn("expected[destination] = input[source];", source)
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        (workload.protocol[0].symbol,),
                    )

    def test_generated_buffer_size_queries_use_typed_inputs(self) -> None:
        expected_oracles = {
            "arm-avgpool-s8-get-buffer-size-dsp": "kChannels * sizeof(int32_t)",
            "arm-avgpool-s8-get-buffer-size-mve": "0",
            "arm-avgpool-s16-get-buffer-size-dsp": "kChannels * sizeof(int32_t)",
            "arm-avgpool-s16-get-buffer-size-mve": "0",
            "arm-fully-connected-s8-get-buffer-size-dsp": "0",
            "arm-fully-connected-s8-get-buffer-size-mve": (
                "kDimensions.c * sizeof(int32_t)"
            ),
            "arm-fully-connected-s16-get-buffer-size-dsp": "0",
            "arm-fully-connected-s16-get-buffer-size-mve": "0",
            "arm-svdf-s8-get-buffer-size-dsp": "0",
            "arm-svdf-s8-get-buffer-size-mve": ("kDimensions.n * sizeof(int32_t)"),
        }
        external_root = corpus_inventory.resolve_externals_root(ROOT)

        for case, expected in expected_oracles.items():
            with self.subTest(case=case):
                workload = _workload(case)
                with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
                    harness = corpus_workload_provider.materialize_cmsis_nn_harness(
                        (workload,),
                        external_root,
                        Path(directory) / "harness",
                    )
                    source = (
                        harness.source_dir
                        / "generated"
                        / "targets"
                        / workload.executable
                        / "OperatorProtocol.c"
                    ).read_text()
                    self.assertIn(f"const int32_t expected = {expected};", source)
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        (workload.protocol[0].symbol,),
                    )

    def test_generated_elementwise_mul_protocols_use_independent_oracles(self) -> None:
        expectations = {
            "arm-elementwise-mul-s16-batch-offset": (
                "arm_elementwise_mul_s16_batch_offset(",
                "reference = input_1[index] * input_2[index] + kOutputOffset",
            ),
            "arm-elementwise-mul-s16-s8": (
                "arm_elementwise_mul_s16_s8(",
                "reference = input_1[index] * input_2[index] + kOutputOffset",
            ),
            "arm-elementwise-mul-acc-s16": (
                "arm_elementwise_mul_acc_s16(",
                "reference = initial_output[index] + input_1[index] * input_2[index]",
                '#include "arm_nnsupportfunctions.h"',
            ),
        }
        external_root = corpus_inventory.resolve_externals_root(ROOT)

        for case, snippets in expectations.items():
            with self.subTest(case=case):
                workload = _workload(case)
                with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
                    harness = corpus_workload_provider.materialize_cmsis_nn_harness(
                        (workload,),
                        external_root,
                        Path(directory) / "harness",
                    )
                    source = (
                        harness.source_dir
                        / "generated"
                        / "targets"
                        / workload.executable
                        / "OperatorProtocol.c"
                    ).read_text()
                    for snippet in snippets:
                        self.assertIn(snippet, source)
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        (workload.protocol[0].symbol,),
                    )

    def test_header_defined_fixed_point_protocols_use_literal_oracles(self) -> None:
        expected_literals = {
            "arm-nn-doubling-high-mult": "2147483647",
            "arm-nn-doubling-high-mult-no-sat": "56779306",
            "arm-nn-divide-by-power-of-two": "964506",
            "arm-nn-requantize": "3520317",
            "arm-nn-requantize-s64": "134213632",
            "arm-nn-mult-by-power-of-two": "2147483640",
            "arm-nn-exp-on-negative-values": "1672462419",
            "arm-nn-one-over-one-plus-x-for-x-in-0-1": "1431655762",
        }
        external_root = corpus_inventory.resolve_externals_root(ROOT)

        for case, expected in expected_literals.items():
            with self.subTest(case=case):
                workload = _workload(case)
                with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
                    harness = corpus_workload_provider.materialize_cmsis_nn_harness(
                        (workload,),
                        external_root,
                        Path(directory) / "harness",
                    )
                    source = (
                        harness.source_dir
                        / "generated"
                        / "targets"
                        / workload.executable
                        / "OperatorProtocol.c"
                    ).read_text()
                    self.assertIn(expected, source)
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        ("loom_corpus_operator_protocol",),
                    )

    def test_header_defined_packed_memory_protocols_preserve_pointer_state(
        self,
    ) -> None:
        cases = (
            "arm-nn-read-q15x2-ia",
            "arm-nn-read-s16x2",
            "arm-nn-read-s8x4-ia",
            "arm-nn-read-s8x4",
            "arm-nn-write-q15x2-ia",
            "arm-nn-write-s8x4-ia",
        )
        external_root = corpus_inventory.resolve_externals_root(ROOT)

        for case in cases:
            with self.subTest(case=case):
                workload = _workload(case)
                with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
                    harness = corpus_workload_provider.materialize_cmsis_nn_harness(
                        (workload,),
                        external_root,
                        Path(directory) / "harness",
                    )
                    source = (
                        harness.source_dir
                        / "generated"
                        / "targets"
                        / workload.executable
                        / "OperatorProtocol.c"
                    ).read_text()
                    self.assertIn("kGroupCount", source)
                    self.assertIn("expected_cursor", source)
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        ("loom_corpus_operator_protocol",),
                    )

    def test_header_defined_shape_predicates_cover_true_and_false_cases(self) -> None:
        cases = (
            "arm-check-broadcast-required",
            "arm-nn-is-convolve-1-x-n",
            "arm-nn-is-convolve-1x1-fast",
            "arm-nn-is-convolve-1x1",
        )
        external_root = corpus_inventory.resolve_externals_root(ROOT)

        for case in cases:
            with self.subTest(case=case):
                workload = _workload(case)
                with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
                    harness = corpus_workload_provider.materialize_cmsis_nn_harness(
                        (workload,),
                        external_root,
                        Path(directory) / "harness",
                    )
                    source = (
                        harness.source_dir
                        / "generated"
                        / "targets"
                        / workload.executable
                        / "OperatorProtocol.c"
                    ).read_text()
                    self.assertIn("expected_true", source)
                    self.assertIn("expected_false", source)
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        ("loom_corpus_operator_protocol",),
                    )


if __name__ == "__main__":
    unittest.main()
