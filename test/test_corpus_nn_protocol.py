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


def _workload_any_profile(case: str) -> corpus_inventory.ProgramWorkload:
    return next(
        workload
        for workload in corpus_inventory.load_workload_inventory(ROOT)
        if workload.suite == "cmsis-nn" and workload.case == case
    )


class GeneratedCmsisNnProtocolTest(unittest.TestCase):
    def test_generated_layout_protocols_own_production_sources(self) -> None:
        expectations = {
            "arm-pad-s8": (
                "arm_pad_s8(",
                "Source/PadFunctions/arm_pad_s8.c",
            ),
            "arm-transpose-s8": (
                "arm_transpose_s8(",
                "Source/TransposeFunctions/arm_transpose_s8.c",
            ),
            "arm-depthwise-conv-wrapper-s16-get-buffer-size": (
                "arm_depthwise_conv_wrapper_s16_get_buffer_size(",
                "Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s16.c",
            ),
        }
        external_root = corpus_inventory.resolve_externals_root(ROOT)

        for case, (invocation, owner) in expectations.items():
            with self.subTest(case=case):
                workload = _workload(case)
                self.assertIsInstance(
                    workload.producer,
                    corpus_inventory.CmsisNnGeneratedWorkloadProducer,
                )
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
                compiled_owner, authoritative_owner = (
                    harness.protocol_source_owner(workload.executable)
                )
                self.assertEqual(compiled_owner, external_root / "cmsis-nn" / owner)
                self.assertEqual(
                    authoritative_owner,
                    external_root / "cmsis-nn" / "Include" / "arm_nnfunctions.h",
                )

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

    def test_generated_protocol_preserves_workload_compiler_semantics(self) -> None:
        workload = _workload("arm-nn-mult-by-power-of-two")
        self.assertEqual(workload.compiler_flags, ("-fwrapv",))

        with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
            harness = corpus_workload_provider.materialize_cmsis_nn_harness(
                (workload,),
                corpus_inventory.resolve_externals_root(ROOT),
                Path(directory) / "harness",
            )
            cmake = (harness.source_dir / "CMakeLists.txt").read_text()
            self.assertIn(
                f"target_compile_options({workload.executable} PRIVATE "
                "-fno-inline-functions -fwrapv)",
                cmake,
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
        mve_only = _workload_any_profile(
            "arm-nn-mat-mult-nt-interleaved-t-even-s4"
        )
        self.assertEqual(mve_only.target_profile, "mve")
        self.assertFalse(
            corpus_workload_provider.supports_cmsis_nn_harness(mve_only)
        )

    def test_generated_softmax_protocols_use_official_tfl_oracle(self) -> None:
        external_root = corpus_inventory.resolve_externals_root(ROOT)
        expectations = {
            "arm-nn-softmax-common-s8": (
                "arm_nn_softmax_common_s8(",
                "softmax_output_ref[index]",
            ),
            "arm-softmax-u8": (
                "arm_softmax_u8(",
                "softmax_output_ref[index] + 128",
            ),
        }

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
                    self.assertIn(
                        '#include "TestCases/TestData/softmax/test_data.h"',
                        source,
                    )
                    for snippet in snippets:
                        self.assertIn(snippet, source)
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        (workload.protocol[0].symbol,),
                    )

    def test_generated_s32_matrix_protocol_uses_strided_reference(self) -> None:
        workload = _workload("arm-nn-mat-mult-nt-t-s8-s32")
        external_root = corpus_inventory.resolve_externals_root(ROOT)

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
            self.assertIn("arm_nn_mat_mult_nt_t_s8_s32(", source)
            self.assertIn("kDstIndexOffset = 2", source)
            self.assertIn("(kLhs[row * kRhsRows + depth] + kLhsOffset)", source)
            self.assertIn("kRhs[column * kRhsRows + depth]", source)
            self.assertIn("expected[destination] +=", source)
            self.assertEqual(
                harness.protocol_symbols(workload.executable),
                ("arm_nn_mat_mult_nt_t_s8_s32",),
            )

    def test_generated_s16_vec_mat_protocols_use_dot_product_reference(self) -> None:
        external_root = corpus_inventory.resolve_externals_root(ROOT)
        cases = (
            "arm-nn-vec-mat-mult-t-s16",
            "arm-nn-vec-mat-mult-t-s16-s16",
        )

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
                    self.assertIn(f"{workload.protocol[0].symbol}(", source)
                    self.assertIn("kReducedMultiplier = 16384", source)
                    self.assertIn("kShift = 1", source)
                    self.assertIn("expected += kLhs[column] *", source)
                    self.assertIn("kRhs[row * kColumnCount + column]", source)
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        (workload.protocol[0].symbol,),
                    )

    def test_generated_accumulating_vec_mat_protocols_preserve_batches(self) -> None:
        external_root = corpus_inventory.resolve_externals_root(ROOT)
        cases = (
            "arm-nn-vec-mat-mul-result-acc-s16",
            "arm-nn-vec-mat-mul-result-acc-s8-s16",
        )

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
                    self.assertIn(f"{workload.protocol[0].symbol}(", source)
                    self.assertIn("kBatchCount = 2", source)
                    self.assertIn("kBatchOffset = 2", source)
                    self.assertIn("expected[batch * kRowCount + row] +=", source)
                    self.assertIn(
                        "kLhs[batch * kColumnCount * kBatchOffset + column]",
                        source,
                    )
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        (workload.protocol[0].symbol,),
                    )

    def test_generated_lstm_protocols_use_composed_primitive_oracles(self) -> None:
        external_root = corpus_inventory.resolve_externals_root(ROOT)
        expectations = {
            "arm-nn-lstm-calculate-gate-s16": (
                "arm_nn_vec_mat_mul_result_acc_s16(",
                "reference_gate_s16(",
            ),
            "arm-nn-lstm-calculate-gate-s8-s16": (
                "arm_nn_vec_mat_mul_result_acc_s8_s16(",
                "reference_gate_s8_s16(",
            ),
            "arm-nn-lstm-step-s16": (
                "arm_elementwise_mul_acc_s16(",
                "reference_step_s16(",
            ),
            "arm-nn-lstm-step-s8": (
                "arm_elementwise_mul_s16_s8(",
                "reference_step_s8(",
            ),
        }

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

                symbol = workload.protocol[0].symbol
                self.assertEqual(source.count(f"{symbol}("), 1)
                self.assertIn("arm_nn_activation_s16(", source)
                for snippet in snippets:
                    self.assertIn(snippet, source)
                self.assertEqual(
                    harness.protocol_symbols(workload.executable),
                    (symbol,),
                )

    def test_generated_s8_vec_mat_protocols_preserve_offsets_and_stride(self) -> None:
        external_root = corpus_inventory.resolve_externals_root(ROOT)
        cases = (
            "arm-nn-vec-mat-mult-t-s8",
            "arm-nn-vec-mat-mult-t-per-ch-s8",
            "arm-nn-vec-mat-mult-t-svdf-s8",
        )

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
                    self.assertIn(f"{workload.protocol[0].symbol}(", source)
                    self.assertIn("kInputOffset = 3", source)
                    self.assertIn("kAddressOffset = 2", source)
                    self.assertIn("kLhs[column] + kInputOffset", source)
                    self.assertIn("row * kAddressOffset", source)
                    self.assertIn("output[index] != expected[index]", source)
                    self.assertEqual(
                        harness.protocol_symbols(workload.executable),
                        (workload.protocol[0].symbol,),
                    )

    def test_generated_s4_vec_mat_protocol_decodes_packed_input_for_oracle(self) -> None:
        workload = _workload("arm-nn-vec-mat-mult-t-s4")
        external_root = corpus_inventory.resolve_externals_root(ROOT)

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

        self.assertIn("arm_nn_vec_mat_mult_t_s4(", source)
        self.assertIn("unpack_s4(kPackedRhs, packed_index)", source)
        self.assertIn("kLhs[column] + kInputOffset", source)
        self.assertIn("output[row] != expected", source)
        self.assertEqual(
            harness.protocol_symbols(workload.executable),
            ("arm_nn_vec_mat_mult_t_s4",),
        )

    def test_generated_nt_t_matrix_protocols_preserve_strided_lhs(self) -> None:
        external_root = corpus_inventory.resolve_externals_root(ROOT)
        cases = (
            "arm-nn-mat-mult-nt-t-s8",
            "arm-nn-mat-mult-nt-t-s4",
        )

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

                self.assertIn(f"{workload.protocol[0].symbol}(", source)
                self.assertIn("kLhsStride", source)
                self.assertIn("lhs_row * kLhsStride + column", source)
                self.assertIn("output[output_index] != expected", source)
                self.assertEqual(
                    harness.protocol_symbols(workload.executable),
                    (workload.protocol[0].symbol,),
                )

    def test_generated_convolution_matrix_kernels_validate_two_vectors(self) -> None:
        external_root = corpus_inventory.resolve_externals_root(ROOT)
        cases = (
            "arm-nn-mat-mult-kernel-s8-s16",
            "arm-nn-mat-mult-kernel-row-offset-s8-s16",
            "arm-nn-mat-mult-kernel-s16",
        )

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

                self.assertIn(f"{workload.protocol[0].symbol}(", source)
                self.assertIn("kInputB[vector * kAlignedColumns + column]", source)
                self.assertIn("returned != output + kOutputSpan", source)
                self.assertIn("output[index] != expected[index]", source)
                self.assertEqual(
                    harness.protocol_symbols(workload.executable),
                    (workload.protocol[0].symbol,),
                )

    def test_generated_s4_convolution_kernel_decodes_weight_rows(self) -> None:
        workload = _workload("arm-nn-mat-mult-kernel-s4-s16")
        external_root = corpus_inventory.resolve_externals_root(ROOT)

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

        self.assertIn("arm_nn_mat_mult_kernel_s4_s16(", source)
        self.assertIn("unpack_s4(kPackedInputA, packed_index)", source)
        self.assertIn("kInputB[vector * kColumns + column]", source)
        self.assertIn("returned != output + kOutputCount", source)
        self.assertEqual(
            harness.protocol_symbols(workload.executable),
            ("arm_nn_mat_mult_kernel_s4_s16",),
        )

    def test_generated_transpose_convolution_row_accumulates_overlap(self) -> None:
        workload = _workload("arm-nn-transpose-conv-row-s8-s32")
        external_root = corpus_inventory.resolve_externals_root(ROOT)

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

        self.assertIn("arm_nn_transpose_conv_row_s8_s32(", source)
        self.assertIn("input_x * kStrideX + filter_x", source)
        self.assertIn("expected[output_index] +=", source)
        self.assertIn("output[index] != expected[index]", source)
        self.assertEqual(
            harness.protocol_symbols(workload.executable),
            ("arm_nn_transpose_conv_row_s8_s32",),
        )

    def test_generated_transpose_convolution_protocol_has_direct_oracle(self) -> None:
        workload = _workload("arm-transpose-conv-s8")
        external_root = corpus_inventory.resolve_externals_root(ROOT)

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

        self.assertIn("arm_transpose_conv_s8(", source)
        self.assertIn("input_y + filter_y", source)
        self.assertIn("input_x + filter_x", source)
        self.assertIn("expected[output_index] +=", source)
        self.assertNotIn("arm_transpose_conv_wrapper_s8(", source)
        self.assertEqual(
            harness.protocol_symbols(workload.executable),
            ("arm_transpose_conv_s8",),
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
            "arm-convolve-s8-get-buffer-size-mve": "704",
            "arm-depthwise-conv-s8-opt-get-buffer-size-dsp": "330",
            "arm-depthwise-conv-s8-opt-get-buffer-size-mve": "7440",
            "arm-convolve-1-x-n-s4-get-buffer-size": "660",
            "arm-convolve-wrapper-s16-get-buffer-size-dsp": "660",
            "arm-convolve-wrapper-s16-get-buffer-size-mve": "1344",
            "arm-convolve-wrapper-s4-get-buffer-size-dsp": "660",
            "arm-convolve-wrapper-s4-get-buffer-size-mve": "704",
            "arm-convolve-wrapper-s8-get-buffer-size-dsp": "672",
            "arm-convolve-wrapper-s8-get-buffer-size-mve": "704",
            "arm-depthwise-conv-wrapper-s16-get-buffer-size-dsp": "330",
            "arm-depthwise-conv-wrapper-s16-get-buffer-size-mve": "1328",
            "arm-depthwise-conv-wrapper-s4-get-buffer-size-dsp": "330",
            "arm-depthwise-conv-wrapper-s4-get-buffer-size-mve": "7440",
            "arm-depthwise-conv-wrapper-s8-get-buffer-size-dsp": "330",
            "arm-depthwise-conv-wrapper-s8-get-buffer-size-mve": "7440",
            "arm-transpose-conv-s8-get-buffer-size-mve": "3588",
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

    def test_s16_activation_protocol_uses_lut_dependency_and_literal_oracle(
        self,
    ) -> None:
        workload = _workload("arm-nn-activation-s16")
        external_root = corpus_inventory.resolve_externals_root(ROOT)

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
            self.assertIn("kExpectedSigmoid", source)
            self.assertIn("32179", source)
            self.assertIn("kExpectedTanh", source)
            self.assertIn("-32746", source)
            self.assertEqual(
                harness.protocol_symbols(workload.executable),
                ("arm_nn_activation_s16",),
            )
            compiled_owner, _ = harness.protocol_source_owner(workload.executable)
            self.assertEqual(
                compiled_owner,
                external_root
                / "cmsis-nn"
                / "Source"
                / "ActivationFunctions"
                / "arm_nn_activation_s16.c",
            )

    def test_header_defined_packed_memory_protocols_preserve_pointer_state(
        self,
    ) -> None:
        cases = (
            "arm-nn-read-q15x2-ia",
            "arm-nn-read-s16x2",
            "arm-nn-read-s8x2-ia",
            "arm-nn-read-s8x2",
            "arm-nn-read-s8x4-ia",
            "arm-nn-read-s8x4",
            "arm-nn-write-q15x2-ia",
            "arm-nn-write-s8x2-ia",
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
