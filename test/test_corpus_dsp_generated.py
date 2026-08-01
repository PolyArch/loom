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
CONVOLUTION_CORRELATION_CASES = {
    "arm-conv-fast-opt-q15",
    "arm-conv-fast-q15",
    "arm-conv-fast-q31",
    "arm-conv-opt-q15",
    "arm-conv-opt-q7",
    "arm-correlate-fast-opt-q15",
    "arm-correlate-fast-q15",
    "arm-correlate-fast-q31",
    "arm-correlate-opt-q15",
    "arm-correlate-opt-q7",
}
STATEFUL_FILTER_CASES = {
    "arm-biquad-cascade-df1-fast-q15",
    "arm-biquad-cascade-df1-fast-q31",
    "arm-fir-decimate-fast-q15",
    "arm-fir-decimate-fast-q31",
    "arm-fir-fast-q15",
    "arm-fir-fast-q31",
    "arm-fir-lattice-f32",
    "arm-fir-lattice-q15",
    "arm-fir-lattice-q31",
    "arm-fir-sparse-f32",
    "arm-fir-sparse-q15",
    "arm-fir-sparse-q31",
    "arm-fir-sparse-q7",
    "arm-iir-lattice-f32",
    "arm-iir-lattice-q15",
    "arm-iir-lattice-q31",
}
MATRIX_MULTIPLICATION_CASES = {
    "arm-mat-mult-fast-q15",
    "arm-mat-mult-q7",
}
FLOATING_MATRIX_CASES = {
    "arm-mat-cmplx-mult-f16",
    "arm-mat-inverse-f16",
    "arm-mat-inverse-f32",
    "arm-mat-inverse-f64",
    "arm-mat-ldlt-f32",
    "arm-mat-ldlt-f64",
    "arm-mat-mult-f16",
    "arm-mat-mult-f32",
    "arm-mat-mult-f64",
    "arm-mat-qr-f16",
    "arm-mat-qr-f32",
    "arm-mat-qr-f64",
}
MATRIX_VECTOR_CASES = {
    "arm-mat-vec-mult-f16",
    "arm-mat-vec-mult-f32",
    "arm-mat-vec-mult-q15",
    "arm-mat-vec-mult-q31",
    "arm-mat-vec-mult-q7",
}
PID_CASES = {
    "arm-pid-f32",
    "arm-pid-q15",
    "arm-pid-q31",
}
LMS_CASES = {
    "arm-lms-f32",
    "arm-lms-norm-f32",
    "arm-lms-norm-q15",
    "arm-lms-norm-q31",
    "arm-lms-q15",
    "arm-lms-q31",
}
LEGACY_CFFT_CASES = {
    "arm-cfft-radix2-f16",
    "arm-cfft-radix2-f32",
    "arm-cfft-radix2-q15",
    "arm-cfft-radix2-q31",
    "arm-cfft-radix4-f16",
    "arm-cfft-radix4-f32",
    "arm-cfft-radix4-q15",
    "arm-cfft-radix4-q31",
}
RADIX8_F16_CASE = "arm-radix8-butterfly-f16"


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

    def test_convolution_protocols_have_one_typed_generated_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in CONVOLUTION_CORRELATION_CASES
        )

        self.assertEqual(len(workloads), len(CONVOLUTION_CORRELATION_CASES))
        for workload in workloads:
            with self.subTest(case=workload.case):
                self.assertIsInstance(
                    workload.producer,
                    corpus_inventory.CmsisDspGeneratedWorkloadProducer,
                )
                self.assertEqual(workload.producer.selector_kind, "filter-completion")
                self.assertTrue(
                    corpus_workload_provider.supports_cmsis_dsp_harness(workload)
                )

    def test_convolution_provider_keeps_oracle_outside_protocol(self) -> None:
        selected_cases = {
            "arm-conv-fast-q31",
            "arm-conv-opt-q7",
            "arm-correlate-fast-opt-q15",
            "arm-correlate-opt-q7",
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
            self.work / "convolution-harness",
        )
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
                baseline = (
                    symbol.replace("_fast_opt_", "_")
                    .replace("_fast_", "_")
                    .replace("_opt_", "_")
                )
                self.assertNotIn(f"{baseline}(", protocol)
                self.assertIn(f"{baseline}(", oracle)
                self.assertIn("output_matches_reference", oracle)
                if "opt" in symbol:
                    self.assertIn("scratch1", protocol)
                if symbol in {"arm_conv_opt_q7", "arm_correlate_opt_q7"}:
                    self.assertIn("scratch2", protocol)

    def test_stateful_filter_protocols_have_one_typed_generated_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in STATEFUL_FILTER_CASES
        )

        self.assertEqual(len(workloads), len(STATEFUL_FILTER_CASES))
        for workload in workloads:
            with self.subTest(case=workload.case):
                self.assertIsInstance(
                    workload.producer,
                    corpus_inventory.CmsisDspGeneratedWorkloadProducer,
                )
                self.assertEqual(workload.producer.selector_kind, "filter-completion")
                self.assertTrue(
                    corpus_workload_provider.supports_cmsis_dsp_harness(workload)
                )

    def test_stateful_filter_provider_keeps_protocol_atomic(self) -> None:
        selected_cases = {
            "arm-biquad-cascade-df1-fast-q15",
            "arm-fir-decimate-fast-q31",
            "arm-fir-lattice-f32",
            "arm-fir-sparse-q7",
            "arm-iir-lattice-q15",
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
            self.work / "stateful-filter-harness",
        )
        for workload in workloads:
            with self.subTest(case=workload.case):
                source_path, authoritative_owner = harness.protocol_source_owner(
                    workload.executable
                )
                self.assertEqual(authoritative_owner.name, "filtering_functions.h")
                source = source_path.read_text(encoding="utf-8")
                protocol, oracle = source.split("int main()", maxsplit=1)
                for call in workload.protocol:
                    self.assertEqual(protocol.count(f"{call.symbol}("), 1)
                    self.assertNotIn(f"{call.symbol}(", oracle)
                self.assertIn("output_matches_reference", oracle)

    def test_matrix_multiplication_protocols_have_one_typed_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in MATRIX_MULTIPLICATION_CASES
            and workload.producer.selector_kind == "benchmark-only"
        )

        self.assertEqual(len(workloads), len(MATRIX_MULTIPLICATION_CASES))
        for workload in workloads:
            with self.subTest(case=workload.case):
                self.assertTrue(
                    corpus_workload_provider.supports_cmsis_dsp_harness(workload)
                )

    def test_matrix_multiplication_oracle_is_outside_the_protocol(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in MATRIX_MULTIPLICATION_CASES
            and workload.producer.selector_kind == "benchmark-only"
        )
        self.assertEqual(len(workloads), len(MATRIX_MULTIPLICATION_CASES))

        harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "matrix-harness",
        )
        for workload in workloads:
            with self.subTest(case=workload.case):
                source_path, authoritative_owner = harness.protocol_source_owner(
                    workload.executable
                )
                self.assertEqual(authoritative_owner.name, "matrix_functions.h")
                source = source_path.read_text(encoding="utf-8")
                protocol, oracle = source.split("int main()", maxsplit=1)
                symbol = workload.protocol[0].symbol
                self.assertEqual(source.count(f"{symbol}("), 1)
                self.assertNotIn(f"{symbol}(", oracle)
                self.assertIn("kExpected", protocol)
                self.assertIn("output_matches_expected(output)", oracle)

    def test_floating_matrix_protocols_have_one_typed_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in FLOATING_MATRIX_CASES
        )
        self.assertEqual(len(workloads), len(FLOATING_MATRIX_CASES))

        for profile in {workload.target_profile for workload in workloads}:
            selected = tuple(
                workload for workload in workloads if workload.target_profile == profile
            )
            harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
                selected,
                corpus_inventory.resolve_externals_root(ROOT),
                self.work / f"floating-matrix-{profile}",
            )
            for workload in selected:
                with self.subTest(case=workload.case):
                    source_path, authoritative_owner = harness.protocol_source_owner(
                        workload.executable
                    )
                    self.assertIn(
                        authoritative_owner.name,
                        {"matrix_functions.h", "matrix_functions_f16.h"},
                    )
                    source = source_path.read_text(encoding="utf-8")
                    protocol, oracle = source.split("int main()", maxsplit=1)
                    symbol = workload.protocol[0].symbol
                    self.assertEqual(protocol.count(f"{symbol}("), 1)
                    self.assertNotIn(f"{symbol}(", oracle)
                    self.assertIn("output_matches_expected", oracle)

    def test_matrix_vector_protocols_use_direct_typed_harnesses(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in MATRIX_VECTOR_CASES
        )
        self.assertEqual({workload.case for workload in workloads}, MATRIX_VECTOR_CASES)

        for profile in {workload.target_profile for workload in workloads}:
            selected = tuple(
                workload for workload in workloads if workload.target_profile == profile
            )
            harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
                selected,
                corpus_inventory.resolve_externals_root(ROOT),
                self.work / f"matrix-vector-{profile}",
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
                        {"matrix_functions.h", "matrix_functions_f16.h"},
                    )
                    source = source_path.read_text(encoding="utf-8")
                    protocol, oracle = source.split("int main()", maxsplit=1)
                    symbol = workload.protocol[0].symbol
                    self.assertEqual(protocol.count(f"{symbol}("), 1)
                    self.assertNotIn(f"{symbol}(", oracle)
                    self.assertIn("output_matches_expected", oracle)

    def test_pid_protocols_have_one_typed_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in PID_CASES
            and workload.producer.selector_kind == "benchmark-only"
        )

        self.assertEqual(len(workloads), len(PID_CASES))
        for workload in workloads:
            with self.subTest(case=workload.case):
                self.assertTrue(
                    corpus_workload_provider.supports_cmsis_dsp_harness(workload)
                )

    def test_pid_provider_keeps_init_and_process_atomic(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in PID_CASES
            and workload.producer.selector_kind == "benchmark-only"
        )
        self.assertEqual(len(workloads), len(PID_CASES))

        harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "pid-harness",
        )
        for workload in workloads:
            with self.subTest(case=workload.case):
                source_path, authoritative_owner = harness.protocol_source_owner(
                    workload.executable
                )
                self.assertEqual(authoritative_owner.name, "controller_functions.h")
                source = source_path.read_text(encoding="utf-8")
                protocol, oracle = source.split("int main()", maxsplit=1)
                for call in workload.protocol:
                    self.assertEqual(protocol.count(f"{call.symbol}("), 1)
                    self.assertNotIn(f"{call.symbol}(", oracle)
                self.assertIn("output_matches_expected(output)", oracle)

    def test_lms_protocols_have_one_typed_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in LMS_CASES
            and workload.producer.selector_kind == "benchmark-only"
        )

        self.assertEqual(len(workloads), len(LMS_CASES))
        for workload in workloads:
            with self.subTest(case=workload.case):
                self.assertTrue(
                    corpus_workload_provider.supports_cmsis_dsp_harness(workload)
                )

    def test_lms_provider_keeps_adaptive_protocol_atomic(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in LMS_CASES
            and workload.producer.selector_kind == "benchmark-only"
        )
        self.assertEqual(len(workloads), len(LMS_CASES))

        harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "lms-harness",
        )
        for workload in workloads:
            with self.subTest(case=workload.case):
                source_path, authoritative_owner = harness.protocol_source_owner(
                    workload.executable
                )
                self.assertEqual(authoritative_owner.name, "filtering_functions.h")
                source = source_path.read_text(encoding="utf-8")
                protocol, oracle = source.split("int main()", maxsplit=1)
                for call in workload.protocol:
                    self.assertEqual(protocol.count(f"{call.symbol}("), 1)
                    self.assertNotIn(f"{call.symbol}(", oracle)
                self.assertIn("error_matches_reference", oracle)
                self.assertIn("coefficients_changed", oracle)

    def test_legacy_cfft_protocols_have_one_typed_owner(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in LEGACY_CFFT_CASES
            and workload.producer.selector_kind == "benchmark-only"
        )

        self.assertEqual(len(workloads), len(LEGACY_CFFT_CASES))
        for workload in workloads:
            with self.subTest(case=workload.case):
                self.assertTrue(
                    corpus_workload_provider.supports_cmsis_dsp_harness(workload)
                )

    def test_legacy_cfft_provider_keeps_transform_protocol_atomic(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp"
            and workload.case in LEGACY_CFFT_CASES
            and workload.producer.selector_kind == "benchmark-only"
        )
        self.assertEqual(len(workloads), len(LEGACY_CFFT_CASES))

        profiles = sorted({workload.target_profile for workload in workloads})
        for profile in profiles:
            selected = tuple(
                workload for workload in workloads if workload.target_profile == profile
            )
            harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
                selected,
                corpus_inventory.resolve_externals_root(ROOT),
                self.work / f"legacy-cfft-harness-{profile}",
            )
            for workload in selected:
                with self.subTest(case=workload.case):
                    source_path, authoritative_owner = harness.protocol_source_owner(
                        workload.executable
                    )
                    self.assertIn(
                        authoritative_owner.name,
                        {"transform_functions.h", "transform_functions_f16.h"},
                    )
                    source = source_path.read_text(encoding="utf-8")
                    protocol, oracle = source.split("int main()", maxsplit=1)
                    for call in workload.protocol:
                        self.assertEqual(protocol.count(f"{call.symbol}("), 1)
                        self.assertNotIn(f"{call.symbol}(", oracle)
                    self.assertIn("output_matches_independent_dft", oracle)

    def test_radix8_f16_generated_protocol_has_source_owned_oracle(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case == RADIX8_F16_CASE
        )
        self.assertEqual(len(workloads), 1)
        workload = workloads[0]
        self.assertTrue(corpus_workload_provider.supports_cmsis_dsp_harness(workload))

        harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
            workloads,
            corpus_inventory.resolve_externals_root(ROOT),
            self.work / "radix8-f16-harness",
        )
        source_path, authoritative_owner = harness.protocol_source_owner(
            workload.executable
        )
        self.assertEqual(authoritative_owner.name, "arm_cfft_radix8_f16.c")
        source = source_path.read_text(encoding="utf-8")
        protocol, oracle = source.split("int main()", maxsplit=1)
        call = workload.protocol[0]
        self.assertEqual(protocol.count(f"{call.symbol}("), 2)
        self.assertNotIn(f"{call.symbol}(", oracle)
        self.assertIn("output_matches_independent_dft", oracle)


if __name__ == "__main__":
    unittest.main()
