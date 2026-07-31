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


if __name__ == "__main__":
    unittest.main()
