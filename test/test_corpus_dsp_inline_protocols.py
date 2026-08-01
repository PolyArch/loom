#!/usr/bin/env python3
"""Anchor tests for header-defined CMSIS-DSP operator protocols."""

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


INLINE_CASES = {
    "arm-div-int64-to-int32",
    "arm-norm-64-to-32u",
    "arm-recip-q15",
    "arm-recip-q31",
    "arm-sqrt-f16",
    "read-q15x2-da",
    "read-q15x2-ia",
    "read-q15x2",
    "read-q7x4-da",
    "read-q7x4-ia",
    "write-q15x2-ia",
    "write-q15x2",
    "write-q7x4-ia",
}


class CmsisDspInlineProtocolTests(unittest.TestCase):
    def test_header_defined_operators_use_one_out_of_line_protocol_root(self) -> None:
        workloads = tuple(
            workload
            for workload in corpus_inventory.load_workload_inventory(ROOT)
            if workload.suite == "cmsis-dsp" and workload.case in INLINE_CASES
        )
        self.assertEqual({workload.case for workload in workloads}, INLINE_CASES)

        external_root = corpus_inventory.resolve_externals_root(ROOT)
        with tempfile.TemporaryDirectory(
            prefix="loom-dsp-inline-", dir=ROOT / "temp"
        ) as directory:
            for ordinal, profile in enumerate(
                sorted({workload.target_profile for workload in workloads})
            ):
                selected = tuple(
                    workload
                    for workload in workloads
                    if workload.target_profile == profile
                )
                harness = corpus_workload_provider.materialize_cmsis_dsp_harness(
                    selected,
                    external_root,
                    Path(directory) / f"harness-{ordinal}",
                )
                for workload in selected:
                    with self.subTest(case=workload.case):
                        self.assertEqual(
                            harness.protocol_symbols(workload.executable),
                            ("loom_corpus_operator_protocol",),
                        )
                        self.assertEqual(
                            harness.expected_entry_result(workload.executable), 0
                        )
                        compiled_owner = harness.protocol_source(workload.executable)
                        self.assertEqual(compiled_owner.name, "OperatorProtocol.cpp")
                        source = compiled_owner.read_text(encoding="utf-8")
                        protocol = source.split("int main()", maxsplit=1)[0]
                        self.assertIn("LOOM_NOINLINE", protocol)
                        self.assertEqual(len(workload.protocol), 1)
                        self.assertIn(f"{workload.protocol[0].symbol}(", protocol)
                        self.assertNotIn("testmain", source)
                        self.assertNotIn("Testing::", source)


if __name__ == "__main__":
    unittest.main()
