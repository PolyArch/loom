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
                compiled_owner, authoritative_owner = harness.protocol_source_owner(
                    workload.executable
                )
                self.assertEqual(compiled_owner.name, "OperatorProtocol.cpp")
                self.assertIn("filtering_functions", authoritative_owner.name)
                source = compiled_owner.read_text(encoding="utf-8")
                protocol = source.split("int main()", maxsplit=1)[0]
                suffix = workload.case.removeprefix("arm-fir-")
                self.assertEqual(protocol.count(f"arm_fir_init_{suffix}("), 1)
                self.assertEqual(protocol.count(f"arm_fir_{suffix}("), 2)
                self.assertIn("kExpected", source)
                self.assertIn("return oracle_matches(output) ? 0 : 1;", source)
                if workload.case.endswith("q15"):
                    self.assertIn("(void)arm_fir_init_q15", source)


if __name__ == "__main__":
    unittest.main()
