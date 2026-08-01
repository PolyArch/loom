#!/usr/bin/env python3
"""Anchor tests for exact corpus target-profile disposition."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))

import corpus_inventory  # noqa: E402
import corpus_target_profile  # noqa: E402


class CorpusTargetProfileTest(unittest.TestCase):
    def test_exact_riscv_cohort_distinguishes_incompatible_isa(self) -> None:
        portable = corpus_target_profile.resolve_target_profile(
            "cmsis-nn",
            corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
            "riscv64-unknown-elf",
        )
        float16 = corpus_target_profile.resolve_target_profile(
            "cmsis-dsp",
            corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE,
            "riscv64-unknown-elf",
        )
        mve = corpus_target_profile.resolve_target_profile(
            "cmsis-nn", "mve", "riscv64-unknown-elf"
        )
        unknown = corpus_target_profile.resolve_target_profile(
            "cmsis-nn", "future-profile", "riscv64-unknown-elf"
        )

        self.assertEqual(
            portable.disposition,
            corpus_target_profile.TargetProfileDisposition.RUNNABLE,
        )
        self.assertEqual(portable.compile_flags, ())
        self.assertEqual(
            float16.disposition,
            corpus_target_profile.TargetProfileDisposition.RUNNABLE,
        )
        self.assertEqual(
            float16.compile_flags,
            ("-D__ARM_FP16_FORMAT_IEEE=1", "-D__fp16=_Float16"),
        )
        self.assertEqual(
            mve.disposition,
            corpus_target_profile.TargetProfileDisposition.INCOMPATIBLE_ISA,
        )
        self.assertIn("requires arm", mve.detail)
        self.assertIn("riscv64", mve.detail)
        self.assertEqual(
            unknown.disposition,
            corpus_target_profile.TargetProfileDisposition.PROVIDER_UNAVAILABLE,
        )

    def test_representative_manifest_has_a_total_profile_disposition(self) -> None:
        workloads = corpus_inventory.load_workload_inventory(ROOT)
        counts = Counter(
            corpus_target_profile.resolve_target_profile(
                workload.suite,
                workload.target_profile,
                "riscv64-unknown-elf",
            ).disposition
            for workload in workloads
        )

        self.assertEqual(len(workloads), 889)
        self.assertEqual(
            counts,
            {
                corpus_target_profile.TargetProfileDisposition.RUNNABLE: 842,
                corpus_target_profile.TargetProfileDisposition.INCOMPATIBLE_ISA: 47,
            },
        )


if __name__ == "__main__":
    unittest.main()
