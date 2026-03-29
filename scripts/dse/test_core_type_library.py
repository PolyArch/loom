"""Unit tests for the combinatorial KHG core type library.

Tests T3-T7 from the plan: naming bijectivity, FU counts, PE types,
SPM config, and array size.
"""

from __future__ import annotations

import re
import unittest

from .core_type_library import (
    ALL_TYPE_IDS,
    ArraySize,
    CombinatorialGenerator,
    ComputeMix,
    KHGTypeParams,
    PEKind,
    SPMPresence,
    decode_type_id,
    encode_type_id,
    is_valid_type_id,
    make_khg_params,
    params_from_type_id,
)


class TestNamingConvention(unittest.TestCase):
    """T3: Naming convention produces all 24 unique IDs."""

    def test_all_24_unique_ids(self):
        """Enumerate all combinations and verify 24 distinct strings."""
        ids = set()
        for c in ComputeMix:
            for p in PEKind:
                for s in SPMPresence:
                    for z in ArraySize:
                        tid = encode_type_id(c, p, s, z)
                        ids.add(tid)
        self.assertEqual(len(ids), 24)

    def test_all_match_regex(self):
        """Each string matches C[IFM][ST][YN](8|12)."""
        pattern = re.compile(r"^C[IFM][ST][YN](8|12)$")
        for tid in ALL_TYPE_IDS:
            self.assertRegex(tid, pattern,
                             f"Type ID '{tid}' does not match expected format")

    def test_decode_inverts_encode(self):
        """Decoder applied to each encoded string recovers the original tuple."""
        for c in ComputeMix:
            for p in PEKind:
                for s in SPMPresence:
                    for z in ArraySize:
                        tid = encode_type_id(c, p, s, z)
                        dc, dp, ds, dz = decode_type_id(tid)
                        self.assertEqual(dc, c)
                        self.assertEqual(dp, p)
                        self.assertEqual(ds, s)
                        self.assertEqual(dz, z)

    def test_no_two_tuples_produce_same_string(self):
        """No two distinct tuples produce the same string."""
        seen = {}
        for c in ComputeMix:
            for p in PEKind:
                for s in SPMPresence:
                    for z in ArraySize:
                        tid = encode_type_id(c, p, s, z)
                        key = (c, p, s, z)
                        self.assertNotIn(
                            tid, seen,
                            f"Duplicate type ID '{tid}' from {key} and {seen.get(tid)}"
                        )
                        seen[tid] = key

    def test_is_valid_type_id(self):
        """is_valid_type_id returns True for all 24 and False for junk."""
        for tid in ALL_TYPE_IDS:
            self.assertTrue(is_valid_type_id(tid))
        self.assertFalse(is_valid_type_id("BOGUS"))
        self.assertFalse(is_valid_type_id("CXSY8"))
        self.assertFalse(is_valid_type_id("CISY7"))
        self.assertFalse(is_valid_type_id(""))

    def test_decode_invalid_raises(self):
        """decode_type_id raises ValueError on invalid input."""
        with self.assertRaises(ValueError):
            decode_type_id("BOGUS")
        with self.assertRaises(ValueError):
            decode_type_id("")

    def test_canonical_list_length(self):
        """ALL_TYPE_IDS has exactly 24 entries."""
        self.assertEqual(len(ALL_TYPE_IDS), 24)
        self.assertEqual(len(set(ALL_TYPE_IDS)), 24)


class TestFUCounts(unittest.TestCase):
    """T4: FU counts match the compute-mix specification."""

    def test_int_heavy(self):
        """CISY8: 6 ALU, 4 MUL, 1 FP."""
        p = params_from_type_id("CISY8")
        self.assertEqual(p.fu_alu_count, 6)
        self.assertEqual(p.fu_mul_count, 4)
        self.assertEqual(p.fu_fp_count, 1)

    def test_fp_heavy(self):
        """CFSY8: 2 ALU, 2 MUL, 6 FP."""
        p = params_from_type_id("CFSY8")
        self.assertEqual(p.fu_alu_count, 2)
        self.assertEqual(p.fu_mul_count, 2)
        self.assertEqual(p.fu_fp_count, 6)

    def test_mixed(self):
        """CMSY8: 4 ALU, 3 MUL, 3 FP."""
        p = params_from_type_id("CMSY8")
        self.assertEqual(p.fu_alu_count, 4)
        self.assertEqual(p.fu_mul_count, 3)
        self.assertEqual(p.fu_fp_count, 3)

    def test_compute_mix_invariant_across_pe_spm_size(self):
        """FU counts depend only on compute mix, not PE/SPM/size."""
        for c_char, expected in [
            ("I", (6, 4, 1)),
            ("F", (2, 2, 6)),
            ("M", (4, 3, 3)),
        ]:
            for suffix in ["SY8", "SY12", "SN8", "SN12",
                           "TY8", "TY12", "TN8", "TN12"]:
                tid = f"C{c_char}{suffix}"
                p = params_from_type_id(tid)
                self.assertEqual(
                    (p.fu_alu_count, p.fu_mul_count, p.fu_fp_count),
                    expected,
                    f"FU mismatch for {tid}"
                )


class TestPEType(unittest.TestCase):
    """T5: PE type dimension produces correct PE and switch types."""

    def test_spatial(self):
        """CMSY8: spatial PE, no instruction slots or registers."""
        p = params_from_type_id("CMSY8")
        self.assertEqual(p.pe_kind, PEKind.SPATIAL)
        self.assertFalse(p.is_temporal)
        self.assertEqual(p.instruction_slots, 0)
        self.assertEqual(p.num_registers, 0)

    def test_temporal(self):
        """CMTY8: temporal PE with instruction_slots=8, num_registers=8."""
        p = params_from_type_id("CMTY8")
        self.assertEqual(p.pe_kind, PEKind.TEMPORAL)
        self.assertTrue(p.is_temporal)
        self.assertEqual(p.instruction_slots, 8)
        self.assertEqual(p.num_registers, 8)


class TestSPMDimension(unittest.TestCase):
    """T6: SPM dimension controls memory presence."""

    def test_with_spm(self):
        """CMSY8: spm_size_kb=16, ld_ports=2, st_ports=2."""
        p = params_from_type_id("CMSY8")
        self.assertTrue(p.has_spm)
        self.assertEqual(p.spm_size_kb, 16)
        self.assertEqual(p.spm_ld_ports, 2)
        self.assertEqual(p.spm_st_ports, 2)

    def test_without_spm(self):
        """CMSN8: spm_size_kb=0, ld_ports=0, st_ports=0."""
        p = params_from_type_id("CMSN8")
        self.assertFalse(p.has_spm)
        self.assertEqual(p.spm_size_kb, 0)
        self.assertEqual(p.spm_ld_ports, 0)
        self.assertEqual(p.spm_st_ports, 0)


class TestArraySize(unittest.TestCase):
    """T7: Size dimension controls array dimensions."""

    def test_8x8(self):
        """CMSY8: 8x8 = 64 PEs."""
        p = params_from_type_id("CMSY8")
        self.assertEqual(p.array_rows, 8)
        self.assertEqual(p.array_cols, 8)
        self.assertEqual(p.total_pes, 64)

    def test_12x12(self):
        """CMSY12: 12x12 = 144 PEs."""
        p = params_from_type_id("CMSY12")
        self.assertEqual(p.array_rows, 12)
        self.assertEqual(p.array_cols, 12)
        self.assertEqual(p.total_pes, 144)


class TestCombinatorialGenerator(unittest.TestCase):
    """Integration tests for the CombinatorialGenerator class."""

    def setUp(self):
        self.gen = CombinatorialGenerator()

    def test_count(self):
        self.assertEqual(self.gen.count, 24)
        self.assertEqual(len(self.gen), 24)

    def test_iteration(self):
        """Iterating yields exactly 24 KHGTypeParams."""
        items = list(self.gen)
        self.assertEqual(len(items), 24)
        for item in items:
            self.assertIsInstance(item, KHGTypeParams)

    def test_get_valid(self):
        p = self.gen.get("CISY8")
        self.assertEqual(p.type_id, "CISY8")

    def test_get_invalid(self):
        with self.assertRaises(KeyError):
            self.gen.get("BOGUS")

    def test_filter_by_compute(self):
        """Filter by INT_HEAVY yields 8 types."""
        results = self.gen.filter_by(compute=ComputeMix.INT_HEAVY)
        self.assertEqual(len(results), 8)
        for p in results:
            self.assertEqual(p.compute_mix, ComputeMix.INT_HEAVY)

    def test_filter_by_pe(self):
        """Filter by TEMPORAL yields 12 types."""
        results = self.gen.filter_by(pe=PEKind.TEMPORAL)
        self.assertEqual(len(results), 12)

    def test_filter_by_spm(self):
        """Filter by WITH_SPM yields 12 types."""
        results = self.gen.filter_by(spm=SPMPresence.WITH_SPM)
        self.assertEqual(len(results), 12)

    def test_filter_by_size(self):
        """Filter by SIZE_8 yields 12 types."""
        results = self.gen.filter_by(size=ArraySize.SIZE_8)
        self.assertEqual(len(results), 12)

    def test_filter_combined(self):
        """Combined filter narrows results."""
        results = self.gen.filter_by(
            compute=ComputeMix.MIXED,
            pe=PEKind.SPATIAL,
        )
        self.assertEqual(len(results), 4)
        for p in results:
            self.assertEqual(p.compute_mix, ComputeMix.MIXED)
            self.assertEqual(p.pe_kind, PEKind.SPATIAL)

    def test_all_type_ids_property(self):
        self.assertEqual(self.gen.type_ids, ALL_TYPE_IDS)

    def test_to_design_space_configs(self):
        """Export produces 24 dicts with expected keys."""
        configs = self.gen.to_design_space_configs()
        self.assertEqual(len(configs), 24)
        expected_keys = {
            "type_id", "pe_grid_rows", "pe_grid_cols",
            "fu_alu_count", "fu_mul_count", "fu_fp_count",
            "spm_size_kb", "pe_kind", "is_temporal",
            "instruction_slots", "num_registers",
        }
        for cfg in configs:
            self.assertEqual(set(cfg.keys()), expected_keys)

    def test_data_width_default(self):
        """All types default to 32-bit data width."""
        for p in self.gen:
            self.assertEqual(p.data_width, 32,
                             f"{p.type_id} should have data_width=32")


if __name__ == "__main__":
    unittest.main()
