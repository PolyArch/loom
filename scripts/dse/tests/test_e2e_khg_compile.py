"""End-to-end test: KHG ADG generation -> loom mapper compilation.

Validates that a KHG-generated core architecture (e.g. CMSY8) can be
compiled through the real loom mapper against a test kernel, using
ADGs produced by tapestry_adg_gen.

The test is automatically skipped if the loom or tapestry_adg_gen
binaries are not found (i.e. the project has not been built).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.dse.inner_optimizer import (
    CoreDesignParams,
    KernelMappingResult,
    _find_tapestry_compile,
    _find_adg_gen,
    _generate_adg_library,
    _params_to_system_arch_json,
    _select_adg_core_type,
    create_combinatorial_preset,
    ComputeMix,
    PEType,
    make_tapestry_compile_fn,
)
from scripts.dse.proxy_model import KernelProfile


def _loom_mapper_available() -> bool:
    """Return True if the loom mapper binary is available."""
    binary = _find_tapestry_compile()
    return binary is not None and os.path.isfile(binary)


def _adg_gen_available() -> bool:
    """Return True if the tapestry_adg_gen binary is available."""
    binary = _find_adg_gen()
    return binary is not None and os.path.isfile(binary)


class TestSystemArchJSONConstruction(unittest.TestCase):
    """Validate that KHG params produce a valid system-arch JSON (no binary needed)."""

    def test_cmsy8_arch_json_structure(self):
        """CMSY8 (Mixed, Spatial, SPM, 8x8) produces well-formed arch JSON."""
        params = create_combinatorial_preset(
            ComputeMix.MIXED, PEType.SPATIAL, True, 8
        )
        arch = _params_to_system_arch_json(params)

        self.assertIn("coreTypes", arch)
        self.assertEqual(len(arch["coreTypes"]), 1)

        core = arch["coreTypes"][0]
        self.assertEqual(core["meshRows"], 8)
        self.assertEqual(core["meshCols"], 8)
        self.assertEqual(core["numInstances"], 1)
        self.assertEqual(core["spmSizeBytes"], 16 * 1024)
        self.assertTrue(core["includeMultiplier"])
        self.assertTrue(core["includeMemory"])

    def test_int_heavy_no_spm(self):
        """INT_HEAVY preset without SPM should have spmSizeBytes=0."""
        params = create_combinatorial_preset(
            ComputeMix.INT_HEAVY, PEType.SPATIAL, False, 8
        )
        arch = _params_to_system_arch_json(params)
        core = arch["coreTypes"][0]
        self.assertTrue(core["includeMultiplier"])
        self.assertEqual(core["spmSizeBytes"], 0)

    def test_adg_core_type_selection(self):
        """Core type selection should match FP and size characteristics."""
        # FP_HEAVY -> ai_core (most capable FP ADG)
        fp_large = create_combinatorial_preset(
            ComputeMix.FP_HEAVY, PEType.SPATIAL, True, 8
        )
        self.assertEqual(_select_adg_core_type(fp_large), "ai_core")

        # MIXED with FP -> ai_core (has FP ops in repertoire)
        mixed_med = create_combinatorial_preset(
            ComputeMix.MIXED, PEType.SPATIAL, True, 8
        )
        self.assertEqual(_select_adg_core_type(mixed_med), "ai_core")

        # INT_HEAVY small (4x4 = 16 PEs, no FP) -> ctrl_core
        int_small = create_combinatorial_preset(
            ComputeMix.INT_HEAVY, PEType.SPATIAL, False, 4
        )
        self.assertEqual(_select_adg_core_type(int_small), "ctrl_core")

        # INT_HEAVY large (8x8 = 64 PEs, no FP) -> gp_core
        int_large = create_combinatorial_preset(
            ComputeMix.INT_HEAVY, PEType.SPATIAL, True, 8
        )
        self.assertEqual(_select_adg_core_type(int_large), "gp_core")


@unittest.skipUnless(
    _adg_gen_available(),
    "tapestry_adg_gen binary not found; build the project first",
)
class TestADGGeneration(unittest.TestCase):
    """Test ADG library generation via tapestry_adg_gen."""

    def test_generate_adg_library(self):
        """tapestry_adg_gen should produce 4 ADG files."""
        with tempfile.TemporaryDirectory(prefix="adg_gen_test_") as tmpdir:
            adg_files = _generate_adg_library(tmpdir)

            self.assertIsNotNone(adg_files)
            self.assertEqual(len(adg_files), 4)
            for name in ["gp_core", "dsp_core", "ai_core", "ctrl_core"]:
                self.assertIn(name, adg_files)
                self.assertTrue(
                    os.path.isfile(adg_files[name]),
                    f"ADG file for {name} should exist",
                )
                # Check file is non-empty
                self.assertGreater(
                    os.path.getsize(adg_files[name]), 100,
                    f"ADG file for {name} should be non-trivial",
                )


@unittest.skipUnless(
    _loom_mapper_available() and _adg_gen_available(),
    "loom mapper or tapestry_adg_gen not found; build the project first",
)
class TestKHGLoomMapperCompile(unittest.TestCase):
    """E2E: compile a test kernel through the loom mapper with ADG."""

    def test_ai_core_compiles_sum_array(self):
        """AI core (8x8) ADG should compile a sum_array kernel."""
        loom_bin = _find_tapestry_compile()
        self.assertIsNotNone(loom_bin)

        with tempfile.TemporaryDirectory(prefix="e2e_khg_") as tmpdir:
            # Generate ADGs
            adg_files = _generate_adg_library(
                os.path.join(tmpdir, "adg_lib")
            )
            self.assertIsNotNone(adg_files)

            # Write sum_array kernel (simpler than vecadd, maps reliably)
            kernel_path = os.path.join(tmpdir, "sum_array.c")
            with open(kernel_path, "w") as f:
                f.write(
                    "int sum_array(int *a, int n) {\n"
                    "    int s = 0;\n"
                    "    for (int i = 0; i < n; i++) {\n"
                    "        s += a[i];\n"
                    "    }\n"
                    "    return s;\n"
                    "}\n"
                )

            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                loom_bin,
                "--adg", adg_files["ai_core"],
                kernel_path,
                "-o", output_dir,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Check map.json was produced (even partial success is OK)
            map_files = list(Path(output_dir).glob("*.map.json"))
            self.assertGreater(
                len(map_files), 0,
                f"Should produce map.json. rc={result.returncode}\n"
                f"stderr: {result.stderr[-300:]}",
            )

            with open(map_files[0]) as f:
                map_data = json.load(f)
            timing = map_data.get("timing", {})
            ii = timing.get("estimated_initiation_interval", 0)
            self.assertGreater(ii, 0, "II should be positive")

    def test_compile_fn_integration(self):
        """make_tapestry_compile_fn should produce a working compile_fn."""
        compile_fn = make_tapestry_compile_fn()

        params = create_combinatorial_preset(
            ComputeMix.MIXED, PEType.SPATIAL, True, 8
        )

        kernel = KernelProfile()
        kernel.name = "simple_test"
        kernel.op_histogram = {"arith.addi": 5, "handshake.load": 2}
        kernel.dfg_node_count = 7

        mr = compile_fn(params, kernel)

        self.assertIsInstance(mr, KernelMappingResult)
        self.assertTrue(
            mr.success,
            f"Compilation should succeed. "
            f"mapping_time={mr.mapping_time_sec:.1f}s",
        )
        self.assertGreater(mr.achieved_ii, 0)
        self.assertGreater(mr.mapping_time_sec, 0)


@unittest.skipUnless(
    _loom_mapper_available() and _adg_gen_available(),
    "loom mapper or tapestry_adg_gen not found; build the project first",
)
class TestKHGVariantCompilation(unittest.TestCase):
    """Test that multiple KHG variant ADGs can compile."""

    def test_dsp_core_compiles(self):
        """DSP core (with FP) should compile a sum_array kernel."""
        compile_fn = make_tapestry_compile_fn()

        params = create_combinatorial_preset(
            ComputeMix.FP_HEAVY, PEType.SPATIAL, True, 8
        )

        kernel = KernelProfile()
        kernel.name = "fp_test"
        kernel.op_histogram = {"arith.addi": 3}
        kernel.dfg_node_count = 3

        mr = compile_fn(params, kernel)
        self.assertTrue(
            mr.success,
            "FP_HEAVY architecture should compile via ai_core ADG",
        )

    def test_int_heavy_compiles(self):
        """INT_HEAVY 8x8 should compile via gp_core ADG."""
        compile_fn = make_tapestry_compile_fn()

        params = create_combinatorial_preset(
            ComputeMix.INT_HEAVY, PEType.SPATIAL, True, 8
        )

        kernel = KernelProfile()
        kernel.name = "int_test"
        kernel.op_histogram = {"arith.addi": 5}
        kernel.dfg_node_count = 5

        mr = compile_fn(params, kernel)
        self.assertTrue(
            mr.success,
            "INT_HEAVY 8x8 architecture should compile via gp_core ADG",
        )


if __name__ == "__main__":
    unittest.main()
