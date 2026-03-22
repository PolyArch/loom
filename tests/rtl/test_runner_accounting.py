#!/usr/bin/env python3
"""Regression tests for run_rtl_checks.py per-check result accounting.

Tests use controlled temp directories and, where needed, monkeypatching
to deterministically exercise specific runner branches.
"""

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add the runner's parent directory to sys.path for importability
RUNNER_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                          "src", "rtl", "python")
sys.path.insert(0, RUNNER_DIR)

RUNNER_SCRIPT = os.path.join(RUNNER_DIR, "run_rtl_checks.py")
SRC_RTL = os.path.join(os.path.dirname(__file__), "..", "..", "src", "rtl")


def run_runner_subprocess(args_list):
    """Run run_rtl_checks.py as subprocess, return (rc, combined output)."""
    cmd = [sys.executable, RUNNER_SCRIPT] + args_list
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def make_test_tree(tmpdir, module_name, test_name, has_dfg=False):
    """Create a minimal test directory structure."""
    mod_dir = os.path.join(tmpdir, "fabric", module_name)
    os.makedirs(mod_dir, exist_ok=True)
    adg_path = os.path.join(mod_dir, f"{test_name}.fabric.mlir")
    with open(adg_path, "w") as f:
        f.write(f'fabric.module @{test_name}(%in0: !fabric.bits<32>) '
                '-> (!fabric.bits<32>) {\n'
                '  fabric.yield %in0 : !fabric.bits<32>\n}\n')
    if has_dfg:
        dfg_path = os.path.join(mod_dir, f"{test_name}.dfg.mlir")
        with open(dfg_path, "w") as f:
            f.write('module {\n  handshake.func @test(%a: i32) -> (i32) '
                    'attributes {argNames = ["a"], resNames = ["r"]} {\n'
                    '    handshake.return %a : i32\n  }\n}\n')


class TestRunnerAccounting(unittest.TestCase):

    def test_missing_dfg_is_behaviour_fail(self):
        """ADG-only test (no .dfg.mlir) -> behaviour failed=1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            make_test_tree(tmpdir, "test_mod", "test_case", has_dfg=False)
            rc, out = run_runner_subprocess([
                "--loom", "/bin/false", "--test-dir", tmpdir,
                "--output-dir", os.path.join(tmpdir, "out"),
                "--src-rtl", SRC_RTL, "--checks", "behaviour",
            ])
            self.assertNotEqual(rc, 0, "Expected nonzero exit")
            self.assertIn("failed=1", out, "Expected behaviour failed=1")

    def test_gen_fail_behaviour_fail_both_counted(self):
        """Gen fails + behaviour fails -> both show failed=1 independently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            make_test_tree(tmpdir, "test_mod", "test_case", has_dfg=False)
            rc, out = run_runner_subprocess([
                "--loom", "/bin/false", "--test-dir", tmpdir,
                "--output-dir", os.path.join(tmpdir, "out"),
                "--src-rtl", SRC_RTL, "--checks", "gen", "behaviour",
            ])
            self.assertNotEqual(rc, 0, "Expected nonzero exit")
            lines = [l.strip() for l in out.split("\n") if "failed=" in l]
            gen_lines = [l for l in lines if l.startswith("gen")]
            beh_lines = [l for l in lines if l.startswith("behaviour")]
            self.assertTrue(gen_lines, "Expected gen summary line")
            self.assertIn("failed=1", gen_lines[0])
            self.assertTrue(beh_lines, "Expected behaviour summary line")
            self.assertIn("failed=1", beh_lines[0])

    def test_negative_excluded_from_generic_discovery(self):
        """'negative' directory excluded from generic discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            make_test_tree(tmpdir, "negative", "bad_test", has_dfg=False)
            make_test_tree(tmpdir, "good_mod", "good_test", has_dfg=False)
            rc, out = run_runner_subprocess([
                "--loom", "/bin/false", "--test-dir", tmpdir,
                "--output-dir", os.path.join(tmpdir, "out"),
                "--src-rtl", SRC_RTL, "--checks", "behaviour",
            ])
            self.assertIn("Discovered 1 test cases", out)

    def test_synth_no_tool_increments_synth_skipped(self):
        """When find_tool returns False for dc_shell, synth skip goes to
        results['synth']['skipped'], not behaviour or gen.

        This test patches find_tool in-process and invokes the runner's
        main loop body directly to deterministically exercise the branch.
        """
        import run_rtl_checks as runner_mod
        import io
        from contextlib import redirect_stdout, redirect_stderr

        with tempfile.TemporaryDirectory() as tmpdir:
            make_test_tree(tmpdir, "test_mod", "test_case", has_dfg=True)

            # Patch find_tool to return False for dc_shell
            def mock_find_tool(name, module_spec=None):
                return False  # All tools unavailable

            # Patch sys.argv to simulate CLI args
            fake_argv = [
                "run_rtl_checks.py",
                "--loom", "/bin/false",
                "--test-dir", tmpdir,
                "--output-dir", os.path.join(tmpdir, "out"),
                "--src-rtl", SRC_RTL,
                "--checks", "synth",
            ]

            captured = io.StringIO()
            with patch.object(runner_mod, 'find_tool', side_effect=mock_find_tool), \
                 patch('sys.argv', fake_argv), \
                 redirect_stdout(captured), \
                 redirect_stderr(captured):
                try:
                    runner_mod.main()
                except SystemExit:
                    pass  # main() calls sys.exit()

            out = captured.getvalue()
            # Assert exact synth accounting
            lines = [l.strip() for l in out.split("\n")
                     if l.strip().startswith("synth")]
            self.assertTrue(lines, f"Expected synth summary line in:\n{out}")
            self.assertIn("passed=0", lines[0],
                          f"Expected synth passed=0:\n{lines[0]}")
            self.assertIn("failed=0", lines[0],
                          f"Expected synth failed=0:\n{lines[0]}")
            self.assertIn("skipped=1", lines[0],
                          f"Expected synth skipped=1:\n{lines[0]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
