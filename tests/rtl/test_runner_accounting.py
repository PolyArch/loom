#!/usr/bin/env python3
"""Regression tests for run_rtl_checks.py per-check result accounting.

Each test creates a controlled temp directory structure so the runner
reaches the exact branch being tested, independent of fcc availability.
"""

import os
import subprocess
import sys
import tempfile


RUNNER = os.path.join(os.path.dirname(__file__), "..", "..",
                      "src", "rtl", "python", "run_rtl_checks.py")
SRC_RTL = os.path.join(os.path.dirname(__file__), "..", "..", "src", "rtl")


def run_runner(args_list):
    """Run run_rtl_checks.py and return (returncode, stdout)."""
    cmd = [sys.executable, RUNNER] + args_list
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def make_test_tree(tmpdir, module_name, test_name, has_dfg=False):
    """Create a minimal test directory structure."""
    mod_dir = os.path.join(tmpdir, "fabric", module_name)
    os.makedirs(mod_dir, exist_ok=True)
    # Create a minimal .fabric.mlir
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


def test_missing_dfg_is_behaviour_fail():
    """ADG-only test (no .dfg.mlir) should report behaviour failed=1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        make_test_tree(tmpdir, "test_mod", "test_case", has_dfg=False)
        rc, out = run_runner([
            "--fcc", "/bin/false",
            "--test-dir", tmpdir,
            "--output-dir", os.path.join(tmpdir, "out"),
            "--src-rtl", SRC_RTL,
            "--checks", "behaviour",
        ])
        assert rc != 0, f"Expected nonzero exit, got {rc}"
        assert "behaviour" in out and "failed=1" in out, \
            f"Expected behaviour failed=1 in output:\n{out}"
        print("PASS: missing DFG is behaviour FAIL")


def test_gen_pass_behaviour_fail_exits_nonzero():
    """With --checks gen behaviour, gen may fail (fcc=/bin/false) but
    behaviour must independently fail for missing DFG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        make_test_tree(tmpdir, "test_mod", "test_case", has_dfg=False)
        rc, out = run_runner([
            "--fcc", "/bin/false",
            "--test-dir", tmpdir,
            "--output-dir", os.path.join(tmpdir, "out"),
            "--src-rtl", SRC_RTL,
            "--checks", "gen", "behaviour",
        ])
        assert rc != 0, f"Expected nonzero exit, got {rc}"
        # Both gen and behaviour should show failures
        assert "behaviour" in out, f"Expected behaviour in output:\n{out}"
        print("PASS: gen+behaviour combined exits nonzero")


def test_negative_excluded_from_generic_discovery():
    """The 'negative' directory should be excluded from generic discovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test under "negative" and one under "positive"
        make_test_tree(tmpdir, "negative", "bad_test", has_dfg=False)
        make_test_tree(tmpdir, "good_mod", "good_test", has_dfg=False)
        rc, out = run_runner([
            "--fcc", "/bin/false",
            "--test-dir", tmpdir,
            "--output-dir", os.path.join(tmpdir, "out"),
            "--src-rtl", SRC_RTL,
            "--checks", "behaviour",
        ])
        assert "Discovered 1 test cases" in out, \
            f"Expected 1 test case (negative excluded), got:\n{out}"
        assert "negative" not in out.split("Discovered")[0], \
            f"negative/ should not appear in discovery:\n{out}"
        print("PASS: negative/ excluded from generic discovery")


if __name__ == "__main__":
    test_missing_dfg_is_behaviour_fail()
    test_gen_pass_behaviour_fail_exits_nonzero()
    test_negative_excluded_from_generic_discovery()
    print("\nAll runner accounting tests passed.")
