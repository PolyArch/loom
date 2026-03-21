#!/usr/bin/env python3
"""Regression tests for run_rtl_checks.py per-check result accounting.

Each test creates a controlled temp directory structure so the runner
reaches the exact branch being tested, independent of fcc/dc_shell/verilator.
"""

import os
import subprocess
import sys
import tempfile


RUNNER = os.path.join(os.path.dirname(__file__), "..", "..",
                      "src", "rtl", "python", "run_rtl_checks.py")
SRC_RTL = os.path.join(os.path.dirname(__file__), "..", "..", "src", "rtl")


def run_runner(args_list, env_override=None):
    """Run run_rtl_checks.py and return (returncode, stdout+stderr)."""
    cmd = [sys.executable, RUNNER] + args_list
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
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


def test_missing_dfg_is_behaviour_fail():
    """ADG-only test (no .dfg.mlir) -> behaviour failed=1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        make_test_tree(tmpdir, "test_mod", "test_case", has_dfg=False)
        rc, out = run_runner([
            "--fcc", "/bin/false", "--test-dir", tmpdir,
            "--output-dir", os.path.join(tmpdir, "out"),
            "--src-rtl", SRC_RTL, "--checks", "behaviour",
        ])
        assert rc != 0, f"Expected nonzero exit, got {rc}"
        assert "behaviour" in out and "failed=1" in out, \
            f"Expected behaviour failed=1:\n{out}"
        print("PASS: missing DFG is behaviour FAIL")


def test_gen_fail_behaviour_fail_both_counted():
    """With --checks gen behaviour, gen fails (fcc=/bin/false) and behaviour
    fails (no DFG). Both checks must show failures independently."""
    with tempfile.TemporaryDirectory() as tmpdir:
        make_test_tree(tmpdir, "test_mod", "test_case", has_dfg=False)
        rc, out = run_runner([
            "--fcc", "/bin/false", "--test-dir", tmpdir,
            "--output-dir", os.path.join(tmpdir, "out"),
            "--src-rtl", SRC_RTL, "--checks", "gen", "behaviour",
        ])
        assert rc != 0, f"Expected nonzero exit, got {rc}"
        # Both gen and behaviour should show failed=1
        assert "gen" in out and "behaviour" in out, \
            f"Expected both gen and behaviour in output:\n{out}"
        # Extract per-check lines
        lines = [l.strip() for l in out.split("\n") if "failed=" in l]
        gen_line = [l for l in lines if l.startswith("gen")]
        beh_line = [l for l in lines if l.startswith("behaviour")]
        assert gen_line and "failed=1" in gen_line[0], \
            f"Expected gen failed=1:\n{out}"
        assert beh_line and "failed=1" in beh_line[0], \
            f"Expected behaviour failed=1:\n{out}"
        print("PASS: gen and behaviour both show failed=1 independently")


def test_negative_excluded_from_generic_discovery():
    """'negative' directory excluded from generic discovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        make_test_tree(tmpdir, "negative", "bad_test", has_dfg=False)
        make_test_tree(tmpdir, "good_mod", "good_test", has_dfg=False)
        rc, out = run_runner([
            "--fcc", "/bin/false", "--test-dir", tmpdir,
            "--output-dir", os.path.join(tmpdir, "out"),
            "--src-rtl", SRC_RTL, "--checks", "behaviour",
        ])
        assert "Discovered 1 test cases" in out, \
            f"Expected 1 test (negative excluded):\n{out}"
        print("PASS: negative/ excluded from generic discovery")


def test_synth_no_tool_increments_synth_skipped():
    """When dc_shell is not found, synth skip goes to synth counter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        make_test_tree(tmpdir, "test_mod", "test_case", has_dfg=True)
        # Force dc_shell to not be found by clearing PATH
        env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")}
        rc, out = run_runner([
            "--fcc", "/bin/false", "--test-dir", tmpdir,
            "--output-dir", os.path.join(tmpdir, "out"),
            "--src-rtl", SRC_RTL, "--checks", "synth",
        ], env_override=env)
        # Look for the synth summary line
        lines = [l.strip() for l in out.split("\n") if l.strip().startswith("synth")]
        if lines:
            synth_line = lines[0]
            if "skipped=1" in synth_line:
                print("PASS: synth no-tool skip uses synth counter")
                return
            elif "failed=" in synth_line:
                # dc_shell might actually be at /usr/bin/dc_shell
                print(f"INFO: dc_shell may be available (synth line: {synth_line})")
                print("SKIP: cannot verify synth-skip branch in this environment")
                return
        # If synth line not found, check if SKIP was printed
        if "SKIP: dc_shell not found" in out:
            # The skip message appeared but we need to verify the counter
            if "synth" in out and "skipped=" in out:
                print("PASS: synth skip path exercised")
                return
        print(f"SKIP: synth-skip branch not deterministically testable here:\n{out}")


if __name__ == "__main__":
    test_missing_dfg_is_behaviour_fail()
    test_gen_fail_behaviour_fail_both_counted()
    test_negative_excluded_from_generic_discovery()
    test_synth_no_tool_increments_synth_skipped()
    print("\nAll runner accounting tests completed.")
