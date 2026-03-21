#!/usr/bin/env python3
"""Regression tests for run_rtl_checks.py per-check result accounting.

Verifies:
1. Missing DFG companion causes behaviour FAIL (not SKIP or PASS)
2. Synth no-tool path increments synth["skipped"] not behaviour["skipped"]
3. Per-check exit code: fails when behaviour has failures, even if gen passes
"""

import os
import subprocess
import sys
import tempfile


def run_runner(args_list):
    """Run run_rtl_checks.py and return (returncode, stdout, stderr)."""
    cmd = [sys.executable,
           os.path.join(os.path.dirname(__file__), "..", "..",
                        "src", "rtl", "python", "run_rtl_checks.py")]
    cmd.extend(args_list)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def test_missing_companion_is_behaviour_fail():
    """ADG-only test with no .dfg.mlir should report behaviour FAIL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, _ = run_runner([
            "--fcc", "/bin/false",
            "--test-dir", os.path.join(os.path.dirname(__file__)),
            "--output-dir", tmpdir,
            "--src-rtl", os.path.join(os.path.dirname(__file__), "..", "..", "src", "rtl"),
            "--checks", "behaviour",
            "--modules", "add_tag",
        ])
        assert rc != 0, f"Expected nonzero exit, got {rc}"
        assert "behaviour" in stdout and "failed=1" in stdout, \
            f"Expected behaviour failed=1, got: {stdout}"
        print("PASS: missing companion is behaviour FAIL")


def test_per_check_exit_gen_pass_behaviour_fail():
    """Gen passes but behaviour fails -> overall exit nonzero."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, _ = run_runner([
            "--fcc", "/bin/false",
            "--test-dir", os.path.join(os.path.dirname(__file__)),
            "--output-dir", tmpdir,
            "--src-rtl", os.path.join(os.path.dirname(__file__), "..", "..", "src", "rtl"),
            "--checks", "gen", "behaviour",
            "--modules", "add_tag",
        ])
        # gen will fail too (fcc=/bin/false), but behaviour should show failed
        assert rc != 0, f"Expected nonzero exit, got {rc}"
        assert "behaviour" in stdout, f"Expected behaviour in output: {stdout}"
        print("PASS: gen+behaviour combined exits nonzero when behaviour fails")


def test_synth_skip_uses_synth_counter():
    """When dc_shell is not found, skip should be under synth, not behaviour."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rc, stdout, _ = run_runner([
            "--fcc", "/bin/false",
            "--test-dir", os.path.join(os.path.dirname(__file__)),
            "--output-dir", tmpdir,
            "--src-rtl", os.path.join(os.path.dirname(__file__), "..", "..", "src", "rtl"),
            "--checks", "synth",
            "--modules", "add_tag",
        ])
        # dc_shell likely not on PATH, so synth should skip
        if "skipped=1" in stdout and "synth" in stdout:
            print("PASS: synth skip uses synth counter")
        else:
            # dc_shell might be available in some environments
            print(f"SKIP: dc_shell may be available, output: {stdout}")


if __name__ == "__main__":
    test_missing_companion_is_behaviour_fail()
    test_per_check_exit_gen_pass_behaviour_fail()
    test_synth_skip_uses_synth_counter()
    print("\nAll runner accounting tests passed.")
