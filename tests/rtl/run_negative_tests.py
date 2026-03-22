#!/usr/bin/env python3
"""Negative test runner: verify that fcc --gen-sv rejects invalid MLIR inputs.

Each test file is expected to cause a non-zero exit from fcc --gen-sv.
The runner also checks that stderr contains a recognizable error message
(not a crash or unexpected failure mode).
"""

import argparse
import os
import subprocess
import sys

# Expected error substrings for each negative test case, keyed by
# the stem of the test file name (without .fabric.mlir).
EXPECTED_ERRORS = {
    "latency_too_low": "latency",
    "transcendental_no_profile": "fp-ip-profile",
    "dataflow_wrong_latency": "latency",
    "interval_too_low": "interval",
}


def run_negative_test(fcc_exec, test_file, output_dir):
    """Run fcc --gen-sv on a test file and verify rejection.

    Returns (passed: bool, message: str).
    """
    test_name = os.path.basename(test_file).replace(".fabric.mlir", "")
    test_output = os.path.join(output_dir, test_name)
    os.makedirs(test_output, exist_ok=True)

    cmd = [fcc_exec, "--gen-sv", "--adg", test_file, "-o", test_output]
    result = subprocess.run(cmd, capture_output=True, text=True)

    combined_output = (result.stdout or "") + (result.stderr or "")

    if result.returncode == 0:
        return False, f"expected non-zero exit but got 0"

    # Verify the error message contains the expected substring
    expected = EXPECTED_ERRORS.get(test_name)
    if expected:
        if expected.lower() not in combined_output.lower():
            return False, (f"exited with code {result.returncode} but error "
                           f"output does not contain '{expected}'")

    return True, f"correctly rejected (exit code {result.returncode})"


def main():
    parser = argparse.ArgumentParser(
        description="Negative test runner for fcc --gen-sv")
    parser.add_argument("--fcc", required=True, help="Path to fcc executable")
    parser.add_argument("--test-files", nargs="+", required=True,
                        help="Paths to negative test .fabric.mlir files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for test artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    passed = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"Running {len(args.test_files)} negative tests")
    print(f"{'='*60}")

    for test_file in sorted(args.test_files):
        test_name = os.path.basename(test_file).replace(".fabric.mlir", "")
        ok, msg = run_negative_test(args.fcc, test_file, args.output_dir)
        status = "PASS" if ok else "FAIL"
        print(f"[negative/{test_name}] {status}: {msg}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Negative Test Summary")
    print(f"{'='*60}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
