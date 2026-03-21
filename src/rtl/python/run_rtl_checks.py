#!/usr/bin/env python3
"""Master RTL verification runner. Dispatches to gen_sv, run_sim, run_synth."""

import argparse
import os
import subprocess
import sys
import json


def find_tool(name, module_spec=None):
    """Check if a tool is available, optionally via module loading."""
    import shutil
    if shutil.which(name):
        return True
    if module_spec:
        try:
            result = subprocess.run(
                ["bash", "-c",
                 f"source /etc/profile.d/modules.sh && module load {module_spec} && which {name}"],
                capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            pass
    return False


def run_check(script, args_list, check_name, output_dir):
    """Run a check script and capture results."""
    log_path = os.path.join(output_dir, f"{check_name}.log")
    cmd = [sys.executable, script] + args_list

    print(f"\n{'='*60}")
    print(f"Running {check_name}: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    with open(log_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    passed = result.returncode == 0
    status = "PASS" if passed else "FAIL"
    print(f"[{check_name}] {status}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Master RTL verification runner")
    parser.add_argument("--fcc", required=True, help="Path to fcc executable")
    parser.add_argument("--test-dir", required=True, help="tests/rtl/ directory")
    parser.add_argument("--output-dir", required=True, help="Output base directory")
    parser.add_argument("--src-rtl", required=True, help="src/rtl/ directory")
    parser.add_argument("--checks", nargs="+", default=["gen", "behaviour", "synth"],
                        choices=["gen", "behaviour", "synth"])
    args = parser.parse_args()

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    results = {"passed": 0, "failed": 0, "skipped": 0}

    # Discover test cases
    fabric_dir = os.path.join(args.test_dir, "fabric")
    if not os.path.isdir(fabric_dir):
        print(f"ERROR: test directory not found: {fabric_dir}")
        sys.exit(1)

    test_cases = []
    for module_name in sorted(os.listdir(fabric_dir)):
        module_dir = os.path.join(fabric_dir, module_name)
        if not os.path.isdir(module_dir):
            continue
        for test_file in sorted(os.listdir(module_dir)):
            if test_file.endswith(".fabric.mlir"):
                test_name = test_file.replace(".fabric.mlir", "")
                test_cases.append({
                    "module": module_name,
                    "test": test_name,
                    "mlir": os.path.join(module_dir, test_file),
                })

    print(f"Discovered {len(test_cases)} test cases")

    for tc in test_cases:
        test_output = os.path.join(args.output_dir, "rtl", "fabric",
                                    tc["module"], tc["test"])
        os.makedirs(test_output, exist_ok=True)

        # Check 2: Generation
        if "gen" in args.checks:
            gen_dir = os.path.join(test_output, "gen-collateral")
            os.makedirs(gen_dir, exist_ok=True)
            passed = run_check(
                os.path.join(scripts_dir, "gen_sv.py"),
                ["--fcc", args.fcc, "--adg", tc["mlir"], "--output-dir", gen_dir],
                f"gen/{tc['module']}/{tc['test']}",
                gen_dir
            )
            results["passed" if passed else "failed"] += 1

        # Check 1: Functional (requires golden traces)
        if "behaviour" in args.checks:
            beh_dir = os.path.join(test_output, "behaviour")
            os.makedirs(beh_dir, exist_ok=True)
            if not find_tool("verilator"):
                print(f"[behaviour/{tc['module']}/{tc['test']}] SKIP: verilator not found")
                results["skipped"] += 1
            else:
                # TODO: generate golden traces first via fcc --simulate --trace-port-dump
                results["skipped"] += 1

        # Check 3: Synthesis
        if "synth" in args.checks:
            phys_dir = os.path.join(test_output, "physical")
            os.makedirs(phys_dir, exist_ok=True)
            if not find_tool("dc_shell"):
                print(f"[synth/{tc['module']}/{tc['test']}] SKIP: dc_shell not found")
                results["skipped"] += 1
            else:
                tcl_template = os.path.join(args.src_rtl, "tcl", "synth_template.tcl")
                gen_dir = os.path.join(test_output, "gen-collateral", "rtl")
                if os.path.isdir(gen_dir):
                    passed = run_check(
                        os.path.join(scripts_dir, "run_synth.py"),
                        ["--rtl-dir", gen_dir, "--design-name", tc["module"],
                         "--output-dir", phys_dir, "--tcl-template", tcl_template],
                        f"synth/{tc['module']}/{tc['test']}",
                        phys_dir
                    )
                    results["passed" if passed else "failed"] += 1
                else:
                    results["skipped"] += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"RTL Verification Summary")
    print(f"{'='*60}")
    print(f"  Passed:  {results['passed']}")
    print(f"  Failed:  {results['failed']}")
    print(f"  Skipped: {results['skipped']}")

    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
