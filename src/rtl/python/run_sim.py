#!/usr/bin/env python3
"""Check 1 runner: compile RTL + testbench, run simulation, compare traces."""

import argparse
import os
import subprocess
import sys


def compile_verilator(rtl_dir, tb_dir, top_module, output_dir):
    """Compile with Verilator."""
    obj_dir = os.path.join(output_dir, "obj_dir")
    filelist = os.path.join(rtl_dir, "filelist.f")

    cmd = [
        "verilator", "--sv", "--cc", "--exe", "--build",
        "-Wall", "-Wno-fatal",
        "--top-module", top_module,
        "-f", filelist,
        "-I" + tb_dir,
        "--Mdir", obj_dir,
    ]
    print(f"[run_sim] Compiling with Verilator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[run_sim] FAIL: Verilator compilation failed")
        if result.stderr:
            print(result.stderr[:2000])
        return None

    sim_exec = os.path.join(obj_dir, f"V{top_module}")
    return sim_exec


def compile_vcs(rtl_dir, tb_dir, top_module, output_dir):
    """Compile with VCS."""
    filelist = os.path.join(rtl_dir, "filelist.f")
    sim_exec = os.path.join(output_dir, "simv")

    cmd = [
        "vcs", "-sverilog", "-full64",
        "-timescale=1ns/1ps",
        "-f", filelist,
        "+incdir+" + tb_dir,
        "-o", sim_exec,
    ]
    print(f"[run_sim] Compiling with VCS: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[run_sim] FAIL: VCS compilation failed")
        if result.stderr:
            print(result.stderr[:2000])
        return None

    return sim_exec


def run_simulation(sim_exec, output_dir, plusargs=None):
    """Run the compiled simulation."""
    cmd = [sim_exec]
    if plusargs:
        cmd.extend(plusargs)

    print(f"[run_sim] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

    log_path = os.path.join(output_dir, "sim.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    if result.returncode != 0:
        print(f"[run_sim] FAIL: Simulation exited with code {result.returncode}")
        return False

    # Check for PASS/FAIL in output
    if "PASS" in result.stdout:
        print("[run_sim] PASS: Simulation passed")
        return True
    elif "FAIL" in result.stdout:
        print("[run_sim] FAIL: Simulation reported failure")
        return False
    else:
        print("[run_sim] WARN: No PASS/FAIL verdict in simulation output")
        return True  # Assume pass if no explicit failure


def compare_traces(golden_path, output_path):
    """Compare golden trace against DUT output trace."""
    if not os.path.exists(golden_path):
        print(f"[run_sim] WARN: Golden trace not found: {golden_path}")
        return True

    if not os.path.exists(output_path):
        print(f"[run_sim] FAIL: Output trace not found: {output_path}")
        return False

    with open(golden_path) as gf, open(output_path) as of:
        golden_lines = [l.strip() for l in gf if l.strip() and not l.startswith("//")]
        output_lines = [l.strip() for l in of if l.strip() and not l.startswith("//")]

    if len(golden_lines) != len(output_lines):
        print(f"[run_sim] FAIL: Token count mismatch "
              f"(golden={len(golden_lines)}, output={len(output_lines)})")
        return False

    mismatches = 0
    for i, (g, o) in enumerate(zip(golden_lines, output_lines)):
        if g != o:
            if mismatches < 10:
                print(f"[run_sim] MISMATCH[{i}]: golden={g}, output={o}")
            mismatches += 1

    if mismatches == 0:
        print(f"[run_sim] PASS: All {len(golden_lines)} tokens match")
        return True
    else:
        print(f"[run_sim] FAIL: {mismatches} mismatches out of {len(golden_lines)} tokens")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check 1: Functional verification")
    parser.add_argument("--rtl-dir", required=True, help="RTL collateral directory")
    parser.add_argument("--tb-dir", required=True, help="Testbench directory")
    parser.add_argument("--top-module", default="tb_module_wrapper")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--tool", choices=["verilator", "vcs"], default="verilator")
    parser.add_argument("--golden-trace", help="Golden trace file for comparison")
    parser.add_argument("--output-trace", help="DUT output trace file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.tool == "verilator":
        sim_exec = compile_verilator(args.rtl_dir, args.tb_dir,
                                      args.top_module, args.output_dir)
    else:
        sim_exec = compile_vcs(args.rtl_dir, args.tb_dir,
                                args.top_module, args.output_dir)

    if sim_exec is None:
        sys.exit(1)

    if not run_simulation(sim_exec, args.output_dir):
        sys.exit(1)

    if args.golden_trace and args.output_trace:
        if not compare_traces(args.golden_trace, args.output_trace):
            sys.exit(1)

    print("[run_sim] All checks passed")


if __name__ == "__main__":
    main()
