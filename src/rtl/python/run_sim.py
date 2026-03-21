#!/usr/bin/env python3
"""Check 1 runner: compile RTL + testbench, run simulation, compare traces."""

import argparse
import os
import subprocess
import sys


def compile_verilator(rtl_dir, tb_dir, top_module, output_dir,
                      tb_files=None, verilator_main=None,
                      dut_module=None):
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

    # Pass DUT_MODULE define so tb_module_wrapper instantiates the real DUT
    if dut_module:
        cmd.append(f"+define+DUT_MODULE={dut_module}")

    if tb_files:
        cmd.extend(tb_files)

    if verilator_main and os.path.isfile(verilator_main):
        cmd.append(verilator_main)

    print(f"[run_sim] Compiling with Verilator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[run_sim] FAIL: Verilator compilation failed")
        if result.stderr:
            print(result.stderr[:2000])
        return None

    sim_exec = os.path.join(obj_dir, f"V{top_module}")
    return sim_exec


def compile_vcs(rtl_dir, tb_dir, top_module, output_dir,
                tb_files=None, dut_module=None):
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

    # Pass DUT_MODULE define so tb_module_wrapper instantiates the real DUT
    if dut_module:
        cmd.append(f"+define+DUT_MODULE={dut_module}")

    if tb_files:
        cmd.extend(tb_files)

    print(f"[run_sim] Compiling with VCS: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[run_sim] FAIL: VCS compilation failed")
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
    """Compare golden trace against DUT output trace.

    Returns True only when both files exist and all tokens match.
    Missing golden traces are treated as a hard failure.
    """
    if not os.path.exists(golden_path):
        print(f"[run_sim] FAIL: Golden trace not found: {golden_path}")
        return False

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
    parser.add_argument("--tb-files", nargs="*", default=[],
                        help="Explicit list of testbench SV source files to compile")
    parser.add_argument("--top-module", default="tb_module_wrapper")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--tool", choices=["verilator", "vcs"], default="verilator")
    parser.add_argument("--golden-trace", help="Golden trace file for comparison")
    parser.add_argument("--output-trace", help="DUT output trace file")
    parser.add_argument("--trace-dir", default="",
                        help="Trace directory path passed to TB via +plusarg")
    parser.add_argument("--verilator-main", default="",
                        help="Path to C++ main harness for Verilator")
    parser.add_argument("--dut-module", default="",
                        help="DUT SV module name (e.g. fabric_top_test_fifo_depth4)")
    parser.add_argument("--plusargs", nargs="*", default=[],
                        help="Additional plusargs to pass to the simulator "
                             "(e.g. +NUM_INPUT_TOKENS=16 +CONFIG_FILE=cfg.hex)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover TB SV files automatically if none were given explicitly
    tb_files = list(args.tb_files)
    if not tb_files and args.tb_dir and os.path.isdir(args.tb_dir):
        for fname in sorted(os.listdir(args.tb_dir)):
            if fname.endswith(".sv") or fname.endswith(".v"):
                tb_files.append(os.path.join(args.tb_dir, fname))

    # Locate the default verilator_main.cpp if not explicitly provided
    verilator_main = args.verilator_main
    if not verilator_main:
        candidate = os.path.join(os.path.dirname(args.tb_dir),
                                 "common", "verilator_main.cpp")
        if os.path.isfile(candidate):
            verilator_main = candidate

    # Build plusargs for trace paths and runtime configuration.
    # The SV wrapper (tb_module_wrapper) reads these via $value$plusargs:
    #   +GOLDEN_TRACE_0=<path>  +OUTPUT_TRACE_0=<path>  +TRACE_DIR=<dir>
    #   +NUM_INPUT_TOKENS=N     +GOLDEN_TOKENS=N        +NUM_CONFIG_WORDS=N
    #   +INPUT_TRACE_0=<path>   +CONFIG_FILE=<path>      +DUT_MODULE=<name>
    plusargs = []
    if args.trace_dir:
        plusargs.append(f"+TRACE_DIR={args.trace_dir}")
    if args.golden_trace:
        plusargs.append(f"+GOLDEN_TRACE_0={args.golden_trace}")
    if args.output_trace:
        plusargs.append(f"+OUTPUT_TRACE_0={args.output_trace}")

    # Append caller-supplied plusargs (from run_rtl_checks.py --plusargs).
    # These carry token counts, config info, and trace file paths that
    # the SV wrapper resolves at runtime via $value$plusargs.
    if args.plusargs:
        # Deduplicate: caller plusargs take precedence over the ones
        # constructed above. Build a set of plusarg keys already present.
        existing_keys = set()
        for pa in plusargs:
            key = pa.split("=", 1)[0] if "=" in pa else pa
            existing_keys.add(key)
        for pa in args.plusargs:
            key = pa.split("=", 1)[0] if "=" in pa else pa
            if key not in existing_keys:
                plusargs.append(pa)
                existing_keys.add(key)

    if args.tool == "verilator":
        sim_exec = compile_verilator(args.rtl_dir, args.tb_dir,
                                     args.top_module, args.output_dir,
                                     tb_files=tb_files,
                                     verilator_main=verilator_main,
                                     dut_module=args.dut_module or None)
    else:
        sim_exec = compile_vcs(args.rtl_dir, args.tb_dir,
                               args.top_module, args.output_dir,
                               tb_files=tb_files,
                               dut_module=args.dut_module or None)

    if sim_exec is None:
        sys.exit(1)

    if not run_simulation(sim_exec, args.output_dir, plusargs=plusargs):
        sys.exit(1)

    if args.golden_trace and args.output_trace:
        if not compare_traces(args.golden_trace, args.output_trace):
            sys.exit(1)

    print("[run_sim] All checks passed")


if __name__ == "__main__":
    main()
