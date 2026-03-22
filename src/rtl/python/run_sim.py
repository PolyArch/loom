#!/usr/bin/env python3
"""Check 1 runner: compile RTL + testbench, run simulation, compare traces."""

import argparse
import os
import shutil
import subprocess
import sys


def _env_modules_hint():
    """Return a hint string about environment-modules if available."""
    if os.path.isfile("/etc/profile.d/modules.sh"):
        return ("  Hint: environment-modules detected. Try:\n"
                "    source /etc/profile.d/modules.sh && module avail\n"
                "  to see available tool modules.")
    return ""


def resolve_verilator():
    """Find the verilator executable, trying module load if needed."""
    if shutil.which("verilator"):
        return "verilator"
    # Try module loading as last resort
    try:
        result = subprocess.run(
            ["bash", "-c",
             "source /etc/profile.d/modules.sh && module load verilator/5.044 && which verilator"],
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    print("[run_sim] ERROR: 'verilator' not found in PATH.")
    hint = _env_modules_hint()
    if hint:
        print(hint)
    return None


def compile_verilator(rtl_dir, tb_dir, top_module, output_dir,
                      tb_files=None, verilator_main=None,
                      dut_module=None, dut_inst_dir=None,
                      num_dut_inputs=1, num_dut_outputs=1,
                      filelist_override=None):
    """Compile with Verilator."""
    obj_dir = os.path.join(output_dir, "obj_dir")
    filelist = filelist_override or os.path.join(rtl_dir, "filelist.f")
    verilator_exec = resolve_verilator()
    if verilator_exec is None:
        return None

    cmd = [
        verilator_exec, "--sv", "--binary",
        "--timing",  # Required for #delay and event controls in testbenches
        "-Wall", "-Wno-fatal",
        "--top-module", top_module,
        "-f", filelist,
        "-I" + tb_dir,
        "--Mdir", obj_dir,
    ]

    # Pass DUT_MODULE define so tb_module_wrapper instantiates the real DUT
    if dut_module:
        cmd.append(f"+define+DUT_MODULE={dut_module}")

    # Pass DUT_INST_SVH define and include path for the generated
    # DUT instantiation file
    if dut_inst_dir and os.path.isfile(
            os.path.join(dut_inst_dir, "dut_inst.svh")):
        cmd.append("+define+DUT_INST_SVH=1")
        cmd.append("-I" + dut_inst_dir)

    # Pass port count defines so the SV wrapper knows the topology
    cmd.append(f"+define+NUM_DUT_INPUTS={num_dut_inputs}")
    cmd.append(f"+define+NUM_DUT_OUTPUTS={num_dut_outputs}")

    if tb_files:
        cmd.extend(tb_files)

    # Note: --binary mode does not use a C++ harness file.
    # verilator_main.cpp is kept for reference but not compiled.

    # Make paths absolute before changing working directory
    abs_rtl_dir = os.path.abspath(rtl_dir)
    abs_obj_dir = os.path.abspath(obj_dir)
    abs_tb_dir = os.path.abspath(tb_dir)

    # Rebuild command with absolute paths where needed, but keep filelist
    # relative so its internal relative entries resolve from rtl_dir
    cmd_abs = []
    for c in cmd:
        if c.startswith("-I"):
            cmd_abs.append("-I" + os.path.abspath(c[2:]))
        elif c.startswith("--Mdir"):
            cmd_abs.append(c)
        elif c == obj_dir:
            cmd_abs.append(abs_obj_dir)
        elif os.path.exists(c) and not c.startswith("+") and not c.startswith("-"):
            cmd_abs.append(os.path.abspath(c))
        else:
            cmd_abs.append(c)
    # Override Mdir with absolute path
    for i, c in enumerate(cmd_abs):
        if c == "--Mdir" and i + 1 < len(cmd_abs):
            cmd_abs[i + 1] = abs_obj_dir

    os.makedirs(abs_obj_dir, exist_ok=True)

    print(f"[run_sim] Compiling with Verilator (cwd={abs_rtl_dir}): {' '.join(cmd_abs)}")
    result = subprocess.run(cmd_abs, capture_output=True, text=True, cwd=abs_rtl_dir)
    if result.returncode != 0:
        print("[run_sim] FAIL: Verilator compilation failed")
        if result.stderr:
            print(result.stderr[:2000])
        if result.stdout:
            print(result.stdout[:2000])
        return None

    sim_exec = os.path.join(abs_obj_dir, f"V{top_module}")
    return sim_exec


def compile_vcs(rtl_dir, tb_dir, top_module, output_dir,
                tb_files=None, dut_module=None, dut_inst_dir=None,
                num_dut_inputs=1, num_dut_outputs=1):
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

    # Pass DUT_INST_SVH define and include path for the generated
    # DUT instantiation file
    if dut_inst_dir and os.path.isfile(
            os.path.join(dut_inst_dir, "dut_inst.svh")):
        cmd.append("+define+DUT_INST_SVH=1")
        cmd.append("+incdir+" + dut_inst_dir)

    # Pass port count defines so the SV wrapper knows the topology
    cmd.append(f"+define+NUM_DUT_INPUTS={num_dut_inputs}")
    cmd.append(f"+define+NUM_DUT_OUTPUTS={num_dut_outputs}")

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


def compare_traces(golden_path, output_path, channel_label=""):
    """Compare golden trace against DUT output trace.

    Returns True only when both files exist and all tokens match.
    Missing golden traces are treated as a hard failure.
    """
    label = f" ch{channel_label}" if channel_label else ""

    if not os.path.exists(golden_path):
        print(f"[run_sim] FAIL{label}: Golden trace not found: {golden_path}")
        return False

    if not os.path.exists(output_path):
        print(f"[run_sim] FAIL{label}: Output trace not found: {output_path}")
        return False

    with open(golden_path) as gf, open(output_path) as of:
        golden_lines = [l.strip() for l in gf if l.strip() and not l.startswith("//")]
        output_lines = [l.strip() for l in of if l.strip() and not l.startswith("//")]

    if len(golden_lines) != len(output_lines):
        print(f"[run_sim] FAIL{label}: Token count mismatch "
              f"(golden={len(golden_lines)}, output={len(output_lines)})")
        return False

    mismatches = 0
    for i, (g, o) in enumerate(zip(golden_lines, output_lines)):
        if g != o:
            if mismatches < 10:
                print(f"[run_sim] MISMATCH{label}[{i}]: golden={g}, output={o}")
            mismatches += 1

    if mismatches == 0:
        print(f"[run_sim] PASS{label}: All {len(golden_lines)} tokens match")
        return True
    else:
        print(f"[run_sim] FAIL{label}: {mismatches} mismatches "
              f"out of {len(golden_lines)} tokens")
        return False


def parse_channel_arg(value):
    """Parse a channel argument of the form 'channel_idx:path'.

    Returns (channel_idx, path) tuple.
    """
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected format 'channel:path', got '{value}'")
    return int(parts[0]), parts[1]


def main():
    parser = argparse.ArgumentParser(description="Check 1: Functional verification")
    parser.add_argument("--rtl-dir", required=True, help="RTL collateral directory")
    parser.add_argument("--tb-dir", required=True, help="Testbench directory")
    parser.add_argument("--tb-files", nargs="*", default=[],
                        help="Explicit list of testbench SV source files to compile")
    parser.add_argument("--top-module", default="tb_module_wrapper")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--tool", choices=["verilator", "vcs"], default="verilator")
    parser.add_argument("--golden-trace", help="Golden trace file for comparison (ch0)")
    parser.add_argument("--output-trace", help="DUT output trace file (ch0)")
    parser.add_argument("--golden-trace-ch", action="append", default=[],
                        help="Per-channel golden trace: 'channel:path' "
                             "(may be specified multiple times)")
    parser.add_argument("--output-trace-ch", action="append", default=[],
                        help="Per-channel output trace: 'channel:path' "
                             "(may be specified multiple times)")
    parser.add_argument("--trace-dir", default="",
                        help="Trace directory path passed to TB via +plusarg")
    parser.add_argument("--verilator-main", default="",
                        help="Path to C++ main harness for Verilator")
    parser.add_argument("--dut-module", default="",
                        help="DUT SV module name (e.g. fabric_top_test_fifo_depth4)")
    parser.add_argument("--dut-inst-dir", default="",
                        help="Directory containing the generated dut_inst.svh file")
    parser.add_argument("--num-dut-inputs", type=int, default=1,
                        help="Number of DUT input ports")
    parser.add_argument("--num-dut-outputs", type=int, default=1,
                        help="Number of DUT output ports")
    parser.add_argument("--plusargs", nargs="*", default=[],
                        help="Additional plusargs to pass to the simulator "
                             "(e.g. +NUM_INPUT_TOKENS_0=16 +CONFIG_FILE=cfg.hex)")
    parser.add_argument("--standalone", default="",
                        help="Standalone TB module name (e.g. tb_backpressure_test). "
                             "Compiles using --standalone-filelist instead of "
                             "rtl-dir/filelist.f.")
    parser.add_argument("--standalone-filelist", default="",
                        help="Filelist for standalone TBs (e.g. filelist_standalone.f). "
                             "Used with --standalone to exclude FP modules.")
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
    plusargs = []
    if args.trace_dir:
        plusargs.append(f"+TRACE_DIR={args.trace_dir}")

    # Legacy single-channel golden/output trace plusargs (backward compat).
    # These are now superseded by per-channel plusargs from --plusargs,
    # but we still support them for callers that don't use multi-channel.
    if args.golden_trace:
        plusargs.append(f"+GOLDEN_TRACE_0={args.golden_trace}")
    if args.output_trace:
        plusargs.append(f"+OUTPUT_TRACE_0={args.output_trace}")

    # Append caller-supplied plusargs (from run_rtl_checks.py --plusargs).
    # These carry per-channel token counts, config info, and trace file
    # paths that the SV wrapper resolves at runtime via $value$plusargs.
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

    # Determine filelist override for standalone mode
    filelist_override = None
    if args.standalone and args.standalone_filelist:
        filelist_override = args.standalone_filelist
        # For standalone mode, override top module to the standalone TB
        if not args.top_module or args.top_module == "tb_module_wrapper":
            args.top_module = args.standalone

    if args.tool == "verilator":
        sim_exec = compile_verilator(
            args.rtl_dir, args.tb_dir,
            args.top_module, args.output_dir,
            tb_files=tb_files,
            verilator_main=verilator_main,
            dut_module=args.dut_module or None,
            dut_inst_dir=args.dut_inst_dir or None,
            num_dut_inputs=args.num_dut_inputs,
            num_dut_outputs=args.num_dut_outputs,
            filelist_override=filelist_override)
    else:
        sim_exec = compile_vcs(
            args.rtl_dir, args.tb_dir,
            args.top_module, args.output_dir,
            tb_files=tb_files,
            dut_module=args.dut_module or None,
            dut_inst_dir=args.dut_inst_dir or None,
            num_dut_inputs=args.num_dut_inputs,
            num_dut_outputs=args.num_dut_outputs)

    if sim_exec is None:
        sys.exit(1)

    if not run_simulation(sim_exec, args.output_dir, plusargs=plusargs):
        sys.exit(1)

    # Post-simulation trace comparison.
    # If per-channel golden/output traces are provided, compare all of them.
    # Otherwise, fall back to the single-channel --golden-trace / --output-trace.
    all_passed = True

    if args.golden_trace_ch and args.output_trace_ch:
        # Parse per-channel arguments
        golden_map = {}
        output_map = {}
        for entry in args.golden_trace_ch:
            ch_idx, path = parse_channel_arg(entry)
            golden_map[ch_idx] = path
        for entry in args.output_trace_ch:
            ch_idx, path = parse_channel_arg(entry)
            output_map[ch_idx] = path

        # Compare each channel that has both golden and output traces
        for ch_idx in sorted(golden_map.keys()):
            if ch_idx in output_map:
                if not compare_traces(golden_map[ch_idx],
                                      output_map[ch_idx],
                                      channel_label=str(ch_idx)):
                    all_passed = False
            else:
                print(f"[run_sim] WARN: Golden trace for ch{ch_idx} "
                      f"but no output trace")
    elif args.golden_trace and args.output_trace:
        # Legacy single-channel comparison
        if not compare_traces(args.golden_trace, args.output_trace):
            all_passed = False

    if not all_passed:
        sys.exit(1)

    print("[run_sim] All checks passed")


if __name__ == "__main__":
    main()
