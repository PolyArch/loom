#!/usr/bin/env python3
"""Master RTL verification runner. Dispatches to gen_sv, run_sim, run_synth."""

import argparse
import glob
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


def generate_golden_traces(fcc_exec, adg_path, module_name, output_dir):
    """Run fcc --simulate --trace-port-dump to produce golden traces."""
    trace_dir = os.path.join(output_dir, "rtl-traces")
    cmd = [
        fcc_exec,
        "--gen-sv",
        "--simulate",
        "--adg", adg_path,
        "--trace-port-dump", module_name,
        "-o", output_dir,
    ]
    print(f"[behaviour] Generating golden traces: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[behaviour] FAIL: fcc golden trace generation failed")
        if result.stderr:
            print(result.stderr[:2000])
        return None
    if os.path.isdir(trace_dir):
        return trace_dir
    return None


def find_golden_traces(test_output):
    """Look for golden trace files under the test output tree."""
    trace_dir = os.path.join(test_output, "rtl-traces")
    if not os.path.isdir(trace_dir):
        return []
    return sorted(glob.glob(os.path.join(trace_dir, "*.hex")))


def extract_module_name(mlir_path):
    """Extract the fabric.module @name from an MLIR file.

    Scans for `fabric.module @<name>` and returns the name string.
    Falls back to None if not found.
    """
    import re
    pattern = re.compile(r'fabric\.module\s+@(\w+)')
    try:
        with open(mlir_path, "r") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def find_output_trace(trace_dir, module_name):
    """Find the output port trace file for a given module.

    Looks for files matching *_out0_tokens.hex or *_mod_out0*.hex
    in the trace directory. Falls back to the first *out*.hex file.
    """
    if not os.path.isdir(trace_dir):
        return None

    # Preferred: exact output port trace patterns
    for pattern_str in [
        os.path.join(trace_dir, f"{module_name}_out0_tokens.hex"),
        os.path.join(trace_dir, f"{module_name}_mod_out0_tokens.hex"),
    ]:
        matches = glob.glob(pattern_str)
        if matches:
            return matches[0]

    # Fallback: any file with 'out' and ending in _tokens.hex
    candidates = glob.glob(os.path.join(trace_dir, "*out*_tokens.hex"))
    if candidates:
        return sorted(candidates)[0]

    # Last resort: any file with 'out' in the name
    candidates = glob.glob(os.path.join(trace_dir, "*out*.hex"))
    if candidates:
        return sorted(candidates)[0]

    return None


def find_input_trace(trace_dir, module_name):
    """Find the input port trace file for a given module.

    Looks for files matching *_in0_tokens.hex in the trace directory.
    """
    if not os.path.isdir(trace_dir):
        return None

    for pattern_str in [
        os.path.join(trace_dir, f"{module_name}_in0_tokens.hex"),
        os.path.join(trace_dir, f"{module_name}_mod_in0_tokens.hex"),
    ]:
        matches = glob.glob(pattern_str)
        if matches:
            return matches[0]

    candidates = glob.glob(os.path.join(trace_dir, "*in0*_tokens.hex"))
    if candidates:
        return sorted(candidates)[0]

    candidates = glob.glob(os.path.join(trace_dir, "*in*.hex"))
    if candidates:
        return sorted(candidates)[0]

    return None


def read_count_file(hex_path):
    """Read a .count file associated with a .hex trace file.

    Given a path like /dir/module_out0_tokens.hex, looks for
    /dir/module_out0_tokens.count and returns the integer count.
    Returns 0 if the .count file does not exist or is unreadable.
    """
    count_path = hex_path.rsplit(".hex", 1)[0] + ".count"
    try:
        with open(count_path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def find_config_hex(gen_dir):
    """Find a config hex file in the generated RTL collateral directory.

    Looks for files matching *config*.hex in the gen directory tree.
    """
    if not os.path.isdir(gen_dir):
        return None

    candidates = glob.glob(os.path.join(gen_dir, "**", "*config*.hex"),
                           recursive=True)
    if candidates:
        return sorted(candidates)[0]

    return None


def count_hex_lines(hex_path):
    """Count non-empty, non-comment lines in a hex file."""
    if not hex_path or not os.path.isfile(hex_path):
        return 0
    count = 0
    try:
        with open(hex_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("//"):
                    count += 1
    except OSError:
        pass
    return count


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

        # Check 1: Functional behaviour verification
        if "behaviour" in args.checks:
            beh_dir = os.path.join(test_output, "behaviour")
            os.makedirs(beh_dir, exist_ok=True)
            if not find_tool("verilator"):
                print(f"[behaviour/{tc['module']}/{tc['test']}] SKIP: verilator not found")
                results["skipped"] += 1
            else:
                gen_dir = os.path.join(test_output, "gen-collateral")
                rtl_dir = os.path.join(gen_dir, "rtl")
                tb_dir = os.path.join(args.src_rtl, "testbench", "common")

                # Extract the real fabric.module @name from MLIR
                fabric_module_name = extract_module_name(tc["mlir"])
                if not fabric_module_name:
                    fabric_module_name = tc["module"]
                    print(f"[behaviour/{tc['module']}/{tc['test']}] "
                          f"WARN: could not extract module name from MLIR, "
                          f"using directory name: {fabric_module_name}")

                # Auto-generate RTL collateral if missing.
                if not os.path.isdir(rtl_dir):
                    print(f"[behaviour/{tc['module']}/{tc['test']}] "
                          "gen-collateral/rtl not found, running gen first")
                    os.makedirs(gen_dir, exist_ok=True)
                    gen_ok = run_check(
                        os.path.join(scripts_dir, "gen_sv.py"),
                        ["--fcc", args.fcc, "--adg", tc["mlir"],
                         "--output-dir", gen_dir],
                        f"gen(auto)/{tc['module']}/{tc['test']}",
                        gen_dir
                    )
                    if not gen_ok:
                        print(f"[behaviour/{tc['module']}/{tc['test']}] "
                              "FAIL: auto-gen failed")
                        results["failed"] += 1
                        continue

                golden_traces = find_golden_traces(test_output)
                if not golden_traces:
                    trace_result = generate_golden_traces(
                        args.fcc, tc["mlir"], fabric_module_name,
                        test_output)
                    if trace_result:
                        golden_traces = find_golden_traces(test_output)

                if not golden_traces:
                    print(f"[behaviour/{tc['module']}/{tc['test']}] "
                          "FAIL: no golden traces available")
                    results["failed"] += 1
                elif not os.path.isdir(rtl_dir):
                    print(f"[behaviour/{tc['module']}/{tc['test']}] "
                          "FAIL: generated RTL not found")
                    results["failed"] += 1
                else:
                    trace_dir = os.path.join(test_output, "rtl-traces")

                    # Find specific golden output trace and set up output path
                    golden_out_trace = find_output_trace(trace_dir,
                                                         fabric_module_name)
                    output_trace_path = os.path.join(beh_dir, "sim_out0.hex")

                    # Find input trace for the module
                    input_trace = find_input_trace(trace_dir,
                                                    fabric_module_name)

                    # Read token counts from .count files produced by
                    # PortTraceExporter, falling back to line counting
                    golden_token_count = 0
                    input_token_count = 0
                    if golden_out_trace:
                        golden_token_count = read_count_file(golden_out_trace)
                        if golden_token_count == 0:
                            golden_token_count = count_hex_lines(
                                golden_out_trace)
                    if input_trace:
                        input_token_count = read_count_file(input_trace)
                        if input_token_count == 0:
                            input_token_count = count_hex_lines(input_trace)

                    # Find config hex file and count words
                    config_hex = find_config_hex(gen_dir)
                    config_word_count = (count_hex_lines(config_hex)
                                         if config_hex else 0)

                    # DUT module name for the generated top-level SV module
                    dut_module = "fabric_top_" + fabric_module_name

                    sim_args = [
                        "--rtl-dir", rtl_dir,
                        "--tb-dir", tb_dir,
                        "--output-dir", beh_dir,
                        "--tool", "verilator",
                        "--trace-dir", trace_dir,
                        "--dut-module", dut_module,
                    ]

                    # Build plusargs for runtime TB configuration
                    plusarg_list = []
                    if input_token_count > 0:
                        plusarg_list.append(
                            f"+NUM_INPUT_TOKENS={input_token_count}")
                    if golden_token_count > 0:
                        plusarg_list.append(
                            f"+GOLDEN_TOKENS={golden_token_count}")
                    if config_word_count > 0:
                        plusarg_list.append(
                            f"+NUM_CONFIG_WORDS={config_word_count}")
                    if input_trace:
                        plusarg_list.append(
                            f"+INPUT_TRACE_0={input_trace}")
                    if config_hex:
                        plusarg_list.append(
                            f"+CONFIG_FILE={config_hex}")

                    if golden_out_trace:
                        plusarg_list.append(
                            f"+GOLDEN_TRACE_0={golden_out_trace}")
                        sim_args.extend([
                            "--golden-trace", golden_out_trace,
                            "--output-trace", output_trace_path,
                        ])
                    if output_trace_path:
                        plusarg_list.append(
                            f"+OUTPUT_TRACE_0={output_trace_path}")

                    # Pass plusargs to run_sim.py
                    if plusarg_list:
                        sim_args.extend(["--plusargs"] + plusarg_list)

                    passed = run_check(
                        os.path.join(scripts_dir, "run_sim.py"),
                        sim_args,
                        f"behaviour/{tc['module']}/{tc['test']}",
                        beh_dir
                    )
                    results["passed" if passed else "failed"] += 1

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
                # Auto-generate RTL collateral if missing.
                if not os.path.isdir(gen_dir):
                    print(f"[synth/{tc['module']}/{tc['test']}] "
                          "gen-collateral/rtl not found, running gen first")
                    gen_base = os.path.join(test_output, "gen-collateral")
                    os.makedirs(gen_base, exist_ok=True)
                    gen_ok = run_check(
                        os.path.join(scripts_dir, "gen_sv.py"),
                        ["--fcc", args.fcc, "--adg", tc["mlir"],
                         "--output-dir", gen_base],
                        f"gen(auto)/{tc['module']}/{tc['test']}",
                        gen_base
                    )
                    if not gen_ok:
                        print(f"[synth/{tc['module']}/{tc['test']}] "
                              "FAIL: auto-gen failed")
                        results["failed"] += 1
                        continue
                if os.path.isdir(gen_dir):
                    # Extract the real fabric.module @name for the
                    # synthesis design name (the top SV module is
                    # fabric_top_<name>).
                    synth_module_name = extract_module_name(tc["mlir"])
                    if not synth_module_name:
                        synth_module_name = tc["module"]
                    synth_design_name = "fabric_top_" + synth_module_name
                    passed = run_check(
                        os.path.join(scripts_dir, "run_synth.py"),
                        ["--rtl-dir", gen_dir, "--design-name", synth_design_name,
                         "--output-dir", phys_dir, "--tcl-template", tcl_template],
                        f"synth/{tc['module']}/{tc['test']}",
                        phys_dir
                    )
                    results["passed" if passed else "failed"] += 1
                else:
                    print(f"[synth/{tc['module']}/{tc['test']}] "
                          "SKIP: gen-collateral/rtl still missing after auto-gen")
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
