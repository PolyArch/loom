#!/usr/bin/env python3
"""Master RTL verification runner. Dispatches to gen_sv, run_sim, run_synth."""

import argparse
import glob
import os
import re
import subprocess
import sys


WELL_KNOWN_TOOLS = {
    "verilator": "/mnt/nas0/software/verilator/5.044/bin/verilator",
    "dc_shell": "/mnt/nas0/software/synopsys/syn/W-2024.09-SP5/bin/dc_shell",
}


def find_tool(name, module_spec=None):
    """Check if a tool is available, optionally via module loading."""
    import shutil
    if shutil.which(name):
        return True
    # Check well-known installation path
    wk = WELL_KNOWN_TOOLS.get(name)
    if wk and os.path.isfile(wk) and os.access(wk, os.X_OK):
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
    # Sanitize check_name for use as filename (replace / with _)
    safe_name = check_name.replace("/", "_").replace("\\", "_")
    log_path = os.path.join(output_dir, f"{safe_name}.log")
    os.makedirs(output_dir, exist_ok=True)
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
    """Generate golden traces from the C++ simulator.

    Golden trace generation requires a full mapped case (DFG+ADG).
    The caller must provide a companion <stem>.dfg.mlir alongside the
    <stem>.fabric.mlir ADG input. Without it, this function returns
    None and the testcase will be skipped by the behaviour runner.

    Returns the trace directory path on success, None on failure or
    when no companion DFG exists.
    """
    trace_dir = os.path.join(output_dir, "rtl-traces")

    # Look for a companion DFG file for mapped simulation
    adg_dir = os.path.dirname(adg_path)
    adg_stem = os.path.basename(adg_path).replace(".fabric.mlir", "")
    dfg_path = os.path.join(adg_dir, adg_stem + ".dfg.mlir")

    if not os.path.isfile(dfg_path):
        # ADG-only: no DFG companion, cannot generate golden traces.
        print(f"[behaviour] No companion DFG for {adg_path}; "
              "golden trace generation requires mapped DFG+ADG input")
        return None

    cmd = [fcc_exec, "--simulate", "--dfg", dfg_path,
           "--adg", adg_path, "--trace-port-dump", module_name,
           "-o", output_dir]
    print(f"[behaviour] Generating golden traces: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[behaviour] FAIL: fcc golden trace generation failed")
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


def extract_trace_target(mlir_path):
    """Extract the mapped hardware instance name for --trace-port-dump.

    The trace exporter uses mapped hardware instance names (e.g. 'fifo_0',
    'pe_0', 'sw0'), not the outer fabric.module name. This function looks
    for fabric.instance sym_name attributes inside the fabric.module body.
    For inline definitions without fabric.instance, it derives a name from
    the op type (e.g. fabric.fifo -> 'fifo_0', fabric.add_tag -> 'add_tag_0').

    Returns the first non-boundary instance name, or None if not found.
    """
    # Inline op type -> instance name prefix mapping
    inline_op_names = {
        "fabric.add_tag": "add_tag",
        "fabric.del_tag": "del_tag",
        "fabric.map_tag": "map_tag",
        "fabric.fifo": "fifo",
        "fabric.spatial_sw": "spatial_sw",
        "fabric.temporal_sw": "temporal_sw",
    }

    inst_pattern = re.compile(r'fabric\.instance\s+@\w+.*\{sym_name\s*=\s*"(\w+)"')
    # Inline ops with explicit @name: e.g. fabric.fifo @fifo_0
    named_inline_pattern = re.compile(
        r'(fabric\.(?:fifo|add_tag|del_tag|map_tag|spatial_sw|temporal_sw))'
        r'\s+@(\w+)')
    # Fallback: inline ops without @name (e.g. fabric.add_tag %in0, fabric.del_tag %in0)
    anon_inline_pattern = re.compile(r'(fabric\.(?:add_tag|del_tag|map_tag|fifo|spatial_sw|temporal_sw))\b')
    in_module = False
    instance_names = []
    inline_count = {}

    try:
        with open(mlir_path, "r") as f:
            for line in f:
                if 'fabric.module' in line:
                    in_module = True
                    continue
                if not in_module:
                    continue

                # Look for fabric.instance with sym_name
                m = inst_pattern.search(line)
                if m:
                    instance_names.append(m.group(1))
                    continue

                # Look for named inline ops: fabric.fifo @fifo_0
                m = named_inline_pattern.search(line)
                if m:
                    instance_names.append(m.group(2))
                    continue

                # Fallback: anonymous inline ops
                m = anon_inline_pattern.search(line)
                if m:
                    op_name = m.group(1)
                    if op_name in inline_op_names:
                        prefix = inline_op_names[op_name]
                        idx = inline_count.get(prefix, 0)
                        instance_names.append(f"{prefix}_{idx}")
                        inline_count[prefix] = idx + 1
    except OSError:
        pass

    return instance_names[0] if instance_names else None


def extract_port_info(mlir_path):
    """Extract visible (non-memref) input/output port counts and indices.

    Parses the MLIR file to find the fabric.module declaration. Counts
    only non-memref ports (matching SVGenTop's filtering). Returns
    (num_visible_inputs, num_outputs, visible_input_indices) where
    visible_input_indices is a list of the original argument numbers
    for the non-memref inputs (used for port naming: mod_in<orig_idx>).

    Falls back to (1, 1, [0]) on failure.
    """
    try:
        with open(mlir_path, "r") as f:
            content = f.read()
    except OSError:
        return 1, 1, [0]

    # Find the fabric.module declaration and extract its full signature.
    # Format: fabric.module @name( %arg0: type, ... ) -> ( type, ... ) {
    mod_pattern = re.compile(
        r'fabric\.module\s+@\w+\s*\((.*?)\)\s*->\s*\((.*?)\)\s*\{',
        re.DOTALL)
    m = mod_pattern.search(content)
    if not m:
        # Try single-output form: -> type {  (no parens around output)
        mod_pattern2 = re.compile(
            r'fabric\.module\s+@\w+\s*\((.*?)\)\s*->\s*([^{]+?)\s*\{',
            re.DOTALL)
        m = mod_pattern2.search(content)
        if not m:
            return 1, 1, [0]

    inputs_str = m.group(1).strip()
    outputs_str = m.group(2).strip()

    # Count input ports: each port is "%name: type"
    # Skip memref arguments (SVGenTop skips them for SV boundary ports).
    # Track original argument indices for port naming (mod_in<orig_idx>).
    num_inputs = 0
    visible_input_indices = []
    if inputs_str:
        arg_idx = 0
        for port_match in re.finditer(r'%\w+\s*:\s*([^,]+)', inputs_str):
            port_type = port_match.group(1).strip()
            if not port_type.startswith("memref"):
                num_inputs += 1
                visible_input_indices.append(arg_idx)
            arg_idx += 1

    # Count output ports: each is a type separated by top-level commas.
    num_outputs = 0
    if outputs_str:
        depth = 0
        count = 1 if outputs_str.strip() else 0
        for ch in outputs_str:
            if ch == '<':
                depth += 1
            elif ch == '>':
                depth -= 1
            elif ch == ',' and depth == 0:
                count += 1
        num_outputs = count

    if num_inputs == 0:
        num_inputs = 1
    if num_outputs == 0:
        num_outputs = 1

    if not visible_input_indices:
        visible_input_indices = list(range(num_inputs))
    return num_inputs, num_outputs, visible_input_indices


def find_port_traces(trace_dir, module_name, num_inputs, num_outputs):
    """Find per-port trace files based on PortTraceExporter naming.

    The PortTraceExporter writes files as:
        <module>_in<pi>_tokens.hex   (input ports)
        <module>_out<pi>_tokens.hex  (output ports)
    where <pi> is the port index in the traced ports vector. Input ports
    come first (0..num_inputs-1), then output ports (num_inputs..total-1).

    Returns:
        input_traces: list of (channel_idx, trace_path) tuples
        output_traces: list of (channel_idx, trace_path) tuples
    """
    input_traces = []
    output_traces = []

    if not os.path.isdir(trace_dir):
        return input_traces, output_traces

    # Find input traces: module_in<pi>_tokens.hex where pi = 0..num_inputs-1
    for pi in range(num_inputs):
        candidates = [
            os.path.join(trace_dir, f"{module_name}_in{pi}_tokens.hex"),
            os.path.join(trace_dir, f"{module_name}_mod_in{pi}_tokens.hex"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                input_traces.append((pi, path))
                break

    # Find output traces: module_out<pi>_tokens.hex
    # Port index in PortTraceExporter: outputs start at index num_inputs
    for oi in range(num_outputs):
        pi = num_inputs + oi
        candidates = [
            os.path.join(trace_dir, f"{module_name}_out{pi}_tokens.hex"),
            os.path.join(trace_dir, f"{module_name}_mod_out{pi}_tokens.hex"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                output_traces.append((oi, path))
                break

    # Fallback: if no traces found with expected naming, try generic patterns
    if not input_traces:
        for pi in range(num_inputs):
            pattern = os.path.join(trace_dir, f"*in{pi}*_tokens.hex")
            matches = sorted(glob.glob(pattern))
            if matches:
                input_traces.append((pi, matches[0]))

    if not output_traces:
        for oi in range(num_outputs):
            pattern = os.path.join(trace_dir, f"*out{oi}*_tokens.hex")
            matches = sorted(glob.glob(pattern))
            if matches:
                output_traces.append((oi, matches[0]))
        # If still nothing, try the old single-output fallback
        if not output_traces:
            pattern = os.path.join(trace_dir, "*out*_tokens.hex")
            matches = sorted(glob.glob(pattern))
            if matches:
                output_traces.append((0, matches[0]))

    return input_traces, output_traces


def generate_dut_inst_svh(output_path, dut_module, num_inputs, num_outputs,
                          visible_input_indices=None):
    """Generate the dut_inst.svh file with DUT port connections.

    Creates a SystemVerilog include file that instantiates the DUT module
    with the correct port topology, connecting to the testbench wrapper's
    drv_data/drv_valid/drv_ready and mon_data/mon_valid/mon_ready arrays.

    visible_input_indices: list of original argument indices for visible
    (non-memref) input ports. SVGenTop uses these indices for port naming:
    mod_in<orig_idx>. If None, defaults to dense [0..num_inputs-1].
    """
    if visible_input_indices is None:
        visible_input_indices = list(range(num_inputs))

    lines = []
    lines.append(f"// Auto-generated DUT instantiation for {dut_module}")
    lines.append(f"// Topology: {num_inputs} visible inputs (indices {visible_input_indices}), {num_outputs} outputs")
    lines.append("")

    # Build port connection list
    ports = []
    ports.append("    .clk            (clk)")
    ports.append("    .rst_n          (rst_n)")

    # Use original argument indices for port naming (matches SVGenTop)
    for drv_idx, orig_idx in enumerate(visible_input_indices):
        ports.append(f"    .mod_in{orig_idx}        (drv_data[{drv_idx}])")
        ports.append(f"    .mod_in{orig_idx}_valid  (drv_valid[{drv_idx}])")
        ports.append(f"    .mod_in{orig_idx}_ready  (drv_ready[{drv_idx}])")

    for o in range(num_outputs):
        ports.append(f"    .mod_out{o}       (mon_data[{o}])")
        ports.append(f"    .mod_out{o}_valid (mon_valid[{o}])")
        ports.append(f"    .mod_out{o}_ready (mon_ready[{o}])")

    ports.append("    .cfg_valid      (cfg_valid)")
    ports.append("    .cfg_wdata      (cfg_wdata)")
    ports.append("    .cfg_last       (cfg_last)")
    ports.append("    .cfg_ready      (cfg_ready)")

    lines.append("`DUT_MODULE u_dut (")
    lines.append(",\n".join(ports))
    lines.append(");")
    lines.append("")

    # Tie off unused input channel ready signals
    lines.append("// Tie off unused input channel ready signals")
    lines.append("generate")
    lines.append("    genvar gtr;")
    lines.append("    for (gtr = 0; gtr < MAX_CHANNELS; gtr = gtr + 1) "
                 "begin : gen_tieoff_ready")
    lines.append(f"        if (gtr >= {num_inputs}) "
                 "begin : tieoff_drv_ready")
    lines.append("            assign drv_ready[gtr] = 1'b1;")
    lines.append("        end")
    lines.append("    end")
    lines.append("endgenerate")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


def find_output_trace(trace_dir, module_name):
    """Find the output port trace file for a given module (legacy helper).

    Looks for files matching *_out0_tokens.hex or *_mod_out0*.hex
    in the trace directory. Falls back to the first *out*.hex file.
    """
    if not os.path.isdir(trace_dir):
        return None

    for pattern_str in [
        os.path.join(trace_dir, f"{module_name}_out0_tokens.hex"),
        os.path.join(trace_dir, f"{module_name}_mod_out0_tokens.hex"),
    ]:
        matches = glob.glob(pattern_str)
        if matches:
            return matches[0]

    candidates = glob.glob(os.path.join(trace_dir, "*out*_tokens.hex"))
    if candidates:
        return sorted(candidates)[0]

    candidates = glob.glob(os.path.join(trace_dir, "*out*.hex"))
    if candidates:
        return sorted(candidates)[0]

    return None


def find_input_trace(trace_dir, module_name):
    """Find the input port trace file for a given module (legacy helper).

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
    parser.add_argument("--modules", nargs="*", default=None,
                        help="Only run tests for these module directories (e.g., e2e)")
    parser.add_argument("--test-filter", default=None,
                        help="Only run tests matching this name (e.g., chess_2x2_stub)")
    args = parser.parse_args()

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    # Per-check result counters so gen passes don't mask behaviour skips
    results = {
        "gen": {"passed": 0, "failed": 0, "skipped": 0},
        "behaviour": {"passed": 0, "failed": 0, "skipped": 0},
        "synth": {"passed": 0, "failed": 0, "skipped": 0},
    }

    # Discover test cases
    fabric_dir = os.path.join(args.test_dir, "fabric")
    if not os.path.isdir(fabric_dir):
        print(f"ERROR: test directory not found: {fabric_dir}")
        sys.exit(1)

    # Directories excluded from generic discovery (have dedicated targets)
    EXCLUDED_DIRS = {"negative"}

    test_cases = []
    for module_name in sorted(os.listdir(fabric_dir)):
        module_dir = os.path.join(fabric_dir, module_name)
        if not os.path.isdir(module_dir):
            continue
        # Skip directories with dedicated targets (negative tests)
        if module_name in EXCLUDED_DIRS:
            continue
        # Filter by --modules if specified
        if args.modules is not None and module_name not in args.modules:
            continue
        for test_file in sorted(os.listdir(module_dir)):
            if test_file.endswith(".fabric.mlir"):
                test_name = test_file.replace(".fabric.mlir", "")
                # Filter by --test-filter if specified
                if args.test_filter is not None and test_name != args.test_filter:
                    continue
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
            results["gen"]["passed" if passed else "failed"] += 1

        # Check 1: Functional behaviour verification
        if "behaviour" in args.checks:
            beh_dir = os.path.join(test_output, "behaviour")
            os.makedirs(beh_dir, exist_ok=True)
            if not find_tool("verilator", "verilator/5.044"):
                print(f"[behaviour/{tc['module']}/{tc['test']}] SKIP: verilator not found")
                results["behaviour"]["skipped"] += 1
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

                # Extract the mapped hardware instance name for trace export.
                # The trace exporter uses mapped instance names (e.g. fifo_0),
                # not the outer fabric.module name.
                trace_target = extract_trace_target(tc["mlir"])
                if not trace_target:
                    trace_target = fabric_module_name
                    print(f"[behaviour/{tc['module']}/{tc['test']}] "
                          f"WARN: no trace target found, falling back to "
                          f"module name: {trace_target}")

                # Extract port topology from MLIR
                num_inputs, num_outputs, visible_input_indices = extract_port_info(tc["mlir"])
                print(f"[behaviour/{tc['module']}/{tc['test']}] "
                      f"Port topology: {num_inputs} inputs, "
                      f"{num_outputs} outputs")

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
                        results["behaviour"]["failed"] += 1
                        continue

                golden_traces = find_golden_traces(test_output)
                if not golden_traces:
                    trace_result = generate_golden_traces(
                        args.fcc, tc["mlir"], trace_target,
                        test_output)
                    if trace_result:
                        golden_traces = find_golden_traces(test_output)

                if not golden_traces:
                    # Missing golden traces = behaviour FAIL. The C++
                    # simulator requires a mapped DFG+ADG input. Add a
                    # companion .dfg.mlir alongside the .fabric.mlir.
                    print(f"[behaviour/{tc['module']}/{tc['test']}] "
                          "FAIL: no golden traces (missing companion "
                          ".dfg.mlir for mapped simulation)")
                    results["behaviour"]["failed"] += 1
                elif not os.path.isdir(rtl_dir):
                    print(f"[behaviour/{tc['module']}/{tc['test']}] "
                          "FAIL: generated RTL not found")
                    results["behaviour"]["failed"] += 1
                else:
                    trace_dir = os.path.join(test_output, "rtl-traces")

                    # Find per-port traces using multi-port discovery.
                    # Use trace_target (mapped instance name) for trace file
                    # naming, since the exporter writes files like
                    # fifo_0_in0_tokens.hex, not test_fifo_depth4_in0_tokens.hex.
                    input_traces, output_traces = find_port_traces(
                        trace_dir, trace_target,
                        num_inputs, num_outputs)

                    # DUT module name for the generated top-level SV module
                    dut_module = "fabric_top_" + fabric_module_name

                    # Generate the DUT instantiation include file
                    dut_inst_path = os.path.join(beh_dir, "dut_inst.svh")
                    generate_dut_inst_svh(
                        dut_inst_path, dut_module,
                        num_inputs, num_outputs,
                        visible_input_indices)
                    print(f"[behaviour/{tc['module']}/{tc['test']}] "
                          f"Generated dut_inst.svh at {dut_inst_path}")

                    # Find config hex file and count words
                    config_hex = find_config_hex(gen_dir)
                    config_word_count = (count_hex_lines(config_hex)
                                         if config_hex else 0)

                    sim_args = [
                        "--rtl-dir", rtl_dir,
                        "--tb-dir", tb_dir,
                        "--output-dir", beh_dir,
                        "--tool", "verilator",
                        "--trace-dir", trace_dir,
                        "--dut-module", dut_module,
                        "--dut-inst-dir", beh_dir,
                        "--num-dut-inputs", str(num_inputs),
                        "--num-dut-outputs", str(num_outputs),
                    ]

                    # Build per-channel plusargs for runtime TB config
                    plusarg_list = []

                    # Per-input-channel plusargs
                    for ch_idx, in_trace in input_traces:
                        in_count = read_count_file(in_trace)
                        if in_count == 0:
                            in_count = count_hex_lines(in_trace)
                        plusarg_list.append(
                            f"+INPUT_TRACE_{ch_idx}={in_trace}")
                        if in_count > 0:
                            plusarg_list.append(
                                f"+NUM_INPUT_TOKENS_{ch_idx}={in_count}")

                    # Per-output-channel plusargs
                    golden_trace_paths = []
                    output_trace_paths = []
                    for ch_idx, out_trace in output_traces:
                        golden_count = read_count_file(out_trace)
                        if golden_count == 0:
                            golden_count = count_hex_lines(out_trace)
                        out_path = os.path.join(
                            beh_dir, f"sim_out{ch_idx}.hex")

                        plusarg_list.append(
                            f"+GOLDEN_TRACE_{ch_idx}={out_trace}")
                        plusarg_list.append(
                            f"+OUTPUT_TRACE_{ch_idx}={out_path}")
                        if golden_count > 0:
                            plusarg_list.append(
                                f"+GOLDEN_TOKENS_{ch_idx}={golden_count}")

                        golden_trace_paths.append(
                            (ch_idx, out_trace, golden_count))
                        output_trace_paths.append(
                            (ch_idx, out_path))

                    # Config plusargs
                    if config_word_count > 0:
                        plusarg_list.append(
                            f"+NUM_CONFIG_WORDS={config_word_count}")
                    if config_hex:
                        plusarg_list.append(
                            f"+CONFIG_FILE={config_hex}")

                    # Pass golden/output trace paths for post-sim
                    # comparison. Use the first output channel for
                    # backward compatibility, plus multi-channel info.
                    if golden_trace_paths:
                        sim_args.extend([
                            "--golden-trace",
                            golden_trace_paths[0][1],
                            "--output-trace",
                            output_trace_paths[0][1],
                        ])
                        # Pass all golden/output pairs for multi-channel
                        for ch_idx, gpath, gcount in golden_trace_paths:
                            sim_args.extend([
                                "--golden-trace-ch",
                                f"{ch_idx}:{gpath}",
                            ])
                        for ch_idx, opath in output_trace_paths:
                            sim_args.extend([
                                "--output-trace-ch",
                                f"{ch_idx}:{opath}",
                            ])

                    # Pass plusargs to run_sim.py
                    if plusarg_list:
                        sim_args.extend(["--plusargs"] + plusarg_list)

                    passed = run_check(
                        os.path.join(scripts_dir, "run_sim.py"),
                        sim_args,
                        f"behaviour/{tc['module']}/{tc['test']}",
                        beh_dir
                    )
                    results["behaviour"]["passed" if passed else "failed"] += 1

        # Check 3: Synthesis
        if "synth" in args.checks:
            phys_dir = os.path.join(test_output, "physical")
            os.makedirs(phys_dir, exist_ok=True)
            if not find_tool("dc_shell", "synopsys/syn/W-2024.09-SP5"):
                print(f"[synth/{tc['module']}/{tc['test']}] SKIP: dc_shell not found")
                results["synth"]["skipped"] += 1
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
                        results["synth"]["failed"] += 1
                        continue
                if os.path.isdir(gen_dir):
                    synth_mod_name = extract_module_name(tc["mlir"])
                    if not synth_mod_name:
                        synth_mod_name = tc["module"]
                    synth_design = f"fabric_top_{synth_mod_name}"
                    passed = run_check(
                        os.path.join(scripts_dir, "run_synth.py"),
                        ["--rtl-dir", gen_dir, "--design-name", synth_design,
                         "--output-dir", phys_dir, "--tcl-template", tcl_template],
                        f"synth/{tc['module']}/{tc['test']}",
                        phys_dir
                    )
                    results["synth"]["passed" if passed else "failed"] += 1
                else:
                    print(f"[synth/{tc['module']}/{tc['test']}] "
                          "SKIP: gen-collateral/rtl still missing")
                    results["synth"]["skipped"] += 1

    # Per-check summary and exit decision
    print(f"\n{'='*60}")
    print("RTL Verification Summary (per-check)")
    print(f"{'='*60}")
    any_failed = False
    for check in ["gen", "behaviour", "synth"]:
        if check not in args.checks:
            continue
        r = results[check]
        print(f"  {check:12s}: passed={r['passed']} failed={r['failed']} skipped={r['skipped']}")
        if r["failed"] > 0:
            any_failed = True

    # Exit nonzero if any check has failures
    if any_failed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
