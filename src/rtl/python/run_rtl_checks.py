#!/usr/bin/env python3
"""Master RTL verification runner. Dispatches to gen_sv, run_sim, run_synth."""

import argparse
import glob
import os
import re
import subprocess
import sys


def _env_modules_hint():
    """Return a hint string about environment-modules if available."""
    if os.path.isfile("/etc/profile.d/modules.sh"):
        return ("  Hint: environment-modules detected. Try:\n"
                "    source /etc/profile.d/modules.sh && module avail\n"
                "  to see available tool modules.")
    return ""


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
            if result.returncode == 0:
                return True
        except Exception:
            pass
    hint = _env_modules_hint()
    print(f"[tool-check] '{name}' not found in PATH.")
    if hint:
        print(hint)
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


def generate_golden_traces(loom_exec, adg_path, module_name, output_dir):
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

    cmd = [loom_exec, "--simulate", "--dfg", dfg_path,
           "--adg", adg_path, "--trace-port-dump", module_name,
           "-o", output_dir]
    print(f"[behaviour] Generating golden traces: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[behaviour] FAIL: loom golden trace generation failed")
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
    'fu_add_0', 'sw0'), not the outer fabric.module name. The simulator
    models FUs, switches, fifos, and memories -- NOT PEs as units. So for
    PE-containing modules, the trace target is the FU INSIDE the PE.

    Algorithm:
      1. Parse the whole file to collect PE definitions and their inner FU
         names (either fabric.instance with sym_name or inline
         fabric.function_unit @name).
      2. Scan the fabric.module body for ops and instances.
      3. For fabric.instance referencing a PE definition, resolve to the FU
         name inside that PE.
      4. For infrastructure instances (memory, extmemory) use sym_name.
      5. For named inline ops (fifo @name, spatial_sw @name, etc.) use @name.
      6. For anonymous inline ops (fabric.add_tag without @name) derive name
         from type + count.
      7. Priority order: FU instances from PEs > infrastructure instances >
         named inline infra ops > anonymous inline ops.
      8. Special: for anonymous inline ops, prefer map_tag over add_tag/del_tag
         (the "main" operation for map_tag test modules).

    Returns the trace target name, or None if not found.
    """
    try:
        with open(mlir_path, "r") as f:
            content = f.read()
    except OSError:
        return None

    lines = content.split('\n')

    # Patterns for parsing definitions
    pe_def_pattern = re.compile(
        r'fabric\.(?:spatial_pe|temporal_pe)\s+@(\w+)')
    fu_inst_pattern = re.compile(
        r'fabric\.instance\s+@(\w+).*\{sym_name\s*=\s*"(\w+)"')
    fu_inline_pattern = re.compile(
        r'fabric\.function_unit\s+@(\w+)')
    memory_def_pattern = re.compile(
        r'fabric\.(?:memory|extmemory)\s+@(\w+)')

    # Collect PE definitions and the FU names they contain.
    # Also collect memory/extmemory definition names so we can recognize
    # instances of them in the module body.
    pe_defs = {}          # pe_def_name -> fu_target_name
    memory_def_names = set()  # names of memory/extmemory definitions

    # Parse definitions outside fabric.module body
    in_module_body = False
    brace_depth = 0
    module_body_start = None

    for i, line in enumerate(lines):
        # Track fabric.module body boundaries
        if not in_module_body and 'fabric.module' in line:
            # Module body starts at the opening brace on or after this line
            for j in range(i, len(lines)):
                if '{' in lines[j]:
                    in_module_body = True
                    module_body_start = j
                    # Count braces from this line
                    brace_depth = lines[j].count('{') - lines[j].count('}')
                    break
            continue

        if in_module_body:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0:
                # End of module body
                break
            continue

        # Outside module body: look for PE and memory definitions
        m = pe_def_pattern.search(line)
        if m:
            pe_name = m.group(1)
            if pe_name not in pe_defs:
                pe_defs[pe_name] = None  # placeholder

        m = memory_def_pattern.search(line)
        if m:
            memory_def_names.add(m.group(1))

    # Now scan inside each PE definition for FU names.
    # PE definitions may span multiple lines before the opening '{', so
    # we track whether we have entered the body (seen the first '{').
    current_pe = None
    pe_brace_depth = 0
    pe_body_entered = False
    for line in lines:
        if current_pe is None:
            m = pe_def_pattern.search(line)
            if m and m.group(1) in pe_defs:
                current_pe = m.group(1)
                pe_brace_depth = 0
                pe_body_entered = False
                if '{' in line:
                    pe_brace_depth = line.count('{') - line.count('}')
                    pe_body_entered = True
                continue
        else:
            if not pe_body_entered:
                # Still looking for the opening brace of the PE body
                if '{' in line:
                    pe_brace_depth = line.count('{') - line.count('}')
                    pe_body_entered = True
                continue

            pe_brace_depth += line.count('{') - line.count('}')

            # Look for FU instance: fabric.instance @fu_name() {sym_name="x"}
            m = fu_inst_pattern.search(line)
            if m:
                # m.group(1) = referenced def name, m.group(2) = sym_name
                pe_defs[current_pe] = m.group(2)

            # Look for inline FU: fabric.function_unit @fu_name
            m = fu_inline_pattern.search(line)
            if m:
                if pe_defs[current_pe] is None:
                    pe_defs[current_pe] = m.group(1)

            if pe_brace_depth <= 0:
                current_pe = None

    # Now scan the fabric.module body to collect trace target candidates
    # in priority buckets.
    fu_from_pe = []       # FU names resolved from PE instances
    infra_instances = []  # memory/extmemory instance sym_names
    named_inline = []     # named inline ops (fifo @name, switch @name, etc.)
    anon_inline = []      # anonymous inline ops

    inst_pattern = re.compile(
        r'fabric\.instance\s+@(\w+).*\{sym_name\s*=\s*"(\w+)"')
    named_inline_pattern = re.compile(
        r'(fabric\.(?:fifo|spatial_sw|temporal_sw))\s+@(\w+)')
    # Memory/extmemory inline definitions (not instances) in module body
    mem_inline_pattern = re.compile(
        r'(fabric\.(?:memory|extmemory))\s+@(\w+)')
    anon_inline_pattern = re.compile(
        r'(fabric\.(?:add_tag|del_tag|map_tag|fifo|spatial_sw|temporal_sw))\b')

    anon_count = {}

    in_module_body = False
    brace_depth = 0
    for i, line in enumerate(lines):
        if not in_module_body and 'fabric.module' in line:
            for j in range(i, len(lines)):
                if '{' in lines[j]:
                    in_module_body = True
                    brace_depth = lines[j].count('{') - lines[j].count('}')
                    break
            continue

        if not in_module_body:
            continue

        brace_depth += line.count('{') - line.count('}')
        if brace_depth <= 0:
            break

        # fabric.instance with sym_name
        m = inst_pattern.search(line)
        if m:
            ref_def = m.group(1)
            sym = m.group(2)
            if ref_def in pe_defs:
                # This is a PE instance -- resolve to inner FU name
                fu_name = pe_defs[ref_def]
                if fu_name:
                    fu_from_pe.append(fu_name)
            elif ref_def in memory_def_names:
                # Memory/extmemory instance
                infra_instances.append(sym)
            else:
                # Other instance (e.g. temporal_sw instance) -- treat as infra
                infra_instances.append(sym)
            continue

        # Named inline ops: fabric.fifo @fifo_0, fabric.spatial_sw @sw0
        m = named_inline_pattern.search(line)
        if m:
            named_inline.append(m.group(2))
            continue

        # Memory/extmemory inline in module body (e.g. fabric.memory @mem_0)
        m = mem_inline_pattern.search(line)
        if m:
            infra_instances.append(m.group(2))
            continue

        # Anonymous inline ops
        m = anon_inline_pattern.search(line)
        if m:
            op_type = m.group(1)
            # Check this is not already captured as named inline
            # (named inline pattern would have matched first via continue)
            prefix_map = {
                "fabric.add_tag": "add_tag",
                "fabric.del_tag": "del_tag",
                "fabric.map_tag": "map_tag",
                "fabric.fifo": "fifo",
                "fabric.spatial_sw": "spatial_sw",
                "fabric.temporal_sw": "temporal_sw",
            }
            prefix = prefix_map.get(op_type)
            if prefix:
                idx = anon_count.get(prefix, 0)
                anon_count[prefix] = idx + 1
                anon_inline.append((prefix, f"{prefix}_{idx}"))

    # Priority: FU from PE > infra instances > named inline > anonymous inline
    if fu_from_pe:
        return fu_from_pe[0]
    if infra_instances:
        return infra_instances[0]
    if named_inline:
        return named_inline[0]
    if anon_inline:
        # For anonymous inline, prefer map_tag over add_tag/del_tag
        # (the "main" operation in map_tag test modules)
        for prefix, name in anon_inline:
            if prefix == "map_tag":
                return name
        return anon_inline[0][1]

    return None


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


def parse_dut_port_widths(gen_sv_path):
    """Parse the generated DUT SV module for per-port bit widths.

    Returns dict mapping port name to total bit width, e.g.:
    {'mod_in0': 32, 'mod_out0': 36}
    Only includes mod_in*/mod_out* data ports (excludes _valid/_ready).
    """
    widths = {}
    if not os.path.isfile(gen_sv_path):
        return widths
    try:
        with open(gen_sv_path) as f:
            for line in f:
                # Match: input/output logic[N:0] mod_inX or mod_outX
                m = re.match(
                    r'\s*(?:input|output)\s+logic\s*\[(\d+):0\]\s+(mod_(?:in|out)\d+)\s*[,)]',
                    line)
                if m:
                    widths[m.group(2)] = int(m.group(1)) + 1
                    continue
                # Match: input/output logic mod_inX (1-bit, unlikely for data)
                m = re.match(
                    r'\s*(?:input|output)\s+logic\s+(mod_(?:in|out)\d+)\s*[,)]',
                    line)
                if m and not m.group(1).endswith(('_valid', '_ready')):
                    widths[m.group(1)] = 1
    except OSError:
        pass
    return widths


def generate_dut_inst_svh(output_path, dut_module, num_inputs, num_outputs,
                          visible_input_indices=None, port_widths=None,
                          port_meta=None):
    """Generate the dut_inst.svh file with DUT port connections.

    port_meta: dict mapping port name -> (data_width, tag_width).
    When available, generates tagged port connections with explicit
    width adaptation.
    """
    if visible_input_indices is None:
        visible_input_indices = list(range(num_inputs))
    if port_widths is None:
        port_widths = {}
    if port_meta is None:
        port_meta = {}

    DATA_WIDTH = 32  # Must match tb_module_wrapper default

    lines = []
    lines.append(f"// Auto-generated DUT instantiation for {dut_module}")
    lines.append(f"// Topology: {num_inputs} visible inputs "
                 f"(indices {visible_input_indices}), {num_outputs} outputs")
    if port_widths:
        lines.append(f"// Port widths: {port_widths}")
    if port_meta:
        lines.append(f"// Port metadata: {port_meta}")
    lines.append("")

    # Fabric width adaptation (WA-5): testbench width adaptation
    # See docs/spec-rtl-width-adaptation.md
    lines.append("// Fabric width adaptation (WA-5): testbench width adaptation")
    lines.append("// See docs/spec-rtl-width-adaptation.md")
    lines.append("")

    # Declare intermediate wires for width-mismatched or tagged ports
    for drv_idx, orig_idx in enumerate(visible_input_indices):
        port_name = f"mod_in{orig_idx}"
        meta = port_meta.get(port_name)
        w = port_widths.get(port_name, DATA_WIDTH)
        if meta and meta[1] > 0:
            # Tagged input: pack {tag, data} from separate testbench arrays
            dw, tw = meta
            total = dw + tw
            lines.append(f"/* verilator lint_off WIDTHTRUNC */")
            lines.append(f"/* verilator lint_off WIDTHEXPAND */")
            lines.append(f"wire [{total-1}:0] _dut_{port_name};")
            lines.append(f"assign _dut_{port_name} = "
                         f"{{drv_tag[{drv_idx}][{tw-1}:0], drv_data[{drv_idx}][{dw-1}:0]}};")
            lines.append(f"/* verilator lint_on WIDTHEXPAND */")
            lines.append(f"/* verilator lint_on WIDTHTRUNC */")
        elif w > DATA_WIDTH:
            pad = w - DATA_WIDTH
            lines.append(f"/* verilator lint_off WIDTHEXPAND */")
            lines.append(f"wire [{w-1}:0] _dut_{port_name};")
            lines.append(f"assign _dut_{port_name} = "
                         f"{{{{{pad}{{1'b0}}}}, drv_data[{drv_idx}]}};")
            lines.append(f"/* verilator lint_on WIDTHEXPAND */")

    for o in range(num_outputs):
        port_name = f"mod_out{o}"
        meta = port_meta.get(port_name)
        w = port_widths.get(port_name, DATA_WIDTH)
        if meta and meta[1] > 0:
            dw, tw = meta
            total = dw + tw
            lines.append(f"wire [{total-1}:0] _dut_{port_name};")
        elif w > DATA_WIDTH:
            lines.append(f"wire [{w-1}:0] _dut_{port_name};")

    lines.append("")

    # Build port connection list
    ports = []
    ports.append("    .clk            (clk)")
    ports.append("    .rst_n          (rst_n)")

    # Input port connections
    for drv_idx, orig_idx in enumerate(visible_input_indices):
        port_name = f"mod_in{orig_idx}"
        meta = port_meta.get(port_name)
        w = port_widths.get(port_name, DATA_WIDTH)
        if (meta and meta[1] > 0) or w > DATA_WIDTH:
            ports.append(f"    .{port_name}        (_dut_{port_name})")
        else:
            ports.append(f"    .{port_name}        (drv_data[{drv_idx}])")
        ports.append(f"    .{port_name}_valid  (drv_valid[{drv_idx}])")
        ports.append(f"    .{port_name}_ready  (drv_ready[{drv_idx}])")

    # Output port connections
    for o in range(num_outputs):
        port_name = f"mod_out{o}"
        meta = port_meta.get(port_name)
        w = port_widths.get(port_name, DATA_WIDTH)
        if (meta and meta[1] > 0) or w > DATA_WIDTH:
            ports.append(f"    .{port_name}       (_dut_{port_name})")
        else:
            ports.append(f"    .{port_name}       (mon_data[{o}])")
        ports.append(f"    .{port_name}_valid (mon_valid[{o}])")
        ports.append(f"    .{port_name}_ready (mon_ready[{o}])")

    ports.append("    .cfg_valid      (cfg_valid)")
    ports.append("    .cfg_wdata      (cfg_wdata)")
    ports.append("    .cfg_last       (cfg_last)")
    ports.append("    .cfg_ready      (cfg_ready)")

    lines.append(f"`DUT_MODULE u_dut (")
    lines.append(",\n".join(ports))
    lines.append(");")
    lines.append("")

    # Width adaptation assigns for output ports
    lines.append("/* verilator lint_off WIDTHTRUNC */")
    lines.append("/* verilator lint_off WIDTHEXPAND */")
    for o in range(num_outputs):
        port_name = f"mod_out{o}"
        meta = port_meta.get(port_name)
        w = port_widths.get(port_name, DATA_WIDTH)
        if meta and meta[1] > 0:
            dw, tw = meta
            lines.append(f"assign mon_data[{o}] = _dut_{port_name}[{dw-1}:0];")
            lines.append(f"assign mon_tag[{o}] = _dut_{port_name}[{dw+tw-1}:{dw}];")
        elif w > DATA_WIDTH:
            lines.append(f"assign mon_data[{o}] = _dut_{port_name}[{DATA_WIDTH-1}:0];")
    lines.append("/* verilator lint_on WIDTHEXPAND */")
    lines.append("/* verilator lint_on WIDTHTRUNC */")

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


def read_trace_meta(hex_path):
    """Read a .meta file associated with a .hex trace file.

    Given a path like /dir/module_out0_tokens.hex, looks for
    /dir/module_out0_tokens.meta and returns (data_width, tag_width).
    Returns (None, None) if the .meta file does not exist or is unreadable.
    """
    meta_path = hex_path.rsplit(".hex", 1)[0] + ".meta"
    data_width = None
    tag_width = None
    try:
        with open(meta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("data_width="):
                    data_width = int(line.split("=", 1)[1])
                elif line.startswith("tag_width="):
                    tag_width = int(line.split("=", 1)[1])
    except (OSError, ValueError):
        pass
    return data_width, tag_width


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
    parser.add_argument("--loom", required=True, help="Path to loom executable")
    parser.add_argument("--test-dir", required=True, help="tests/rtl/ directory")
    parser.add_argument("--output-dir", required=True, help="Output base directory")
    parser.add_argument("--src-rtl", required=True, help="src/rtl/ directory")
    parser.add_argument("--checks", nargs="+", default=["gen", "behaviour", "synth"],
                        choices=["gen", "behaviour", "synth"])
    parser.add_argument("--modules", nargs="*", default=None,
                        help="Only run tests for these module directories (e.g., e2e)")
    parser.add_argument("--test-filter", default=None,
                        help="Only run tests matching this name (e.g., chess_2x2_stub)")
    parser.add_argument("--lib-search-path", default="",
                        help="Cell library search path for synthesis "
                             "(overrides LOOM_SYNTH_LIB_PATH env var)")
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
                ["--loom", args.loom, "--adg", tc["mlir"], "--output-dir", gen_dir],
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
                        ["--loom", args.loom, "--adg", tc["mlir"],
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
                        args.loom, tc["mlir"], trace_target,
                        test_output)
                    if trace_result:
                        golden_traces = find_golden_traces(test_output)

                if not golden_traces:
                    # Fallback: use checked-in golden trace fixtures.
                    # Infrastructure-only ADGs (add_tag, del_tag, etc.) can't
                    # produce traces via the mapper. Pre-computed fixtures in
                    # golden_traces/ provide the expected DUT port behaviour.
                    checked_in_dir = os.path.join(
                        os.path.dirname(tc["mlir"]), "golden_traces")
                    if os.path.isdir(checked_in_dir):
                        # Copy checked-in traces to the trace output dir
                        trace_out = os.path.join(test_output, "rtl-traces")
                        os.makedirs(trace_out, exist_ok=True)
                        import shutil as _shutil
                        for fname in os.listdir(checked_in_dir):
                            _shutil.copy2(
                                os.path.join(checked_in_dir, fname),
                                os.path.join(trace_out, fname))
                        golden_traces = find_golden_traces(test_output)
                        if golden_traces:
                            print(f"[behaviour/{tc['module']}/{tc['test']}] "
                                  "Using checked-in golden trace fixtures")

                if not golden_traces:
                    print(f"[behaviour/{tc['module']}/{tc['test']}] "
                          "FAIL: no golden traces (no mapper path and "
                          "no checked-in fixtures)")
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

                    # Parse per-port widths from the generated DUT SV module.
                    # When ports are wider than DATA_WIDTH (e.g. tagged ports),
                    # dut_inst.svh uses intermediate wires for adaptation.
                    gen_sv_path = os.path.join(
                        rtl_dir, "generated", dut_module + ".sv")
                    port_widths = parse_dut_port_widths(gen_sv_path)

                    # Read per-port width metadata from .meta files.
                    port_meta = {}
                    max_tag_width = 0
                    for ch_idx, trace_path in input_traces:
                        port_name = f"mod_in{visible_input_indices[ch_idx] if ch_idx < len(visible_input_indices) else ch_idx}"
                        dw, tw = read_trace_meta(trace_path)
                        if dw is not None:
                            port_meta[port_name] = (dw, tw or 0)
                            if tw and tw > max_tag_width:
                                max_tag_width = tw
                    for ch_idx, trace_path in output_traces:
                        port_name = f"mod_out{ch_idx}"
                        dw, tw = read_trace_meta(trace_path)
                        if dw is not None:
                            port_meta[port_name] = (dw, tw or 0)
                            if tw and tw > max_tag_width:
                                max_tag_width = tw

                    # Generate the DUT instantiation include file
                    dut_inst_path = os.path.join(beh_dir, "dut_inst.svh")
                    generate_dut_inst_svh(
                        dut_inst_path, dut_module,
                        num_inputs, num_outputs,
                        visible_input_indices,
                        port_widths=port_widths,
                        port_meta=port_meta)
                    print(f"[behaviour/{tc['module']}/{tc['test']}] "
                          f"Generated dut_inst.svh at {dut_inst_path}")

                    # Find config hex file and count words.
                    # First check gen-collateral, then rtl-traces (has
                    # copies of checked-in golden_traces), then fall back
                    # to checked-in golden_traces/ directly.
                    config_hex = find_config_hex(gen_dir)
                    if not config_hex:
                        config_hex = find_config_hex(trace_dir)
                    if not config_hex:
                        checked_in_cfg = os.path.join(
                            os.path.dirname(tc["mlir"]), "golden_traces")
                        config_hex = find_config_hex(checked_in_cfg)
                    if config_hex:
                        config_hex = os.path.abspath(config_hex)
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

                    if max_tag_width > 0:
                        sim_args.extend(["--tag-width", str(max_tag_width)])

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
                        ["--loom", args.loom, "--adg", tc["mlir"],
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
                    synth_args = [
                        "--rtl-dir", gen_dir, "--design-name", synth_design,
                        "--output-dir", phys_dir, "--tcl-template", tcl_template]
                    lib_path = args.lib_search_path or os.environ.get(
                        "LOOM_SYNTH_LIB_PATH", "")
                    if lib_path:
                        synth_args += ["--lib-search-path", lib_path]
                    passed = run_check(
                        os.path.join(scripts_dir, "run_synth.py"),
                        synth_args,
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
