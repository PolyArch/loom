#!/usr/bin/env python3
"""Python wrapper for running Synopsys DC synthesis.

Parses arguments, invokes dc_shell with the appropriate scripts,
collects results (area, timing, power), and generates a JSON summary.

Usage:
    python run_synthesis.py --design <name> --rtl-dir <path> \\
        [--pdk saed14|asap7|saed32] [--clock-period 10.0] \\
        [--output-dir ./synth_out]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

# Add the existing RTL python path for backend_config access
RTL_PYTHON_DIR = os.path.join(REPO_ROOT, "src", "rtl", "python")
if RTL_PYTHON_DIR not in sys.path:
    sys.path.insert(0, RTL_PYTHON_DIR)

try:
    import backend_config
except ImportError:
    backend_config = None


def resolve_dc_shell():
    """Find dc_shell binary using backend_config or PATH."""
    if backend_config:
        dc = backend_config.resolve_tool("dc_shell", "synopsys/syn/W-2024.09-SP5")
        if dc:
            return dc

    dc = shutil.which("dc_shell")
    if dc:
        return dc

    # Try module loading directly
    try:
        result = subprocess.run(
            ["bash", "-c",
             "source /etc/profile.d/modules.sh && "
             "module load synopsys/syn/W-2024.09-SP5 && which dc_shell"],
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    return None


def collect_sv_files(rtl_dir, filelist=None):
    """Collect SystemVerilog source files from an RTL directory."""
    if filelist and os.path.isfile(filelist):
        base_dir = os.path.dirname(filelist)
        files = []
        with open(filelist) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//") or line.startswith("#"):
                    continue
                # Handle -incdir and other flags
                if line.startswith("-") or line.startswith("+"):
                    continue
                path = os.path.join(base_dir, line)
                if os.path.isfile(path):
                    files.append(os.path.abspath(path))
        return files

    # Walk directory for .sv files
    files = []
    for root, _, filenames in os.walk(rtl_dir):
        for fn in sorted(filenames):
            if fn.endswith(".sv"):
                files.append(os.path.join(root, fn))
    return files


def build_tcl_vars(args, sv_files):
    """Build the Tcl variable definitions for dc_shell invocation."""
    tcl_vars = []
    tcl_vars.append(f'set DESIGN_NAME "{args.design}"')
    tcl_vars.append(f'set OUTPUT_DIR "{os.path.abspath(args.output_dir)}"')
    tcl_vars.append(f'set PDK_TARGET "{args.pdk}"')
    tcl_vars.append(f'set CLOCK_PERIOD {args.clock_period}')

    # Build SV_FILES as a proper Tcl list
    files_tcl = " ".join(f'"{f}"' for f in sv_files)
    tcl_vars.append(f'set SV_FILES [list {files_tcl}]')

    if args.clock_name:
        tcl_vars.append(f'set CLOCK_NAME "{args.clock_name}"')
    if args.reset_name:
        tcl_vars.append(f'set RESET_NAME "{args.reset_name}"')
    if args.hierarchical:
        tcl_vars.append('set HIERARCHICAL 1')

    return "\n".join(tcl_vars)


def run_dc_synthesis(dc_shell, args, sv_files):
    """Run DC synthesis and return success status."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate wrapper TCL that sets variables and sources dc_compile.tcl
    tcl_vars = build_tcl_vars(args, sv_files)
    compile_script = os.path.join(SCRIPT_DIR, "dc_compile.tcl")

    wrapper_tcl = f"""{tcl_vars}
source "{compile_script}"
"""
    wrapper_path = os.path.join(args.output_dir, "synth_wrapper.tcl")
    with open(wrapper_path, "w") as f:
        f.write(wrapper_tcl)

    # Set up environment
    env = os.environ.copy()
    dc_dir = os.path.dirname(os.path.dirname(dc_shell))
    default_license = "27020@pyrito.cs.ucla.edu"

    cfg = {}
    if backend_config:
        cfg = backend_config.load_config()

    license_server = (cfg.get("synth", {}).get("license_server", "")
                      or default_license)
    env.setdefault("SNPSLMD_LICENSE_FILE", license_server)
    env.setdefault("LM_LICENSE_FILE", license_server)
    env.setdefault("SYNOPSYS_HOME", dc_dir)

    log_path = os.path.join(args.output_dir, "synth.log")
    cmd = [dc_shell, "-f", wrapper_path]

    print(f"[run_synthesis] Invoking: {' '.join(cmd)}")
    print(f"[run_synthesis] Log: {log_path}")

    with open(log_path, "w") as log_file:
        result = subprocess.run(
            cmd, stdout=log_file, stderr=subprocess.STDOUT,
            cwd=args.output_dir, env=env)

    if result.returncode != 0:
        print(f"[run_synthesis] DC exited with code {result.returncode}")
        return False

    # Verify expected outputs
    expected_files = [
        f"{args.design}_netlist.v",
        f"{args.design}.ddc",
        "area_report.rpt",
        "timing_report.rpt",
        "power_report.rpt",
    ]
    missing = [f for f in expected_files
               if not os.path.isfile(os.path.join(args.output_dir, f))]
    if missing:
        print(f"[run_synthesis] Missing output files: {missing}")
        return False

    return True


def parse_area_report(report_path):
    """Parse DC area report and extract key metrics."""
    result = {"total_area": 0.0, "cell_area": 0.0, "net_area": 0.0}
    if not os.path.isfile(report_path):
        return result

    with open(report_path) as f:
        content = f.read()

    # Parse total cell area
    m = re.search(r"Total\s+cell\s+area:\s+([\d.]+)", content)
    if m:
        result["cell_area"] = float(m.group(1))

    # Parse total net area (interconnect)
    m = re.search(r"Total\s+area:\s+([\d.]+)", content)
    if m:
        result["total_area"] = float(m.group(1))

    m = re.search(r"Net\s+Interconnect\s+area:\s+([\d.]+)", content)
    if m:
        result["net_area"] = float(m.group(1))

    return result


def parse_timing_report(report_path):
    """Parse DC timing report and extract worst slack."""
    result = {"worst_slack_ns": 0.0, "met_timing": False,
              "worst_path_from": "", "worst_path_to": ""}
    if not os.path.isfile(report_path):
        return result

    with open(report_path) as f:
        content = f.read()

    # Parse slack
    m = re.search(r"slack\s*\(?(\w*)\)?\s+([-\d.]+)", content)
    if m:
        result["worst_slack_ns"] = float(m.group(2))
        result["met_timing"] = result["worst_slack_ns"] >= 0

    # Parse start/end points
    m = re.search(r"Startpoint:\s+(\S+)", content)
    if m:
        result["worst_path_from"] = m.group(1)

    m = re.search(r"Endpoint:\s+(\S+)", content)
    if m:
        result["worst_path_to"] = m.group(1)

    return result


def parse_power_report(report_path):
    """Parse DC power report and extract key metrics."""
    result = {"total_power_mw": 0.0, "dynamic_power_mw": 0.0,
              "leakage_power_mw": 0.0, "switching_power_mw": 0.0,
              "internal_power_mw": 0.0}
    if not os.path.isfile(report_path):
        return result

    with open(report_path) as f:
        content = f.read()

    # Parse total dynamic power
    m = re.search(r"Total\s+Dynamic\s+Power\s+=\s+([\d.eE+-]+)\s*(\w+)", content)
    if m:
        value = float(m.group(1))
        unit = m.group(2)
        result["dynamic_power_mw"] = _convert_power_to_mw(value, unit)

    # Parse cell leakage power
    m = re.search(r"Cell\s+Leakage\s+Power\s+=\s+([\d.eE+-]+)\s*(\w+)", content)
    if m:
        value = float(m.group(1))
        unit = m.group(2)
        result["leakage_power_mw"] = _convert_power_to_mw(value, unit)

    result["total_power_mw"] = (result["dynamic_power_mw"]
                                + result["leakage_power_mw"])

    return result


def _convert_power_to_mw(value, unit):
    """Convert power value to milliwatts."""
    unit = unit.lower()
    multipliers = {
        "w": 1e3,
        "mw": 1.0,
        "uw": 1e-3,
        "nw": 1e-6,
        "pw": 1e-9,
    }
    return value * multipliers.get(unit, 1.0)


def generate_summary(args, success):
    """Parse DC reports and generate JSON summary."""
    summary = {
        "design_name": args.design,
        "pdk": args.pdk,
        "clock_period_ns": args.clock_period,
        "synthesis_success": success,
    }

    if success:
        area = parse_area_report(
            os.path.join(args.output_dir, "area_report.rpt"))
        timing = parse_timing_report(
            os.path.join(args.output_dir, "timing_report.rpt"))
        power = parse_power_report(
            os.path.join(args.output_dir, "power_report.rpt"))

        summary["area"] = area
        summary["timing"] = timing
        summary["power"] = power

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[run_synthesis] Summary written to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run Synopsys DC synthesis with report collection")
    parser.add_argument("--design", required=True,
                        help="Top-level design/module name")
    parser.add_argument("--rtl-dir", required=True,
                        help="Directory containing RTL source files")
    parser.add_argument("--filelist", default=None,
                        help="Explicit filelist path (default: <rtl-dir>/filelist.f)")
    parser.add_argument("--pdk", default="saed14",
                        choices=["saed14", "asap7", "saed32"],
                        help="Target PDK (default: saed14)")
    parser.add_argument("--clock-period", type=float, default=10.0,
                        help="Clock period in ns (default: 10.0)")
    parser.add_argument("--clock-name", default="clk",
                        help="Clock port name (default: clk)")
    parser.add_argument("--reset-name", default="rst_n",
                        help="Reset port name (default: rst_n)")
    parser.add_argument("--output-dir", default="./synth_out",
                        help="Output directory (default: ./synth_out)")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Use hierarchical compile (no autoungroup)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate TCL but do not invoke DC")
    args = parser.parse_args()

    # Collect source files
    filelist = args.filelist or os.path.join(args.rtl_dir, "filelist.f")
    sv_files = collect_sv_files(args.rtl_dir, filelist)
    if not sv_files:
        print(f"[run_synthesis] No .sv files found in {args.rtl_dir}")
        sys.exit(1)
    print(f"[run_synthesis] Found {len(sv_files)} source files")

    if args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
        tcl_vars = build_tcl_vars(args, sv_files)
        wrapper_path = os.path.join(args.output_dir, "synth_wrapper.tcl")
        with open(wrapper_path, "w") as f:
            f.write(tcl_vars + "\n")
            f.write(f'source "{os.path.join(SCRIPT_DIR, "dc_compile.tcl")}"\n')
        print(f"[run_synthesis] Dry run: TCL written to {wrapper_path}")
        sys.exit(0)

    # Resolve DC
    dc_shell = resolve_dc_shell()
    if not dc_shell:
        print("[run_synthesis] SKIP: dc_shell not available")
        sys.exit(0)

    print(f"[run_synthesis] Using dc_shell: {dc_shell}")
    print(f"[run_synthesis] PDK: {args.pdk}")
    print(f"[run_synthesis] Clock: {args.clock_period} ns")

    # Run synthesis
    success = run_dc_synthesis(dc_shell, args, sv_files)

    # Generate summary
    summary = generate_summary(args, success)

    if success:
        print("[run_synthesis] Synthesis completed successfully")
        if summary.get("timing", {}).get("met_timing"):
            print(f"  Slack: {summary['timing']['worst_slack_ns']:.3f} ns")
        else:
            slack = summary.get("timing", {}).get("worst_slack_ns", "N/A")
            print(f"  Timing NOT met, slack: {slack}")
    else:
        print("[run_synthesis] Synthesis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
