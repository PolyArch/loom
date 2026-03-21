#!/usr/bin/env python3
"""Check 3 runner: Synopsys DC synthesis verification."""

import argparse
import os
import subprocess
import sys
import shutil


def find_dc_shell():
    """Find dc_shell executable."""
    dc = shutil.which("dc_shell")
    if dc:
        return dc
    # Try module loading
    try:
        result = subprocess.run(
            ["bash", "-c",
             "source /etc/profile.d/modules.sh && module load synopsys/syn/X-2025.06-SP3 && which dc_shell"],
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def generate_synth_tcl(rtl_dir, design_name, output_dir, tcl_template):
    """Generate a concrete synthesis TCL script from the template."""
    # Collect all SV files
    sv_files = []
    for root, _, files in os.walk(rtl_dir):
        for f in files:
            if f.endswith(".sv"):
                sv_files.append(os.path.join(root, f))

    sv_files_str = " ".join(sv_files)

    # Read template and substitute
    with open(tcl_template) as tf:
        template = tf.read()

    concrete = template.replace("${DESIGN_NAME}", design_name)
    concrete = concrete.replace("${SV_FILES}", sv_files_str)
    concrete = concrete.replace("${OUTPUT_DIR}", output_dir)

    tcl_path = os.path.join(output_dir, "synth.tcl")
    with open(tcl_path, "w") as f:
        f.write(concrete)

    return tcl_path


def run_synthesis(dc_shell, tcl_path, output_dir):
    """Run DC synthesis."""
    log_path = os.path.join(output_dir, "synth.log")
    cmd = [dc_shell, "-f", tcl_path]

    print(f"[run_synth] Running: {' '.join(cmd)}")
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                cwd=output_dir)

    if result.returncode != 0:
        print(f"[run_synth] FAIL: DC exited with code {result.returncode}")
        return False

    # Check for key report files
    area_rpt = os.path.join(output_dir, "area_report.rpt")
    timing_rpt = os.path.join(output_dir, "timing_report.rpt")

    if not os.path.exists(area_rpt) or os.path.getsize(area_rpt) == 0:
        print("[run_synth] FAIL: area report missing or empty")
        return False

    if not os.path.exists(timing_rpt) or os.path.getsize(timing_rpt) == 0:
        print("[run_synth] FAIL: timing report missing or empty")
        return False

    # Check for structural errors in log
    with open(log_path) as lf:
        log_content = lf.read()
        if "Error:" in log_content and "Warning:" not in log_content.split("Error:")[0][-100:]:
            error_lines = [l for l in log_content.split("\n") if "Error:" in l]
            print(f"[run_synth] FAIL: DC reported errors:")
            for line in error_lines[:5]:
                print(f"  {line.strip()}")
            return False

    print("[run_synth] PASS: Synthesis completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Check 3: Synthesis verification")
    parser.add_argument("--rtl-dir", required=True)
    parser.add_argument("--design-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tcl-template", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dc_shell = find_dc_shell()
    if not dc_shell:
        print("[run_synth] SKIP: dc_shell not found")
        sys.exit(0)  # Graceful skip

    tcl_path = generate_synth_tcl(args.rtl_dir, args.design_name,
                                   args.output_dir, args.tcl_template)
    if not run_synthesis(dc_shell, tcl_path, args.output_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()
