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
    # Well-known installation path (avoids module load dependency)
    well_known = "/mnt/nas0/software/synopsys/syn/W-2024.09-SP5/bin/dc_shell"
    if os.path.isfile(well_known) and os.access(well_known, os.X_OK):
        return well_known
    # Try module loading as last resort
    try:
        result = subprocess.run(
            ["bash", "-c",
             "source /etc/profile.d/modules.sh && module load synopsys/syn/W-2024.09-SP5 && which dc_shell"],
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

    # Wrap each file in braces for Tcl list safety, then join with space
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
    """Run DC synthesis using the discovered dc_shell path directly."""
    log_path = os.path.join(output_dir, "synth.log")
    cmd = [dc_shell, "-f", tcl_path]

    # Set up environment: DC needs license server and library paths.
    # Derive these from the dc_shell installation path.
    env = os.environ.copy()
    dc_dir = os.path.dirname(os.path.dirname(dc_shell))  # e.g. .../syn/W-...
    license_server = "27020@pyrito.cs.ucla.edu"
    env.setdefault("SNPSLMD_LICENSE_FILE", license_server)
    env.setdefault("LM_LICENSE_FILE", license_server)
    env.setdefault("SYNOPSYS_HOME", dc_dir)
    env.setdefault("DC_HOME", dc_dir)
    lib_path = os.path.join(dc_dir, "lib")
    if os.path.isdir(lib_path):
        env["LD_LIBRARY_PATH"] = lib_path + ":" + env.get("LD_LIBRARY_PATH", "")

    print(f"[run_synth] Running: {dc_shell} -f {tcl_path}")
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                cwd=output_dir, env=env)

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
