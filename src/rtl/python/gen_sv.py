#!/usr/bin/env python3
"""Check 2 runner: invoke loom --gen-sv and validate output."""

import argparse
import os
import subprocess
import sys


def run_gen_sv(loom_exec, adg_path, output_dir, fp_ip_profile=None):
    """Run loom --gen-sv and return (success, rtl_dir)."""
    rtl_dir = os.path.join(output_dir, "rtl")
    cmd = [loom_exec, "--gen-sv", "--adg", adg_path, "-o", output_dir]
    if fp_ip_profile:
        cmd.extend(["--fp-ip-profile", fp_ip_profile])

    print(f"[gen_sv] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[gen_sv] FAIL: loom --gen-sv exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr)
        return False, rtl_dir

    if not os.path.isdir(rtl_dir):
        print(f"[gen_sv] FAIL: output directory {rtl_dir} not created")
        return False, rtl_dir

    return True, rtl_dir


def resolve_verilator():
    """Find verilator executable, trying module load if not on PATH."""
    import shutil
    if shutil.which("verilator"):
        return "verilator"
    try:
        result = subprocess.run(
            ["bash", "-c",
             "source /etc/profile.d/modules.sh && module load verilator/5.044 && which verilator"],
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "verilator"


def run_verilator_lint(rtl_dir):
    """Run verilator --lint-only on all .sv files in rtl_dir."""
    filelist = os.path.join(rtl_dir, "filelist.f")
    if not os.path.exists(filelist):
        print(f"[gen_sv] WARN: no filelist.f found in {rtl_dir}")
        return False

    verilator_exec = resolve_verilator()
    # Find the generated top module name from filelist.f
    top_module = None
    try:
        with open(filelist) as fl:
            for line in fl:
                line = line.strip()
                if line.startswith("generated/fabric_top_"):
                    top_module = line.replace("generated/", "").replace(".sv", "")
                    break
    except OSError:
        pass

    cmd = [verilator_exec, "--lint-only", "-Wall",
           "-Wno-UNUSEDSIGNAL",      # Generated wiring may have unused signals
           "-Wno-UNUSEDPARAM",       # Package constants may be unused in some modules
           "-Wno-SYNCASYNCNET",      # Mixed sync/async reset is intentional design
           "-Wno-UNDRIVEN",          # Some modules may not drive all outputs (e.g., no-config del_tag)
           "-Wno-PINMISSING",        # Instance port connections may be partial in generated code
           "-Wno-PINCONNECTEMPTY",   # Unconnected output ports use empty reference (intentional)
           "-Wno-UNSIGNED",          # Unsigned comparison in generated/prewritten RTL is intentional
           # No -Wno-fatal: any remaining warning (including unwaived WIDTHTRUNC/
           # WIDTHEXPAND) is a hard lint error.
           # See docs/spec-rtl-width-adaptation.md for the strict-by-default policy.
           "-f", filelist]
    if top_module:
        cmd.extend(["--top-module", top_module])
    print(f"[gen_sv] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=rtl_dir)
    if result.returncode != 0:
        print(f"[gen_sv] FAIL: verilator lint errors")
        if result.stderr:
            print(result.stderr[:2000])
        return False

    print("[gen_sv] PASS: verilator lint clean")
    return True


def main():
    parser = argparse.ArgumentParser(description="Check 2: Generation verification")
    parser.add_argument("--loom", required=True, help="Path to loom executable")
    parser.add_argument("--adg", required=True, help="Path to .fabric.mlir")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--fp-ip-profile", default=None)
    parser.add_argument("--skip-lint", action="store_true")
    args = parser.parse_args()

    success, rtl_dir = run_gen_sv(args.loom, args.adg, args.output_dir,
                                   args.fp_ip_profile)
    if not success:
        sys.exit(1)

    if not args.skip_lint:
        if not run_verilator_lint(rtl_dir):
            sys.exit(1)

    print("[gen_sv] All checks passed")


if __name__ == "__main__":
    main()
