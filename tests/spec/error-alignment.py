#!/usr/bin/env python3
"""Error code alignment test.

Checks that error codes in the spec, C++ header, and SystemVerilog header
stay in sync, and that every defined code is actually used somewhere.

Exit code: 0 if both test points pass, 1 if any fail.
"""

import re
import subprocess
import sys
from pathlib import Path

SPEC_FILE = "docs/spec-fabric-error.md"
H_FILE = "include/loom/Hardware/Common/FabricError.h"
SVH_FILE = "lib/loom/Hardware/SystemVerilog/Common/fabric_error.svh"

SEARCH_DIRS = ["lib/", "include/"]

ERROR_CODE_RE = re.compile(r"\b(CPL|CFG|RT)_[A-Z][A-Z_]*[A-Z]\b")


def extract_spec_codes(root: Path) -> set[str]:
    """Extract error codes from spec markdown table rows."""
    codes: set[str] = set()
    with open(root / SPEC_FILE) as f:
        for line in f:
            if not line.strip().startswith("|"):
                continue
            for m in ERROR_CODE_RE.finditer(line):
                codes.add(m.group())
    return codes


def extract_h_codes(root: Path) -> set[str]:
    """Extract error codes from C++ header, skipping comment lines."""
    codes: set[str] = set()
    with open(root / H_FILE) as f:
        for line in f:
            if line.strip().startswith("//"):
                continue
            for m in ERROR_CODE_RE.finditer(line):
                codes.add(m.group())
    return codes


def extract_svh_codes(root: Path) -> set[str]:
    """Extract error codes from SystemVerilog header, skipping comment lines."""
    codes: set[str] = set()
    with open(root / SVH_FILE) as f:
        for line in f:
            if line.strip().startswith("//"):
                continue
            for m in ERROR_CODE_RE.finditer(line):
                codes.add(m.group())
    return codes


CPL_ERROR_NS_RE = re.compile(r"CplError::([A-Z][A-Z_]*[A-Z])\b")


def build_usage_map(root: Path) -> dict[str, set[str]]:
    """Run ripgrep over lib/ and include/ to find all error code usages.

    Searches for:
    - Full error code strings (CFG_XXX, RT_XXX, CPL_XXX)
    - CplError::XXX namespace references (mapped back to CPL_XXX)
    """
    usage: dict[str, set[str]] = {}
    search_paths = [str(root / d) for d in SEARCH_DIRS]

    # Search for full error code strings
    cmd = [
        "rg", "-o", "--no-line-number", "--no-heading",
        r"\b(CPL|CFG|RT)_[A-Z][A-Z_]*[A-Z]\b",
    ] + search_paths
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        filepath, code = line.split(":", 1)
        code = code.strip()
        if ERROR_CODE_RE.fullmatch(code):
            usage.setdefault(code, set()).add(filepath)

    # Search for CplError::XXX namespace references (C++ usage pattern)
    cmd = [
        "rg", "-o", "--no-line-number", "--no-heading",
        r"CplError::[A-Z][A-Z_]*[A-Z]\b",
    ] + search_paths
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        filepath, ref = line.split(":", 1)
        ref = ref.strip()
        m = CPL_ERROR_NS_RE.fullmatch(ref)
        if m:
            code = "CPL_" + m.group(1)
            usage.setdefault(code, set()).add(filepath)

    return usage


def run_tp1(spec_codes: set[str], source_codes: set[str]) -> bool:
    """TP1: Spec <-> Source equivalence."""
    print("--- TP1: Spec <-> Source Equivalence ---")
    print(f"  Spec codes: {len(spec_codes)}   Source codes: {len(source_codes)}")

    only_spec = sorted(spec_codes - source_codes)
    only_source = sorted(source_codes - spec_codes)

    if not only_spec and not only_source:
        print("  [PASS]")
        return True

    print("  [FAIL]")
    if only_spec:
        print(f"  In spec but not in source ({len(only_spec)}):")
        for c in only_spec:
            print(f"    {c}")
    if only_source:
        print(f"  In source but not in spec ({len(only_source)}):")
        for c in only_source:
            print(f"    {c}")
    return False


def run_tp2(source_codes: set[str], usage_map: dict[str, set[str]],
            root: Path) -> bool:
    """TP2: Usage check -- every defined code is used somewhere."""
    print("--- TP2: Usage ---")

    definition_files = {
        str(root / H_FILE),
        str(root / SVH_FILE),
    }

    unused: list[str] = []
    for code in sorted(source_codes):
        files = usage_map.get(code, set()) - definition_files
        if not files:
            unused.append(code)

    used = len(source_codes) - len(unused)
    print(f"  {len(source_codes)} codes checked, {used} used, {len(unused)} unused")

    if not unused:
        print("  [PASS]")
        return True

    print("  [FAIL]")
    print(f"  Unused codes ({len(unused)}):")
    for c in unused:
        print(f"    {c}")
    return False


def main() -> int:
    root = Path(__file__).resolve().parent.parent.parent
    if not (root / SPEC_FILE).exists():
        print(f"error: cannot find {SPEC_FILE} from root {root}", file=sys.stderr)
        return 1

    spec_codes = extract_spec_codes(root)
    h_codes = extract_h_codes(root)
    svh_codes = extract_svh_codes(root)
    source_codes = h_codes | svh_codes

    usage_map = build_usage_map(root)

    tp1_ok = run_tp1(spec_codes, source_codes)
    print()
    tp2_ok = run_tp2(source_codes, usage_map, root)

    total = 2
    passed = sum([tp1_ok, tp2_ok])
    failed = total - passed
    print()
    print(f"ERRCODE_RESULT total={total} pass={passed} fail={failed}")

    return 0 if (tp1_ok and tp2_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
