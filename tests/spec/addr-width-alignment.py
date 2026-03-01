#!/usr/bin/env python3
"""Address bit-width alignment test.

Checks that the address bit-width constant is consistent between:
  - C++:          ADDR_BIT_WIDTH in include/loom/Hardware/Common/FabricConstants.h
  - SystemVerilog: FABRIC_ADDR_BIT_WIDTH in lib/loom/Hardware/SystemVerilog/Common/fabric_common.svh

Exit code: 0 if both values match, 1 otherwise.
"""

import re
import sys
from pathlib import Path

CPP_FILE = "include/loom/Hardware/Common/FabricConstants.h"
SVH_FILE = "lib/loom/Hardware/SystemVerilog/Common/fabric_common.svh"

CPP_RE = re.compile(r"inline\s+constexpr\s+unsigned\s+ADDR_BIT_WIDTH\s*=\s*(\d+)\s*;")
SVH_RE = re.compile(r"`define\s+FABRIC_ADDR_BIT_WIDTH\s+(\d+)")


def extract_value(root: Path, rel_path: str, pattern: re.Pattern) -> int | None:
    path = root / rel_path
    if not path.exists():
        print(f"  ERROR: {rel_path} not found", file=sys.stderr)
        return None
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return int(m.group(1))
    return None


def main() -> int:
    root = Path(__file__).resolve().parent.parent.parent

    cpp_val = extract_value(root, CPP_FILE, CPP_RE)
    svh_val = extract_value(root, SVH_FILE, SVH_RE)

    if cpp_val is None:
        print(f"[FAIL] Could not extract ADDR_BIT_WIDTH from {CPP_FILE}")
        print()
        print("ADDRWIDTH_RESULT total=1 pass=0 fail=1")
        return 1

    if svh_val is None:
        print(f"[FAIL] Could not extract FABRIC_ADDR_BIT_WIDTH from {SVH_FILE}")
        print()
        print("ADDRWIDTH_RESULT total=1 pass=0 fail=1")
        return 1

    if cpp_val == svh_val:
        print(f"[PASS] Address bit-width consistent: C++={cpp_val}, SV={svh_val}")
        print()
        print("ADDRWIDTH_RESULT total=1 pass=1 fail=0")
        return 0
    else:
        print(f"[FAIL] Address bit-width mismatch: C++={cpp_val}, SV={svh_val}")
        print(f"  {CPP_FILE}: ADDR_BIT_WIDTH = {cpp_val}")
        print(f"  {SVH_FILE}: FABRIC_ADDR_BIT_WIDTH = {svh_val}")
        print()
        print("ADDRWIDTH_RESULT total=1 pass=0 fail=1")
        return 1


if __name__ == "__main__":
    sys.exit(main())
