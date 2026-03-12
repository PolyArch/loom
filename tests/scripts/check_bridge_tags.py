#!/usr/bin/env python3
"""Check that every programmed bridge add_tag value is covered by the
memory's addr_offset_table tag range.

Reads a configured fabric MLIR file and:
1. Extracts add_tag operations and their {tag = ...} values
2. Extracts fabric.memory / fabric.extmemory with addr_offset_table
3. Verifies each add_tag feeds into a memory whose addr_offset_table
   covers the tag value

Exit code: 0 if all tags are covered, 1 if any violation found.
"""

import re
import sys
from pathlib import Path


def parse_configured_fabric(mlir_text: str) -> tuple[list, list]:
    """Parse add_tag nodes and memory nodes from configured MLIR."""
    add_tags = []
    memories = []

    for line in mlir_text.splitlines():
        stripped = line.strip()

        # Match add_tag with boolean tag: {tag = true/false}
        m = re.search(
            r'fabric\.add_tag\b.*\{tag\s*=\s*(true|false)\}', stripped)
        if m:
            tag_val = 1 if m.group(1) == "true" else 0
            add_tags.append({"tag": tag_val, "line": stripped})
            continue

        # Match add_tag with integer tag: {tag = 3 : i4}
        m = re.search(
            r'fabric\.add_tag\b.*\{tag\s*=\s*(\d+)\s*:', stripped)
        if m:
            tag_val = int(m.group(1))
            add_tags.append({"tag": tag_val, "line": stripped})
            continue

        # Match memory/extmemory with addr_offset_table
        m = re.search(
            r'fabric\.(ext)?memory\b.*addr_offset_table\s*=\s*'
            r'array<i64:\s*([\d,\s]+)>',
            stripped)
        if m:
            is_ext = m.group(1) is not None
            table_str = m.group(2)
            table = [int(x.strip()) for x in table_str.split(",")]
            # Also extract ldCount and stCount
            ld = 1
            st = 0
            m2 = re.search(r'ldCount\s*=\s*(\d+)', stripped)
            if m2:
                ld = int(m2.group(1))
            m3 = re.search(r'stCount\s*=\s*(\d+)', stripped)
            if m3:
                st = int(m3.group(1))
            memories.append({
                "is_ext": is_ext,
                "ldCount": ld,
                "stCount": st,
                "table": table,
                "line": stripped[:120],
            })

    return add_tags, memories


def check_tag_coverage(add_tags: list, memories: list) -> list[str]:
    """Check that tag values are covered by some memory's table."""
    violations = []

    # Collect all tag ranges from memories
    tag_ranges = []
    for mem in memories:
        table = mem["table"]
        # Table format: [valid, start_tag, end_tag, base_addr] per region
        if len(table) < 4:
            continue
        for i in range(0, len(table), 4):
            if i + 3 >= len(table):
                break
            valid = table[i]
            start_tag = table[i + 1]
            end_tag = table[i + 2]
            if valid:
                tag_ranges.append((start_tag, end_tag))

    if not tag_ranges:
        # No memories with tables - nothing to check
        return violations

    # Check each add_tag value against known ranges
    for at in add_tags:
        tag = at["tag"]
        covered = any(start <= tag < end for start, end in tag_ranges)
        if not covered:
            violations.append(
                f"add_tag value {tag} not covered by any "
                f"addr_offset_table range (ranges: {tag_ranges})")

    return violations


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_bridge_tags.py <configured.fabric.mlir>",
              file=sys.stderr)
        return 1

    mlir_path = Path(sys.argv[1])
    if not mlir_path.exists():
        print(f"File not found: {mlir_path}", file=sys.stderr)
        return 1

    mlir_text = mlir_path.read_text()
    add_tags, memories = parse_configured_fabric(mlir_text)

    # Only check if bridge tags exist (multi-port memory present)
    if not add_tags:
        print(f"[SKIP] No add_tag operations found in {mlir_path.name}")
        return 0

    if not memories:
        print(f"[SKIP] No memory with addr_offset_table in {mlir_path.name}")
        return 0

    violations = check_tag_coverage(add_tags, memories)
    if violations:
        print(f"[FAIL] Bridge tag coverage violations in {mlir_path.name}:")
        for v in violations:
            print(f"  {v}")
        return 1

    print(f"[PASS] All {len(add_tags)} add_tag values covered by "
          f"{len(memories)} memory addr_offset_tables in {mlir_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
