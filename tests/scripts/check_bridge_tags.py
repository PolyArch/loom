#!/usr/bin/env python3
"""Check that every programmed bridge add_tag value is covered by the
specific downstream memory's addr_offset_table tag range.

Reads a configured fabric MLIR file and:
1. Builds an SSA use-def graph from the MLIR text
2. For each fabric.add_tag, traces its output through temporal_sw ops
   to the consuming fabric.memory / fabric.extmemory
3. Verifies that specific memory's addr_offset_table covers the tag value

This avoids false negatives from checking against the global union of all
memory tables -- each add_tag must be covered by *its own* downstream memory.

Exit code: 0 if all tags are covered, 1 if any violation found.
"""

import re
import sys
from pathlib import Path


def parse_ssa_value(token: str) -> list[str]:
    """Parse an SSA value reference like '%345' or '%345#0' or '%345:2'.
    Returns the base SSA name(s) produced by this definition."""
    m = re.match(r'%(\w+)', token)
    if m:
        return [m.group(1)]
    return []


def build_ssa_graph(mlir_text: str) -> dict:
    """Build SSA graph: for each defined value, record op type, attributes,
    and operand SSA names.

    Returns dict keyed by SSA base name -> {op, attrs_str, operands: [str], line}
    """
    graph = {}

    for line in mlir_text.splitlines():
        stripped = line.strip()

        # Match SSA definitions: %NAME = fabric.OP or %NAME:N = fabric.OP
        m = re.match(
            r'(%\w+(?::\d+)?)\s*=\s*(fabric\.\w+)\s*(.*)', stripped)
        if not m:
            continue

        lhs = m.group(1)
        op = m.group(2)
        rest = m.group(3)

        # Extract base name from lhs
        base_m = re.match(r'%(\w+)', lhs)
        if not base_m:
            continue
        base_name = base_m.group(1)

        # Extract operand SSA references from the rest of the line.
        # Operands appear as %name or %name#index in the operand list.
        operands = re.findall(r'%(\w+)(?:#\d+)?', rest)

        graph[base_name] = {
            "op": op,
            "rest": rest,
            "operands": operands,
            "line": stripped,
        }

    return graph


def extract_tag_value(rest: str) -> int | None:
    """Extract the tag value from an add_tag op's attribute string."""
    # Boolean tag: {tag = true/false}
    m = re.search(r'\{tag\s*=\s*(true|false)\}', rest)
    if m:
        return 1 if m.group(1) == "true" else 0

    # Integer tag: {tag = 3 : i4}
    m = re.search(r'\{tag\s*=\s*(\d+)\s*:', rest)
    if m:
        return int(m.group(1))

    return None


def extract_addr_offset_table(rest: str) -> list[int] | None:
    """Extract addr_offset_table from a memory/extmemory op."""
    m = re.search(r'addr_offset_table\s*=\s*array<i64:\s*([\d,\s]+)>', rest)
    if not m:
        return None
    return [int(x.strip()) for x in m.group(1).split(",")]


def get_tag_ranges(table: list[int]) -> list[tuple[int, int]]:
    """Extract valid [start_tag, end_tag) ranges from addr_offset_table."""
    ranges = []
    for i in range(0, len(table) - 3, 4):
        valid = table[i]
        start_tag = table[i + 1]
        end_tag = table[i + 2]
        if valid:
            ranges.append((start_tag, end_tag))
    return ranges


def trace_to_memory(graph: dict, start_name: str,
                    visited: set | None = None) -> str | None:
    """Follow SSA uses from an add_tag output through temporal_sw ops
    to find the consuming memory/extmemory.

    Returns the SSA base name of the memory op, or None if not found.
    """
    if visited is None:
        visited = set()
    if start_name in visited:
        return None
    visited.add(start_name)

    # Find all ops that consume this SSA value as an operand
    for name, info in graph.items():
        if start_name not in info["operands"]:
            continue

        op = info["op"]
        if op in ("fabric.memory", "fabric.extmemory"):
            return name
        if op == "fabric.temporal_sw":
            # temporal_sw output feeds further downstream
            result = trace_to_memory(graph, name, visited)
            if result:
                return result

    return None


def check_bridge_tags(mlir_text: str) -> tuple[list[str], int, int]:
    """Check that each add_tag's value is covered by its downstream memory.

    Returns (violations, num_add_tags, num_memories).
    """
    graph = build_ssa_graph(mlir_text)
    violations = []

    # Collect add_tag and memory ops
    add_tags = {}
    memories = {}
    for name, info in graph.items():
        if info["op"] == "fabric.add_tag":
            tag_val = extract_tag_value(info["rest"])
            if tag_val is not None:
                add_tags[name] = tag_val
        elif info["op"] in ("fabric.memory", "fabric.extmemory"):
            table = extract_addr_offset_table(info["rest"])
            if table is not None:
                memories[name] = table

    if not add_tags or not memories:
        return violations, len(add_tags), len(memories)

    # Trace each add_tag to its downstream memory and validate
    for at_name, tag_val in add_tags.items():
        mem_name = trace_to_memory(graph, at_name)
        if mem_name is None:
            violations.append(
                f"add_tag %{at_name} (tag={tag_val}) has no downstream "
                f"memory/extmemory in SSA graph")
            continue

        if mem_name not in memories:
            violations.append(
                f"add_tag %{at_name} (tag={tag_val}) feeds memory "
                f"%{mem_name} which has no addr_offset_table")
            continue

        table = memories[mem_name]
        ranges = get_tag_ranges(table)
        covered = any(start <= tag_val < end for start, end in ranges)
        if not covered:
            violations.append(
                f"add_tag %{at_name} (tag={tag_val}) feeds memory "
                f"%{mem_name} whose addr_offset_table ranges {ranges} "
                f"do not cover tag value {tag_val}")

    return violations, len(add_tags), len(memories)


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
    violations, num_tags, num_mems = check_bridge_tags(mlir_text)

    if num_tags == 0:
        print(f"[SKIP] No add_tag operations found in {mlir_path.name}")
        return 0

    if num_mems == 0:
        print(f"[SKIP] No memory with addr_offset_table in {mlir_path.name}")
        return 0

    if violations:
        print(f"[FAIL] Bridge tag coverage violations in {mlir_path.name}:")
        for v in violations:
            print(f"  {v}")
        return 1

    print(f"[PASS] All {num_tags} add_tag values covered by their "
          f"downstream memory addr_offset_tables in {mlir_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
