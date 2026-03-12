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
import struct
import sys
from pathlib import Path

ADDR_BIT_WIDTH = 57


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


def extract_ld_st_count(rest: str) -> tuple[int, int]:
    """Extract ldCount and stCount from a memory/extmemory op."""
    ld, st = 1, 1
    m = re.search(r'ldCount\s*=\s*(\d+)', rest)
    if m:
        ld = int(m.group(1))
    m = re.search(r'stCount\s*=\s*(\d+)', rest)
    if m:
        st = int(m.group(1))
    return ld, st


def parse_addr_header(text: str) -> dict[str, tuple[int, int]]:
    """Parse _addr.h to get {node_id: (word_offset, word_count)}."""
    addrs: dict[str, list[int]] = {}
    for m in re.finditer(r'LOOM_ADDR_NODE_(\d+)\s+(\d+)', text):
        addrs.setdefault(m.group(1), [0, 0])[0] = int(m.group(2))
    for m in re.finditer(r'LOOM_SIZE_NODE_(\d+)\s+(\d+)', text):
        addrs.setdefault(m.group(1), [0, 0])[1] = int(m.group(2))
    return {k: (v[0], v[1]) for k, v in addrs.items()}


def decode_memory_config(
        words: list[int], ld_count: int, st_count: int
) -> list[tuple[int, int, int, int]]:
    """Decode config words into (valid, start_tag, end_tag, addr_offset).

    Field layout per region (low-to-high):
      addr_offset(ADDR_BIT_WIDTH), end_tag(tw+1), start_tag(tw), valid(1)
    where tw = clog2(max(ldCount, stCount)).
    """
    is_bridge = (ld_count > 1 or st_count > 1)
    tw = 0
    if is_bridge:
        tw = 1
        while (1 << tw) < max(ld_count, st_count):
            tw += 1
    bits_per_region = ADDR_BIT_WIDTH + 1
    if tw > 0:
        bits_per_region += tw + (tw + 1)

    combined = 0
    for idx, w in enumerate(words):
        combined |= (w << (32 * idx))

    total_bits = len(words) * 32
    num_regions = total_bits // bits_per_region if bits_per_region > 0 else 0

    regions = []
    bit_pos = 0
    for _ in range(num_regions):
        addr_off = (combined >> bit_pos) & ((1 << ADDR_BIT_WIDTH) - 1)
        bit_pos += ADDR_BIT_WIDTH
        end_tag, start_tag = 0, 0
        if tw > 0:
            end_tag = (combined >> bit_pos) & ((1 << (tw + 1)) - 1)
            bit_pos += tw + 1
            start_tag = (combined >> bit_pos) & ((1 << tw) - 1)
            bit_pos += tw
        valid = (combined >> bit_pos) & 1
        bit_pos += 1
        regions.append((valid, start_tag, end_tag, addr_off))
    return regions


def extract_config_node_id(rest: str) -> str | None:
    """Extract config_node_id attribute from a memory/extmemory op."""
    m = re.search(r'config_node_id\s*=\s*(\d+)', rest)
    return m.group(1) if m else None


def verify_config_binary(mlir_text: str, addr_text: str,
                         config_data: bytes) -> list[str]:
    """Verify .config.bin matches .fabric.mlir for bridge memories."""
    graph = build_ssa_graph(mlir_text)
    addr_map = parse_addr_header(addr_text)
    violations: list[str] = []

    for name, info in graph.items():
        if info["op"] not in ("fabric.memory", "fabric.extmemory"):
            continue
        ld_count, st_count = extract_ld_st_count(info["rest"])
        if ld_count <= 1 and st_count <= 1:
            continue
        mlir_table = extract_addr_offset_table(info["rest"])
        if mlir_table is None:
            continue
        # Use configNodeId attribute to correlate with _addr.h.
        node_id = extract_config_node_id(info["rest"])
        if node_id is None:
            violations.append(
                f"bridge memory %{name} has no configNodeId attribute")
            continue
        if node_id not in addr_map:
            violations.append(
                f"bridge memory %{name} (node {node_id}) not in _addr.h")
            continue
        word_offset, word_count = addr_map[node_id]
        if word_count == 0:
            violations.append(f"bridge memory %{name} has size 0 in _addr.h")
            continue
        byte_end = (word_offset + word_count) * 4
        if byte_end > len(config_data):
            violations.append(
                f"bridge memory %{name}: config.bin too short "
                f"(need {byte_end} bytes, have {len(config_data)})")
            continue
        words = list(struct.unpack_from(
            f'<{word_count}I', config_data, word_offset * 4))
        bin_regions = decode_memory_config(words, ld_count, st_count)
        # Parse MLIR table: [valid, start, end, base] * N
        mlir_regions = []
        for j in range(0, len(mlir_table) - 3, 4):
            mlir_regions.append(
                (mlir_table[j], mlir_table[j+1], mlir_table[j+2],
                 mlir_table[j+3]))
        for r_idx, (m_valid, m_start, m_end, _) in enumerate(mlir_regions):
            if r_idx >= len(bin_regions):
                if m_valid:
                    violations.append(
                        f"bridge memory %{name} region {r_idx}: "
                        f"MLIR valid but absent in config.bin")
                continue
            b_valid, b_start, b_end, _ = bin_regions[r_idx]
            if b_valid != m_valid:
                violations.append(
                    f"bridge memory %{name} region {r_idx}: "
                    f"valid mismatch (bin={b_valid}, mlir={m_valid})")
            elif m_valid and (b_start != m_start or b_end != m_end):
                violations.append(
                    f"bridge memory %{name} region {r_idx}: "
                    f"tag range mismatch "
                    f"(bin=[{b_start},{b_end}), mlir=[{m_start},{m_end}))")
    return violations


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
        print("Usage: check_bridge_tags.py <configured.fabric.mlir> "
              "[addr.h] [config.bin]", file=sys.stderr)
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

    # Optional config binary verification.
    if len(sys.argv) >= 4:
        addr_path = Path(sys.argv[2])
        config_path = Path(sys.argv[3])
        for p in (addr_path, config_path):
            if not p.exists():
                print(f"File not found: {p}", file=sys.stderr)
                return 1
        cfg_violations = verify_config_binary(
            mlir_text, addr_path.read_text(), config_path.read_bytes())
        if cfg_violations:
            print(f"[FAIL] Config binary violations in {mlir_path.name}:")
            for v in cfg_violations:
                print(f"  {v}")
            return 1
        print(f"[PASS] Config binary matches MLIR for bridge memories "
              f"in {mlir_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
