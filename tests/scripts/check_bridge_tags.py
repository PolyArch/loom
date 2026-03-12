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


def compute_expected_words(
        ld_count: int, st_count: int, num_region: int
) -> int:
    """Compute expected 32-bit word count for a bridge memory config.

    Uses the same formula as RTL/genMemoryConfig:
      tw = clog2(max(ldCount, stCount))
      bits_per_region = ADDR_BIT_WIDTH + (tw+1) + tw + 1
      total_bits = num_region * bits_per_region
      words = ceil(total_bits / 32)
    """
    is_bridge = (ld_count > 1 or st_count > 1)
    tw = 0
    if is_bridge:
        tw = 1
        while (1 << tw) < max(ld_count, st_count):
            tw += 1
    bits_per_region = ADDR_BIT_WIDTH + 1 + (tw + (tw + 1) if tw > 0 else 0)
    total_bits = num_region * bits_per_region
    return (total_bits + 31) // 32


def extract_num_region(rest: str) -> int:
    """Extract numRegion from a memory/extmemory op (default 1)."""
    m = re.search(r'numRegion\s*=\s*(\d+)', rest)
    return int(m.group(1)) if m else 1


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
    """Parse _addr.h to get {suffix: (word_offset, word_count)}.

    Matches both LOOM_ADDR_NODE_<id> and LOOM_ADDR_<SYM_NAME> forms.
    Keys are the full uppercase suffix after ``LOOM_ADDR_``/``LOOM_SIZE_``.
    """
    addrs: dict[str, list[int]] = {}
    for m in re.finditer(r'LOOM_ADDR_(\S+)\s+(\d+)', text):
        addrs.setdefault(m.group(1), [0, 0])[0] = int(m.group(2))
    for m in re.finditer(r'LOOM_SIZE_(\S+)\s+(\d+)', text):
        addrs.setdefault(m.group(1), [0, 0])[1] = int(m.group(2))
    return {k: (v[0], v[1]) for k, v in addrs.items()}


def decode_memory_config(
        words: list[int], ld_count: int, st_count: int,
        num_region: int | None = None
) -> list[tuple[int, int, int, int]]:
    """Decode config words into (valid, start_tag, end_tag, addr_offset).

    Field layout per region (low-to-high):
      addr_offset(ADDR_BIT_WIDTH), end_tag(tw+1), start_tag(tw), valid(1)
    where tw = clog2(max(ldCount, stCount)).

    If num_region is given, decode exactly that many regions (hardware sizes
    config by numRegion). Otherwise infer from total bit count.
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

    if num_region is not None:
        n_regions = num_region
    else:
        total_bits = len(words) * 32
        n_regions = total_bits // bits_per_region if bits_per_region > 0 else 0

    regions = []
    bit_pos = 0
    for _ in range(n_regions):
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


def extract_sym_name(rest: str) -> str | None:
    """Extract sym_name attribute from a memory/extmemory op."""
    m = re.search(r'sym_name\s*=\s*"([^"]+)"', rest)
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
        # ConfigGen emits LOOM_ADDR_<SYM_NAME> when sym_name is present,
        # otherwise LOOM_ADDR_NODE_<id>. Try both keys.
        node_id = extract_config_node_id(info["rest"])
        if node_id is None:
            violations.append(
                f"bridge memory %{name} has no configNodeId attribute")
            continue
        addr_key = "NODE_" + node_id
        if addr_key not in addr_map:
            sym = extract_sym_name(info["rest"])
            if sym:
                upper = sym.upper().replace('.', '_').replace('-', '_')
                if upper in addr_map:
                    addr_key = upper
        if addr_key not in addr_map:
            violations.append(
                f"bridge memory %{name} (node {node_id}) not in _addr.h")
            continue
        word_offset, word_count = addr_map[addr_key]
        num_region = extract_num_region(info["rest"])
        expected_words = compute_expected_words(ld_count, st_count, num_region)
        if word_count != expected_words:
            violations.append(
                f"bridge memory %{name}: _addr.h word count {word_count} "
                f"!= expected {expected_words} "
                f"(numRegion={num_region}, ldCount={ld_count}, "
                f"stCount={st_count})")
            continue
        byte_end = (word_offset + word_count) * 4
        if byte_end > len(config_data):
            violations.append(
                f"bridge memory %{name}: config.bin too short "
                f"(need {byte_end} bytes, have {len(config_data)})")
            continue
        words = list(struct.unpack_from(
            f'<{word_count}I', config_data, word_offset * 4))
        bin_regions = decode_memory_config(
            words, ld_count, st_count, num_region=num_region)
        # Parse MLIR table: [valid, start, end, base] * N
        mlir_regions = []
        for j in range(0, len(mlir_table) - 3, 4):
            mlir_regions.append(
                (mlir_table[j], mlir_table[j+1], mlir_table[j+2],
                 mlir_table[j+3]))
        # Require exact region count match (hardware always has numRegion).
        if len(bin_regions) != len(mlir_regions):
            violations.append(
                f"bridge memory %{name}: region count mismatch "
                f"(bin={len(bin_regions)}, mlir={len(mlir_regions)})")
            continue
        for r_idx, (m_valid, m_start, m_end, m_base) in enumerate(
                mlir_regions):
            b_valid, b_start, b_end, b_base = bin_regions[r_idx]
            if b_valid != m_valid:
                violations.append(
                    f"bridge memory %{name} region {r_idx}: "
                    f"valid mismatch (bin={b_valid}, mlir={m_valid})")
            elif m_valid:
                mismatches = []
                if b_start != m_start or b_end != m_end:
                    mismatches.append(
                        f"tags bin=[{b_start},{b_end}) "
                        f"mlir=[{m_start},{m_end})")
                if b_base != m_base:
                    mismatches.append(
                        f"addr_offset bin={b_base} mlir={m_base}")
                if mismatches:
                    violations.append(
                        f"bridge memory %{name} region {r_idx}: "
                        + ", ".join(mismatches))
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


def _build_test_config_words(
        ld_count: int, st_count: int, num_region: int,
        regions: list[tuple[int, int, int, int]]
) -> list[int]:
    """Pack region tuples (valid, start_tag, end_tag, addr_offset) into words.

    Uses the same RTL layout as genMemoryConfig / decode_memory_config.
    """
    is_bridge = (ld_count > 1 or st_count > 1)
    tw = 0
    if is_bridge:
        tw = 1
        while (1 << tw) < max(ld_count, st_count):
            tw += 1
    bits_per_region = ADDR_BIT_WIDTH + 1 + (tw + (tw + 1) if tw > 0 else 0)

    combined = 0
    bit_pos = 0
    for valid, start_tag, end_tag, addr_off in regions:
        combined |= (addr_off << bit_pos)
        bit_pos += ADDR_BIT_WIDTH
        if tw > 0:
            combined |= (end_tag << bit_pos)
            bit_pos += tw + 1
            combined |= (start_tag << bit_pos)
            bit_pos += tw
        combined |= (valid << bit_pos)
        bit_pos += 1
    # Pad remaining regions (if fewer tuples than num_region) with zeros.
    for _ in range(num_region - len(regions)):
        bit_pos += bits_per_region

    total_bits = num_region * bits_per_region
    num_words = (total_bits + 31) // 32
    return [(combined >> (32 * i)) & 0xFFFFFFFF for i in range(num_words)]


def run_self_tests() -> int:
    """Verify decode_memory_config and verify_config_binary."""
    errors = []
    ld_count, st_count, num_region = 2, 0, 2

    # --- Test 1: decode_memory_config with numRegion > active regions ---
    words = _build_test_config_words(
        ld_count, st_count, num_region,
        [(1, 0, 2, 0)])  # region 0 active, region 1 zero-padded

    regions = decode_memory_config(words, ld_count, st_count,
                                   num_region=num_region)
    if len(regions) != 2:
        errors.append(f"decode: expected 2 regions, got {len(regions)}")
    else:
        v0, s0, e0, a0 = regions[0]
        if (v0, s0, e0, a0) != (1, 0, 2, 0):
            errors.append(
                f"decode region 0: expected (1,0,2,0), got "
                f"({v0},{s0},{e0},{a0})")
        v1, s1, e1, a1 = regions[1]
        if (v1, s1, e1, a1) != (0, 0, 0, 0):
            errors.append(
                f"decode region 1: expected (0,0,0,0), got "
                f"({v1},{s1},{e1},{a1})")

    # Inferred path should match explicit.
    regions2 = decode_memory_config(words, ld_count, st_count)
    if len(regions2) != len(regions):
        errors.append(
            f"inferred region count {len(regions2)} != explicit {len(regions)}")

    # --- Synthetic fixtures for verify_config_binary ---
    # MLIR: bridge extmemory with numRegion=2, addr_offset_table covers both.
    mlir_fixture = (
        "%mem0 = fabric.extmemory "
        "[ldCount = 2, stCount = 0, numRegion = 2] "
        "addr_offset_table = array<i64: 1, 0, 2, 0, 0, 0, 0, 0>, "
        "config_node_id = 100 "
        ": memref<?xi32>")

    expected_wc = compute_expected_words(ld_count, st_count, num_region)
    config_bin = struct.pack(f'<{expected_wc}I', *words)

    # --- Test 2: verify_config_binary PASS with correct word count ---
    addr_h_ok = (
        f"#define LOOM_ADDR_NODE_100 0\n"
        f"#define LOOM_SIZE_NODE_100 {expected_wc}\n")
    v = verify_config_binary(mlir_fixture, addr_h_ok, config_bin)
    if v:
        errors.append(f"verify pass case: unexpected violations: {v}")

    # --- Test 3: verify_config_binary FAIL with truncated word count ---
    truncated_wc = expected_wc - 1
    addr_h_trunc = (
        f"#define LOOM_ADDR_NODE_100 0\n"
        f"#define LOOM_SIZE_NODE_100 {truncated_wc}\n")
    trunc_bin = config_bin[:truncated_wc * 4]
    v = verify_config_binary(mlir_fixture, addr_h_trunc, trunc_bin)
    if not v:
        errors.append("verify truncated case: expected violation but got none")
    elif "word count" not in v[0]:
        errors.append(f"verify truncated case: wrong violation: {v[0]}")

    # --- Test 4: verify_config_binary FAIL with wrong addr_offset ---
    bad_words = _build_test_config_words(
        ld_count, st_count, num_region,
        [(1, 0, 2, 42)])  # addr_offset=42 but MLIR says 0
    bad_bin = struct.pack(f'<{expected_wc}I', *bad_words)
    v = verify_config_binary(mlir_fixture, addr_h_ok, bad_bin)
    if not v:
        errors.append("verify bad addr_offset: expected violation but none")
    elif "addr_offset" not in v[0]:
        errors.append(f"verify bad addr_offset: wrong violation: {v[0]}")

    if errors:
        for e in errors:
            print(f"  SELF-TEST FAIL: {e}")
        return 1
    print("[PASS] Self-tests passed (decode + verify_config_binary)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-test":
        sys.exit(run_self_tests())
    sys.exit(main())
