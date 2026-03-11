#!/usr/bin/env python3
"""Viz HTML serializer validation.

Validates semantic content of .viz.html files beyond simple marker checks.
Checks adgGraph node coordinates, FU identity, DFG DOT validity,
mapping data structure, and metadata completeness.

Usage: viz_serializer_check.py <viz.html> [--temporal]
  --temporal: also validate temporal PE coordinates and FU body ops
"""
import json
import re
import sys


def extract_json_block(html, var_name):
    """Extract a JSON object assigned to a const variable in HTML."""
    # Try pattern ending with next const (works for non-last variables)
    pattern = r'const\s+' + re.escape(var_name) + r'\s*=\s*(\{.*?\});\s*\n\s*const'
    m = re.search(pattern, html, re.DOTALL)
    if not m:
        # Fallback: for the last variable before script/code
        pattern = r'const\s+' + re.escape(var_name) + r'\s*=\s*(\{.*?\});\s*\n'
        m = re.search(pattern, html, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as e:
        print(f"FAIL: {var_name} JSON parse error: {e}", file=sys.stderr)
        return None


def extract_string_block(html, var_name):
    """Extract a string literal assigned to a const variable."""
    # Try double-quoted string first (escaped newlines)
    pattern = r'const\s+' + re.escape(var_name) + r'\s*=\s*"((?:[^"\\]|\\.)*)";'
    m = re.search(pattern, html, re.DOTALL)
    if m:
        return m.group(1).replace('\\n', '\n').replace('\\"', '"')
    # Fallback: backtick template literal
    pattern = r'const\s+' + re.escape(var_name) + r'\s*=\s*`(.*?)`;\s*\n'
    m = re.search(pattern, html, re.DOTALL)
    return m.group(1) if m else None


def validate_adg_graph(adg, temporal=False):
    """Validate adgGraph JSON structure and content."""
    errors = []
    if not adg:
        return ["adgGraph is empty or missing"]

    nodes = adg.get("nodes", [])
    edges = adg.get("edges", [])
    if not nodes:
        errors.append("adgGraph has no nodes")
    if not edges:
        errors.append("adgGraph has no edges")

    # Every node must have id, name, type, class
    for n in nodes:
        for field in ("id", "name", "type", "class"):
            if field not in n:
                errors.append(f"node missing '{field}': {n.get('id', '?')}")

    # Named PE/TPE nodes (with sym_name like pe_a_R_C or tpe_X) must have
    # non-null grid coordinates
    if temporal:
        named_pe_re = re.compile(
            r'^(pe_[a-z]_\d+_\d+|tpe_[a-z](_\d+_\d+)?)$'
        )
        for n in nodes:
            name = n.get("name", "")
            if named_pe_re.match(name):
                if n.get("gridCol") is None or n.get("gridRow") is None:
                    errors.append(
                        f"temporal node '{name}' has null coordinates"
                    )

    # FU nodes in PE params must use contiguous local indices (fu_0, fu_1, ...)
    global_fu_re = re.compile(r'/fu_(\d+)$')
    for n in nodes:
        fu_nodes = n.get("params", {}).get("fuNodes", [])
        seen_indices = set()
        for fu in fu_nodes:
            fu_name = fu.get("name", "")
            m = global_fu_re.search(fu_name)
            if m:
                idx = int(m.group(1))
                seen_indices.add(idx)
        if seen_indices:
            expected = set(range(len(seen_indices)))
            if seen_indices != expected:
                errors.append(
                    f"FU indices in '{n.get('name', '?')}' are not "
                    f"contiguous 0..{len(seen_indices)-1}: "
                    f"{sorted(seen_indices)}"
                )

    # Temporal FU nodes must have semantic body ops (not generic labels)
    if temporal:
        for n in nodes:
            if "temporal_pe" in n.get("type", ""):
                fu_nodes = n.get("params", {}).get("fuNodes", [])
                for fu in fu_nodes:
                    op = fu.get("op", "")
                    if not op:
                        errors.append(
                            f"temporal FU '{fu.get('name', '?')}' "
                            f"missing op field"
                        )
                    elif op in ("fabric.pe", "fabric.temporal_pe"):
                        errors.append(
                            f"temporal FU '{fu.get('name', '?')}' "
                            f"has generic '{op}' label instead of "
                            f"semantic body-op identity"
                        )

    return errors


def validate_dfg_dot(dot_str):
    """Validate DFG DOT string."""
    errors = []
    if not dot_str:
        return ["dfgDot is empty or missing"]
    if "digraph" not in dot_str:
        errors.append("dfgDot missing 'digraph' keyword")
    if "->" not in dot_str:
        errors.append("dfgDot has no edges (missing '->')")
    # Validate stable sw_ prefixed node IDs
    sw_nodes = re.findall(r'"(sw_\d+)"', dot_str)
    if not sw_nodes:
        errors.append("dfgDot has no stable sw_ prefixed node IDs")
    # Validate stable swedge_ prefixed edge IDs
    sw_edges = re.findall(r'id="(swedge_\d+)"', dot_str)
    if not sw_edges:
        errors.append("dfgDot has no stable swedge_ prefixed edge IDs")
    return errors


def validate_mapping_data(mapping):
    """Validate mappingData JSON structure."""
    errors = []
    if not mapping:
        return ["mappingData is empty or missing"]
    if "swToHw" not in mapping:
        errors.append("mappingData missing 'swToHw'")
    if "hwToSw" not in mapping:
        errors.append("mappingData missing 'hwToSw'")
    sw_to_hw = mapping.get("swToHw", {})
    if not sw_to_hw:
        errors.append("mappingData.swToHw is empty")
    return errors


def validate_metadata(meta, label, required_fields=None):
    """Validate metadata JSON structure and representative fields."""
    errors = []
    if not meta:
        return [f"{label} is empty or missing"]
    if not isinstance(meta, dict) or len(meta) == 0:
        errors.append(f"{label} has no entries")
        return errors
    # Spot-check: first entry must contain required fields
    if required_fields:
        first_key = next(iter(meta))
        first_val = meta[first_key]
        for field in required_fields:
            if field not in first_val:
                errors.append(
                    f"{label}['{first_key}'] missing field '{field}'"
                )
    return errors


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <viz.html> [--temporal]", file=sys.stderr)
        sys.exit(2)

    viz_path = sys.argv[1]
    temporal = "--temporal" in sys.argv

    try:
        html = open(viz_path).read()
    except FileNotFoundError:
        print(f"FAIL: file not found: {viz_path}", file=sys.stderr)
        sys.exit(1)

    all_errors = []

    adg = extract_json_block(html, "adgGraph")
    all_errors.extend(validate_adg_graph(adg, temporal=temporal))

    dot = extract_string_block(html, "dfgDot")
    all_errors.extend(validate_dfg_dot(dot))

    mapping = extract_json_block(html, "mappingData")
    all_errors.extend(validate_mapping_data(mapping))

    sw_meta = extract_json_block(html, "swNodeMetadata")
    all_errors.extend(validate_metadata(
        sw_meta, "swNodeMetadata", required_fields=["op", "types"]
    ))

    hw_meta = extract_json_block(html, "hwNodeMetadata")
    all_errors.extend(validate_metadata(
        hw_meta, "hwNodeMetadata", required_fields=["name", "type", "ports"]
    ))

    if all_errors:
        print(f"FAIL: {viz_path}", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    print(f"PASS: {viz_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
