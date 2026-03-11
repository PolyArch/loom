#!/usr/bin/env python3
"""Focused viz.html serializer regression test.

Validates exact semantic content from known small mapping fixtures.
Each assertion targets a specific reopened bug or P1 acceptance criterion.

Usage:
  viz_serializer_regression.py <viz.html> [options]

Options:
  --check-temporal-coords name:col:row[,...]
      Assert exact gridCol/gridRow for named PE/TPE nodes.

  --check-fu-identity container:fu_N:op[,...]
      Assert FU sub-nodes use local indices and have semantic op labels.

  --check-dfg-ids sw_ids edge_ids
      Assert DFG DOT contains stable sw_/swedge_ prefixed IDs.

  --check-dfg-label label
      Assert DFG DOT contains an MLIR-enriched label substring.

  --check-sw-meta-op sw_id:op[,...]
      Assert swNodeMetadata entries have expected op field values.

  --check-hw-meta-type hw_name:type[,...]
      Assert hwNodeMetadata entries have expected type field values.

  --check-temporal-mapping-funame-prefix prefix
      Assert all mappingData.temporal entries have fuName starting
      with the given prefix (not a raw global node ID).
"""
import json
import re
import sys


def extract_json_block(html, var_name):
    """Extract a JSON object from a const variable in HTML."""
    # Try pattern ending with next const (works for non-last variables)
    pattern = r'const\s+' + re.escape(var_name) + r'\s*=\s*(\{.*?\});\s*\n\s*const'
    m = re.search(pattern, html, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Fallback: for the last variable before script/code
    pattern = r'const\s+' + re.escape(var_name) + r'\s*=\s*(\{.*?\});\s*\n'
    m = re.search(pattern, html, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    return None


def extract_dot_string(html, var_name):
    """Extract a DOT string from a const variable in HTML."""
    pattern = (r'const\s+' + re.escape(var_name) +
               r'\s*=\s*"((?:[^"\\]|\\.)*)";')
    m = re.search(pattern, html, re.DOTALL)
    if m:
        return m.group(1).replace('\\n', '\n').replace('\\"', '"')
    return None


def check_temporal_coords(adg, spec):
    """Check exact grid coordinates for named nodes.
    spec: "name:col:row,name:col:row,..."
    """
    errors = []
    node_map = {n['name']: n for n in adg['nodes']}
    for item in spec.split(','):
        parts = item.strip().split(':')
        if len(parts) != 3:
            errors.append(f"bad coord spec: {item}")
            continue
        name, expected_col, expected_row = parts[0], int(parts[1]), int(parts[2])
        node = node_map.get(name)
        if not node:
            errors.append(f"node '{name}' not found in adgGraph")
            continue
        actual_col = node.get('gridCol')
        actual_row = node.get('gridRow')
        if actual_col != expected_col or actual_row != expected_row:
            errors.append(
                f"node '{name}': expected ({expected_col},{expected_row}), "
                f"got ({actual_col},{actual_row})"
            )
    return errors


def check_fu_identity(adg, spec):
    """Check FU sub-nodes have local indices and semantic op labels.
    spec: "container:fu_N:op,container:fu_N:op,..."
    """
    errors = []
    # Build container -> fuNodes map
    container_fus = {}
    for n in adg['nodes']:
        fus = n.get('params', {}).get('fuNodes', [])
        if fus:
            container_fus[n['name']] = fus

    for item in spec.split(','):
        parts = item.strip().split(':')
        if len(parts) != 3:
            errors.append(f"bad FU spec: {item}")
            continue
        container, fu_suffix, expected_op = parts
        fus = container_fus.get(container, [])
        # Find the FU by its name suffix
        expected_name = f"{container}/{fu_suffix}"
        matching = [f for f in fus if f.get('name') == expected_name]
        if not matching:
            available = [f.get('name') for f in fus]
            errors.append(
                f"FU '{expected_name}' not found. Available: {available}"
            )
            continue
        fu = matching[0]
        actual_op = fu.get('op', '')
        if expected_op not in actual_op:
            errors.append(
                f"FU '{expected_name}': expected op containing "
                f"'{expected_op}', got '{actual_op}'"
            )
        # Reject generic fabric.pe wrapper labels
        if actual_op == 'fabric.pe':
            errors.append(
                f"FU '{expected_name}': has generic 'fabric.pe' label "
                f"instead of semantic body-op identity"
            )
    return errors


def check_dfg_ids(dot, sw_spec, edge_spec):
    """Check DFG DOT contains stable sw_/swedge_ prefixed IDs.
    sw_spec: "sw_0,sw_1,..."
    edge_spec: "swedge_0,swedge_1,..."
    """
    errors = []
    for sw_id in sw_spec.split(','):
        sw_id = sw_id.strip()
        if f'"{sw_id}"' not in dot:
            errors.append(f"DFG DOT missing sw node '{sw_id}'")
    for edge_id in edge_spec.split(','):
        edge_id = edge_id.strip()
        if f'id="{edge_id}"' not in dot and f"id=\"{edge_id}\"" not in dot:
            errors.append(f"DFG DOT missing edge ID '{edge_id}'")
    return errors


def check_dfg_label(dot, label):
    """Check DFG DOT contains an MLIR-enriched label substring."""
    if label not in dot:
        return [f"DFG DOT missing label substring '{label}'"]
    return []


def check_sw_meta_op(sw_meta, spec):
    """Check swNodeMetadata entries have expected op values.
    spec: "sw_id:op,sw_id:op,..."
    """
    errors = []
    if not sw_meta:
        return ["swNodeMetadata is missing"]
    for item in spec.split(','):
        parts = item.strip().split(':')
        if len(parts) != 2:
            errors.append(f"bad sw-meta spec: {item}")
            continue
        sw_id, expected_op = parts
        entry = sw_meta.get(sw_id)
        if not entry:
            errors.append(f"swNodeMetadata missing '{sw_id}'")
            continue
        actual_op = entry.get('op', '')
        if actual_op != expected_op:
            errors.append(
                f"swNodeMetadata['{sw_id}'].op: "
                f"expected '{expected_op}', got '{actual_op}'"
            )
    return errors


def check_hw_meta_type(hw_meta, spec):
    """Check hwNodeMetadata entries have expected type values.
    spec: "hw_name:type,hw_name:type,..."
    """
    errors = []
    if not hw_meta:
        return ["hwNodeMetadata is missing"]
    # Build name -> entry map
    name_map = {}
    for k, v in hw_meta.items():
        name_map[v.get('name', '')] = v

    for item in spec.split(','):
        parts = item.strip().split(':')
        if len(parts) != 2:
            errors.append(f"bad hw-meta spec: {item}")
            continue
        hw_name, expected_type = parts
        entry = name_map.get(hw_name)
        if not entry:
            errors.append(f"hwNodeMetadata missing node named '{hw_name}'")
            continue
        actual_type = entry.get('type', '')
        if actual_type != expected_type:
            errors.append(
                f"hwNodeMetadata['{hw_name}'].type: "
                f"expected '{expected_type}', got '{actual_type}'"
            )
    return errors


def check_temporal_mapping_funame_prefix(mapping, prefix):
    """Check all temporal mapping entries have fuName with the given prefix."""
    errors = []
    temporal = mapping.get('temporal', {})
    if not temporal:
        errors.append("mappingData.temporal is empty (expected temporal mappings)")
        return errors
    for sw_id, entry in temporal.items():
        fu_name = entry.get('fuName', '')
        if not fu_name.startswith(prefix):
            errors.append(
                f"temporal['{sw_id}'].fuName '{fu_name}' "
                f"does not start with '{prefix}'"
            )
        # fuName must contain /fu_N with a small local index
        fu_match = re.search(r'/fu_(\d+)$', fu_name)
        if not fu_match:
            errors.append(
                f"temporal['{sw_id}'].fuName '{fu_name}' "
                f"missing /fu_<localIndex> suffix"
            )
        elif int(fu_match.group(1)) > 20:
            errors.append(
                f"temporal['{sw_id}'].fuName '{fu_name}' has "
                f"suspiciously large FU index (likely global, not local)"
            )
    return errors


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <viz.html> [options]", file=sys.stderr)
        sys.exit(2)

    viz_path = sys.argv[1]
    args = sys.argv[2:]

    try:
        html = open(viz_path).read()
    except FileNotFoundError:
        print(f"FAIL: file not found: {viz_path}", file=sys.stderr)
        sys.exit(1)

    # Extract all data blocks
    adg = extract_json_block(html, 'adgGraph')
    dot = extract_dot_string(html, 'dfgDot')
    mapping = extract_json_block(html, 'mappingData')
    sw_meta = extract_json_block(html, 'swNodeMetadata')
    hw_meta = extract_json_block(html, 'hwNodeMetadata')

    if not adg:
        print("FAIL: could not extract adgGraph", file=sys.stderr)
        sys.exit(1)
    if not dot:
        print("FAIL: could not extract dfgDot", file=sys.stderr)
        sys.exit(1)
    if not mapping:
        print("FAIL: could not extract mappingData", file=sys.stderr)
        sys.exit(1)

    all_errors = []
    i = 0
    while i < len(args):
        flag = args[i]
        if flag == '--check-temporal-coords' and i + 1 < len(args):
            all_errors.extend(check_temporal_coords(adg, args[i + 1]))
            i += 2
        elif flag == '--check-fu-identity' and i + 1 < len(args):
            all_errors.extend(check_fu_identity(adg, args[i + 1]))
            i += 2
        elif flag == '--check-dfg-ids' and i + 2 < len(args):
            all_errors.extend(check_dfg_ids(dot, args[i + 1], args[i + 2]))
            i += 3
        elif flag == '--check-dfg-label' and i + 1 < len(args):
            all_errors.extend(check_dfg_label(dot, args[i + 1]))
            i += 2
        elif flag == '--check-sw-meta-op' and i + 1 < len(args):
            all_errors.extend(check_sw_meta_op(sw_meta, args[i + 1]))
            i += 2
        elif flag == '--check-hw-meta-type' and i + 1 < len(args):
            all_errors.extend(check_hw_meta_type(hw_meta, args[i + 1]))
            i += 2
        elif flag == '--check-temporal-mapping-funame-prefix' and i + 1 < len(args):
            all_errors.extend(
                check_temporal_mapping_funame_prefix(mapping, args[i + 1])
            )
            i += 2
        else:
            print(f"Unknown option: {flag}", file=sys.stderr)
            sys.exit(2)

    if all_errors:
        print(f"FAIL: {viz_path}", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
