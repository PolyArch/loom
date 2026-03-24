#!/usr/bin/env python3
"""E03: Contract Inference Quality -- compare inferred vs manual contract fields.

For each domain, loads the full-contract TDG (all fields manually specified)
and builds a minimal-contract TDG (only ordering + data_type). Invokes the
compiler's ContractInferencePass to infer the remaining fields, then compares
the inferred values against the manual ground truth.

If the tapestry_compile binary is not available, falls back to default-value
inference using ONLY the minimal inputs (ordering + data_type). The fallback
does NOT read any ground-truth fields to avoid circular inference.

Usage:
    python3 scripts/experiments/run_e03_inference.py
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Default values the compiler uses when fields are not specified.
# These are the fallback when the real compiler is unavailable.
# Crucially, these do NOT depend on any ground-truth field values.
INFERENCE_DEFAULTS = {
    "production_rate": 1024,
    "tile_shape": [1024],
    "visibility": "LOCAL_SPM",
    "double_buffering": False,
    "backpressure": "BLOCK",
}

# Fields that are always manually specified (even in minimal mode)
REQUIRED_FIELDS = {"ordering", "data_type"}

# Fields that are inferable by the compiler
INFERABLE_FIELDS = [
    "rate", "tile_shape", "visibility", "double_buffering", "backpressure",
]

# Reference TDG files with full manual contracts
DOMAINS = {
    "ai_llm": {
        "tdg": "benchmarks/tapestry/ai_llm/tdg_transformer.py",
        "label": "Transformer Layer",
    },
    "dsp_ofdm": {
        "tdg": "benchmarks/tapestry/dsp_ofdm/tdg_ofdm.py",
        "label": "OFDM Receiver",
    },
    "arvr_stereo": {
        "tdg": "benchmarks/tapestry/arvr_stereo/tdg_stereo.py",
        "label": "Stereo Vision",
    },
    "robotics_vio": {
        "tdg": "benchmarks/tapestry/robotics_vio/tdg_vio.py",
        "label": "VIO Pipeline",
    },
    "graph_analytics": {
        "tdg": "benchmarks/tapestry/graph_analytics/tdg_graph.py",
        "label": "Graph Analytics",
    },
    "zk_stark": {
        "tdg": "benchmarks/tapestry/zk_stark/tdg_stark.py",
        "label": "STARK Proof",
    },
}


def git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_tdg_contracts(tdg_path):
    """Load contracts from a Python TDG description file."""
    content = (REPO_ROOT / tdg_path).read_text()

    contracts = []
    contracts_match = re.search(
        r'contracts\s*=\s*\[(.+?)\]\s*$', content, re.DOTALL | re.MULTILINE)
    if not contracts_match:
        return contracts

    block = contracts_match.group(1)

    # Parse each contract dict
    dict_pattern = re.compile(r'\{([^}]+)\}', re.DOTALL)
    for dm in dict_pattern.finditer(block):
        contract = {}
        dict_str = dm.group(1)

        # Parse key-value pairs
        for kv in re.finditer(
                r'"(\w+)"\s*:\s*("([^"]*)"|([\d.]+)|(True|False)|'
                r'\[([^\]]*)\])', dict_str):
            key = kv.group(1)
            if kv.group(3) is not None:  # string value
                contract[key] = kv.group(3)
            elif kv.group(4) is not None:  # numeric value
                val = kv.group(4)
                contract[key] = int(val) if '.' not in val else float(val)
            elif kv.group(5) is not None:  # boolean value
                contract[key] = kv.group(5) == "True"
            elif kv.group(6) is not None:  # list value
                items = [int(x.strip()) for x in kv.group(6).split(',')
                         if x.strip()]
                contract[key] = items
        contracts.append(contract)

    return contracts


def load_tdg_kernels(tdg_path):
    """Load kernel list from a Python TDG description file."""
    content = (REPO_ROOT / tdg_path).read_text()
    kernels = []
    kernels_match = re.search(
        r'kernels\s*=\s*\[(.+?)\]', content, re.DOTALL)
    if not kernels_match:
        return kernels
    for m in re.finditer(r'"name"\s*:\s*"(\w+)"', kernels_match.group(1)):
        kernels.append(m.group(1))
    return kernels


def build_minimal_mlir(kernels, contracts):
    """Build a minimal MLIR TDG with only ordering + data_type per contract."""
    lines = []
    lines.append('module @minimal_test {')
    lines.append('  tdg.graph @minimal_test {')
    for k in kernels:
        lines.append(f'    tdg.kernel @{k} {{')
        lines.append(f'      execution_target = "CGRA",')
        lines.append(f'      source = "{k}.c",')
        lines.append(f'      function = "{k}"')
        lines.append(f'    }}')
    for c in contracts:
        producer = c.get("producer", "unknown")
        consumer = c.get("consumer", "unknown")
        ordering = c.get("ordering", "FIFO")
        data_type = c.get("data_type", "float32")
        # Map Python data_type names to MLIR type names
        mlir_type = data_type
        type_map = {
            "float32": "f32", "float64": "f64",
            "int32": "i32", "int64": "i64",
            "complex64": "complex<f32>", "complex128": "complex<f64>",
        }
        if data_type in type_map:
            mlir_type = type_map[data_type]
        lines.append(f'    tdg.contract @{producer} -> @{consumer} {{')
        lines.append(f'      ordering = "{ordering}",')
        lines.append(f'      data_type = "{mlir_type}"')
        lines.append(f'    }}')
    lines.append('  }')
    lines.append('}')
    return '\n'.join(lines)


def try_compiler_inference(kernels, contracts):
    """Try running tapestry_compile with minimal contracts to get inferred values.

    Returns a dict mapping (producer, consumer) -> {field: value} for inferred
    fields, or None if the compiler is not available or fails.
    """
    binary = REPO_ROOT / "build" / "bin" / "tapestry_compile"
    if not binary.exists():
        return None

    mlir_text = build_minimal_mlir(kernels, contracts)

    try:
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.mlir', delete=False,
                dir=str(REPO_ROOT / "out" / "experiments" / "E03")) as f:
            f.write(mlir_text)
            tmp_path = f.name

        result = subprocess.run(
            [str(binary), "--tdg", tmp_path, "--dump-inferred-contracts"],
            capture_output=True, text=True, timeout=120,
            cwd=str(REPO_ROOT)
        )

        os.unlink(tmp_path)

        if result.returncode != 0:
            return None

        # Parse inferred contract fields from compiler output
        inferred_map = {}
        current_edge = None
        for line in result.stdout.split('\n'):
            edge_m = re.match(
                r'\s*contract\s+(\w+)\s*->\s*(\w+)\s*:', line)
            if edge_m:
                current_edge = (edge_m.group(1), edge_m.group(2))
                inferred_map[current_edge] = {}
                continue
            if current_edge:
                kv_m = re.match(r'\s+(\w+)\s*=\s*(.+)', line)
                if kv_m:
                    key = kv_m.group(1)
                    val_str = kv_m.group(2).strip()
                    # Parse value
                    if val_str.startswith('['):
                        val = [int(x.strip()) for x in
                               val_str.strip('[]').split(',') if x.strip()]
                    elif val_str in ('true', 'false'):
                        val = val_str == 'true'
                    elif val_str.isdigit():
                        val = int(val_str)
                    else:
                        val = val_str.strip('"')
                    inferred_map[current_edge][key] = val

        return inferred_map if inferred_map else None

    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None


def simulate_inference_from_minimal(ordering, data_type):
    """Simulate inference using ONLY minimal inputs (ordering + data_type).

    This avoids circular inference: we do NOT read tile_shape, production_rate,
    or any other ground-truth field. The inference uses only the compiler's
    default values, which is what would happen if a user provided only the
    required fields.
    """
    inferred = {
        "ordering": ordering,
        "data_type": data_type,
    }

    # All inferable fields use fixed defaults -- no ground-truth leakage
    inferred["production_rate"] = INFERENCE_DEFAULTS["production_rate"]
    inferred["tile_shape"] = list(INFERENCE_DEFAULTS["tile_shape"])
    inferred["visibility"] = INFERENCE_DEFAULTS["visibility"]
    inferred["double_buffering"] = INFERENCE_DEFAULTS["double_buffering"]
    inferred["backpressure"] = INFERENCE_DEFAULTS["backpressure"]

    return inferred


def compute_ii_estimate(contract):
    """Estimate initiation interval from contract parameters.

    II is proportional to tile elements. This is a simplified model;
    real II comes from the mapper.
    """
    tile_shape = contract.get("tile_shape", [1024])
    tile_elements = 1
    for d in tile_shape:
        tile_elements *= d
    return max(1, tile_elements)


def compare_field(manual_val, inferred_val):
    """Compare a manual value against an inferred value."""
    if isinstance(manual_val, list) and isinstance(inferred_val, list):
        if len(manual_val) != len(inferred_val):
            return False
        return all(m == i for m, i in zip(manual_val, inferred_val))
    return manual_val == inferred_val


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E03"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "inference_quality.csv"

    rows = []
    used_compiler = False

    print("E03: Contract Inference Quality")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")
    print()

    for domain_name, info in DOMAINS.items():
        contracts = load_tdg_contracts(info["tdg"])
        kernels = load_tdg_kernels(info["tdg"])
        print(f"  {domain_name} ({info['label']}): {len(contracts)} contracts")

        # Try real compiler inference
        compiler_result = try_compiler_inference(kernels, contracts)
        if compiler_result is not None:
            used_compiler = True
            method = "compiler"
        else:
            method = "simulated"

        for contract in contracts:
            producer = contract.get("producer", "?")
            consumer = contract.get("consumer", "?")
            edge_name = f"{producer}->{consumer}"

            # Get inferred values -- either from compiler or simulation
            if compiler_result and (producer, consumer) in compiler_result:
                inferred = compiler_result[(producer, consumer)]
                # Fill in required fields
                inferred["ordering"] = contract.get("ordering", "FIFO")
                inferred["data_type"] = contract.get("data_type", "float32")
            else:
                # Use defaults only -- no ground-truth leakage
                inferred = simulate_inference_from_minimal(
                    contract.get("ordering", "FIFO"),
                    contract.get("data_type", "float32"),
                )

            # Compare inferable fields
            ii_manual = compute_ii_estimate(contract)
            ii_inferred = compute_ii_estimate(inferred)
            ii_delta_pct = (
                abs(ii_inferred - ii_manual) / ii_manual * 100.0
                if ii_manual > 0 else 0.0
            )

            for field in INFERABLE_FIELDS:
                manual_key = field
                if field == "rate":
                    manual_key = "production_rate"

                manual_val = contract.get(manual_key, "N/A")
                inferred_val = inferred.get(manual_key, "N/A")

                match = compare_field(manual_val, inferred_val)

                row = {
                    "domain": domain_name,
                    "edge": edge_name,
                    "field": field,
                    "manual_value": str(manual_val),
                    "inferred_value": str(inferred_val),
                    "match": 1 if match else 0,
                    "ii_with_manual": ii_manual,
                    "ii_with_inferred": ii_inferred,
                    "ii_delta_pct": round(ii_delta_pct, 2),
                    "method": method,
                    "git_hash": ghash,
                    "timestamp": timestamp,
                }
                rows.append(row)

            match_count = sum(
                1 for f in INFERABLE_FIELDS
                if compare_field(
                    contract.get(
                        "production_rate" if f == "rate" else f, "N/A"),
                    inferred.get(
                        "production_rate" if f == "rate" else f, "N/A")
                )
            )
            print(f"    {edge_name:40s}: "
                  f"{match_count}/{len(INFERABLE_FIELDS)} fields match "
                  f"[{method}], II delta={ii_delta_pct:.1f}%")

    # Write CSV
    fieldnames = [
        "domain", "edge", "field", "manual_value", "inferred_value",
        "match", "ii_with_manual", "ii_with_inferred", "ii_delta_pct",
        "method", "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary statistics
    total_fields = len(rows)
    matching_fields = sum(r["match"] for r in rows)
    avg_ii_delta = (
        sum(r["ii_delta_pct"] for r in rows) / len(rows) if rows else 0.0
    )

    # Per-field breakdown
    field_stats = {}
    for field in INFERABLE_FIELDS:
        field_rows = [r for r in rows if r["field"] == field]
        field_match = sum(r["match"] for r in field_rows)
        field_total = len(field_rows)
        field_stats[field] = (field_match, field_total)

    print(f"\nSummary:")
    print(f"  Total field comparisons: {total_fields}")
    print(f"  Matching fields: {matching_fields} "
          f"({matching_fields/total_fields:.1%})")
    print(f"  Average II delta: {avg_ii_delta:.1f}%")
    print(f"  Per-field breakdown:")
    for field, (match, total) in field_stats.items():
        pct = match / total * 100 if total > 0 else 0.0
        print(f"    {field:20s}: {match}/{total} ({pct:.1f}%)")

    if not used_compiler:
        print("\nNOTE: Used simulated inference (compiler binary not available")
        print("      or --dump-inferred-contracts not supported).")
        print("      Simulated data uses fixed defaults for all inferable fields.")
        print("      Results are marked as 'simulated' in the CSV provenance.")

    print(f"\nWrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
