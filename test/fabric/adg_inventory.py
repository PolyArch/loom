#!/usr/bin/env python3
"""Emit an ADG inventory JSON artifact from generated Fabric MLIR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]


SYMBOL_PATTERN = r'"(?:\\.|[^"\\])*"|[A-Za-z_.$-][A-Za-z0-9_.$-]*'
MODULE_RE = re.compile(rf"^\s*fabric\.module @(?P<name>{SYMBOL_PATTERN})")
SYSTEM_RE = re.compile(
    rf"^\s*fabric\.system @(?P<name>{SYMBOL_PATTERN})\s+memory_model\s*="
)
SYSTEM_NODE_RE = re.compile(r'\bfabric\.node\s+@\S+\s+kind\s*=\s*"(?P<kind>[^"]+)"')
TILE_RE = re.compile(r"\bfabric\.(pe|switch|mem)\b")
SCHEDULE_RE = re.compile(r"\[(spatial|temporal)\]")
LINK_RE = re.compile(r"\bfabric\.link\b")


TOPOLOGY_MATRIX_CASES = (
    ("chain-1d", "regular", "chain_1d"),
    ("mesh-2d", "regular", "mesh_2d"),
    ("systolic-array", "regular", "systolic_array"),
    ("clustered-array", "regular", "clustered_array"),
    ("reduction-tree", "irregular", "reduction_tree"),
    ("cross-coupled-switch", "irregular", "cross_coupled_switch"),
    ("sparse-long-link", "irregular", "sparse_long_link"),
    ("heterogeneous-islands", "irregular", "heterogeneous_islands"),
)

TOPOLOGY_CLASSIFICATION_BY_RECIPE = {
    f"adg-builder::topology-{case}": (layout_class, topology_family)
    for case, layout_class, topology_family in TOPOLOGY_MATRIX_CASES
}


BUILDER_RECIPES = (
    {
        "recipe_id": "adg-builder::minimal-spatial",
        "arguments": ["--minimal-spatial"],
        "filename": "minimal-spatial.mlir",
    },
    {
        "recipe_id": "adg-builder::minimal-temporal",
        "arguments": ["--minimal-temporal"],
        "filename": "minimal-temporal.mlir",
    },
    {
        "recipe_id": "adg-builder::shared-reduction",
        "arguments": ["--shared-reduction"],
        "filename": "shared-reduction.mlir",
    },
    {
        "recipe_id": "adg-builder::shared-vector-alu",
        "arguments": ["--shared-vector-alu"],
        "filename": "shared-vector-alu.mlir",
    },
    {
        "recipe_id": "adg-builder::full-spatialcore",
        "arguments": ["--full-spatialcore"],
        "filename": "full-spatialcore.mlir",
    },
    {
        "recipe_id": "adg-builder::heterogeneous-soc",
        "arguments": ["--heterogeneous-soc"],
        "filename": "heterogeneous-soc.mlir",
    },
    *(
        {
            "recipe_id": f"adg-builder::topology-{case}",
            "arguments": ["--topology-matrix-case", case],
            "filename": f"topology-{case}.mlir",
        }
        for case, _, _ in TOPOLOGY_MATRIX_CASES
    ),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mlir-output-dir")
    return parser.parse_args(argv)


def path_reference(path: Path, anchor_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), anchor_dir.resolve())).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def loom_tool() -> str | None:
    value = os.environ.get("LOOM")
    if value:
        return value
    built = ROOT / "build" / "tools" / "loom" / "loom"
    if built.is_file():
        return str(built)
    return shutil.which("loom")


def builder_tool() -> Path | None:
    built = ROOT / "build" / "tools" / "loom-adg-builder-test" / "loom-adg-builder-test"
    if built.is_file():
        return built
    found = shutil.which("loom-adg-builder-test")
    return Path(found) if found else None


def first_diagnostic(result: subprocess.CompletedProcess[str]) -> str:
    lines = (result.stderr.strip() or result.stdout.strip()).splitlines()
    return lines[0] if lines else f"command exited {result.returncode}"


def run_loom(tool: str, input_path: Path) -> tuple[str | None, str]:
    result = subprocess.run(
        [tool, str(input_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout, ""
    return None, first_diagnostic(result)


def symbol_name(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1]
    return raw


def iter_symbol_bodies(text: str, pattern: re.Pattern[str]) -> list[tuple[str, list[str]]]:
    bodies: list[tuple[str, list[str]]] = []
    active_name: str | None = None
    active_body: list[str] = []
    brace_depth = 0

    for line in text.splitlines():
        if active_name is None:
            match = pattern.match(line)
            if match is None:
                continue
            active_name = symbol_name(match.group("name"))
            active_body = [line]
            brace_depth = line.count("{") - line.count("}")
            if brace_depth == 0:
                bodies.append((active_name, active_body))
                active_name = None
                active_body = []
            continue

        active_body.append(line)
        brace_depth += line.count("{") - line.count("}")
        if brace_depth == 0:
            bodies.append((active_name, active_body))
            active_name = None
            active_body = []

    return bodies


def module_bodies(text: str) -> list[tuple[str, list[str]]]:
    return iter_symbol_bodies(text, MODULE_RE)


def system_bodies(text: str) -> list[tuple[str, list[str]]]:
    return iter_symbol_bodies(text, SYSTEM_RE)


def classify_layout(recipe_id: str, root_kind: str, root_symbol: str) -> tuple[str, str]:
    if root_kind == "fabric.system":
        return "irregular", "heterogeneous_soc"
    if recipe_id in TOPOLOGY_CLASSIFICATION_BY_RECIPE:
        return TOPOLOGY_CLASSIFICATION_BY_RECIPE[recipe_id]
    if "minimal" in recipe_id:
        return "regular", "small_array"
    if "shared-reduction" in recipe_id:
        return "irregular", "reduction_network"
    if "shared-vector-alu" in recipe_id:
        return "irregular", "vector_alu_network"
    if "full-spatialcore" in recipe_id:
        return "irregular", "mixed_spatial_temporal"
    if "pe_" in root_symbol:
        return "regular", "pe_array"
    return "irregular", "explicit_graph"


def visual_role(body: Iterable[str]) -> str:
    text = "\n".join(body)
    if "visual" in text or "visual_layout" in text:
        return "metadata_only"
    return "absent"


def module_coverage(body: list[str]) -> dict[str, object]:
    tile_counts = {"pe": 0, "switch": 0, "mem": 0}
    schedule_kinds: set[str] = set()
    for line in body:
        tile_match = TILE_RE.search(line)
        if tile_match is not None:
            tile_counts[tile_match.group(1)] += 1
        schedule_match = SCHEDULE_RE.search(line)
        if schedule_match is not None:
            schedule_kinds.add(schedule_match.group(1))
    return {
        "node_count": sum(tile_counts.values())
        + sum(1 for line in body if re.search(r"\bfabric\.(fifo|boundary|instantiate)\b", line)),
        "link_count": sum(1 for line in body if LINK_RE.search(line)),
        "tile_kinds": sorted(kind for kind, count in tile_counts.items() if count),
        "tile_counts": tile_counts,
        "schedule_kinds": sorted(schedule_kinds),
        "node_kinds": [],
        "fu_count": sum(1 for line in body if re.search(r"\bfabric\.fu\b", line)),
        "fifo_count": sum(1 for line in body if re.search(r"\bfabric\.fifo\b", line)),
        "boundary_count": sum(1 for line in body if re.search(r"\bfabric\.boundary\b", line)),
        "instantiate_count": sum(1 for line in body if re.search(r"\bfabric\.instantiate\b", line)),
    }


def system_coverage(body: list[str]) -> dict[str, object]:
    node_kinds: set[str] = set()
    node_count = 0
    for line in body:
        match = SYSTEM_NODE_RE.search(line)
        if match is None:
            continue
        node_count += 1
        node_kinds.add(match.group("kind"))
    return {
        "node_count": node_count,
        "link_count": sum(1 for line in body if LINK_RE.search(line)),
        "tile_kinds": [],
        "tile_counts": {},
        "schedule_kinds": [],
        "node_kinds": sorted(node_kinds),
    }


def consumer_records(
    *,
    verifier_status: str,
    verifier_diagnostic: str,
    layout_class: str,
    visual_metadata_role: str,
) -> list[dict[str, str]]:
    records = [
        {
            "consumer": "fabric_verifier",
            "status": verifier_status,
            "diagnostic": verifier_diagnostic or "Fabric verifier accepted candidate",
        },
        {
            "consumer": "adg_hardware_summary",
            "status": "not_run",
            "diagnostic": "run hardware summary with --inventory to project this candidate",
        },
    ]
    if layout_class == "regular" and visual_metadata_role == "absent":
        records.append(
            {
                "consumer": "mapping_visualization",
                "status": "blocked",
                "diagnostic": "regular candidate has no visual metadata yet; coordinates remain non-semantic",
            }
        )
    return records


def candidate_record(
    *,
    recipe_id: str,
    source_path: Path,
    anchor_dir: Path,
    root_kind: str,
    root_symbol: str,
    body: list[str],
    verifier_status: str,
    verifier_diagnostic: str,
) -> dict[str, object]:
    layout_class, topology_family = classify_layout(recipe_id, root_kind, root_symbol)
    coverage = module_coverage(body) if root_kind == "fabric.module" else system_coverage(body)
    source_rel = path_reference(source_path, anchor_dir)
    config = {
        "recipe_id": recipe_id,
        "root_kind": root_kind,
        "root_symbol": root_symbol,
        "topology_family": topology_family,
    }
    metadata_role = visual_role(body)
    return {
        "candidate_id": f"{recipe_id}::{root_symbol}",
        "recipe_id": recipe_id,
        "config_id": f"{recipe_id}::default",
        "config_fingerprint": sha256_text(json.dumps(config, sort_keys=True)),
        "fabric_root": root_symbol,
        "root_kind": root_kind,
        "topology_class": "fabric_module_template" if root_kind == "fabric.module" else "fabric_system",
        "layout_class": layout_class,
        "topology_family": topology_family,
        "source_mlir": source_rel,
        "source_mlir_fingerprint": sha256_file(source_path),
        "hardware_identity": f"{source_rel}::{root_symbol}",
        "construct_coverage": coverage,
        "semantic_connectivity_source": "graph_region_ssa"
        if root_kind == "fabric.module"
        else "fabric.link",
        "visual_metadata_role": metadata_role,
        "coordinates_semantic": False,
        "verifier_status": verifier_status,
        "diagnostic": verifier_diagnostic or "Fabric verifier accepted candidate",
        "downstream_consumers": consumer_records(
            verifier_status=verifier_status,
            verifier_diagnostic=verifier_diagnostic,
            layout_class=layout_class,
            visual_metadata_role=metadata_role,
        ),
    }


def generate_recipes(out_dir: Path, diagnostics: list[str]) -> list[tuple[str, Path]]:
    tool = builder_tool()
    if tool is None:
        diagnostics.append("loom-adg-builder-test is unavailable")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[str, Path]] = []
    for recipe in BUILDER_RECIPES:
        output = out_dir / str(recipe["filename"])
        arguments = recipe["arguments"]
        result = subprocess.run(
            [str(tool), *[str(argument) for argument in arguments], "--output", str(output)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            diagnostics.append(
                f"{recipe['recipe_id']} generation failed: {first_diagnostic(result)}"
            )
            continue
        generated.append((str(recipe["recipe_id"]), output))
    return generated


def inventory_for_inputs(
    inputs: list[tuple[str, Path]],
    diagnostics: list[str],
    anchor_dir: Path,
) -> list[dict[str, object]]:
    tool = loom_tool()
    if tool is None:
        diagnostics.append("loom verifier is unavailable")
        return []
    candidates: list[dict[str, object]] = []
    for recipe_id, source_path in inputs:
        verified_text, diagnostic = run_loom(tool, source_path)
        if verified_text is None:
            candidates.append(
                {
                    "candidate_id": f"{recipe_id}::{source_path.stem}",
                    "recipe_id": recipe_id,
                    "config_id": f"{recipe_id}::default",
                    "config_fingerprint": sha256_text(recipe_id),
                    "fabric_root": "",
                    "root_kind": "unknown",
                    "topology_class": "unknown",
                    "layout_class": "irregular",
                    "topology_family": "unknown",
                    "source_mlir": path_reference(source_path, anchor_dir),
                    "source_mlir_fingerprint": sha256_file(source_path),
                    "hardware_identity": path_reference(source_path, anchor_dir),
                    "construct_coverage": {},
                    "semantic_connectivity_source": "unknown",
                    "visual_metadata_role": "absent",
                    "coordinates_semantic": False,
                    "verifier_status": "fail",
                    "diagnostic": diagnostic,
                    "downstream_consumers": consumer_records(
                        verifier_status="fail",
                        verifier_diagnostic=diagnostic,
                        layout_class="irregular",
                        visual_metadata_role="absent",
                    ),
                }
            )
            continue
        roots = [
            ("fabric.module", name, body) for name, body in module_bodies(verified_text)
        ] + [
            ("fabric.system", name, body) for name, body in system_bodies(verified_text)
        ]
        if not roots:
            diagnostics.append(f"{source_path} verified but contains no Fabric ADG root")
            continue
        for root_kind, root_symbol, body in roots:
            candidates.append(
                candidate_record(
                    recipe_id=recipe_id,
                    source_path=source_path,
                    anchor_dir=anchor_dir,
                    root_kind=root_kind,
                    root_symbol=root_symbol,
                    body=body,
                    verifier_status="pass",
                    verifier_diagnostic="",
                )
            )
    candidates.sort(key=lambda item: str(item["candidate_id"]))
    return candidates


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    mlir_dir = Path(args.mlir_output_dir) if args.mlir_output_dir else output.parent / "adg-inventory-mlir"
    anchor_dir = output.parent
    diagnostics: list[str] = []
    generated = generate_recipes(mlir_dir, diagnostics)
    candidates = inventory_for_inputs(generated, diagnostics, anchor_dir)
    inventory = {
        "schema_version": 1,
        "kind": "adg_inventory",
        "inventory_id": "adg-inventory::builder-baseline",
        "producer": "test/fabric/adg_inventory.py",
        "candidate_count": len(candidates),
        "input_artifact_fingerprints": {
            path_reference(path, anchor_dir): sha256_file(path)
            for _, path in generated
            if path.is_file()
        },
        "candidates": candidates,
        "diagnostics": diagnostics,
        "status": "pass" if candidates and not diagnostics else "blocked",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    return 0 if candidates else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
