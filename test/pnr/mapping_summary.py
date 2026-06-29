#!/usr/bin/env python3
"""Emit PnR mapping summary rows from software and hardware summaries."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402
import candidate_summary_common  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--primitive-coverage")
    parser.add_argument("--hardware-summary")
    parser.add_argument("--dfg-mlir")
    parser.add_argument("--graph")
    parser.add_argument("--hardware-mlir")
    parser.add_argument("--hardware")
    parser.add_argument("--hardware-root-kind")
    parser.add_argument("--acc-core")
    parser.add_argument("--workload")
    parser.add_argument("--artifact")
    return parser.parse_args(argv)


def mapping_row(workload: str, hardware: str) -> dict[str, str]:
    return {
        "workload": workload,
        "hardware": hardware,
        "mapping_id": "",
        "placed_records": "",
        "routed_edges": "",
        "unrouted_edges": "",
        "unplaced_records": "",
        "status": "blocked",
        "diagnostic": "explicit mapper inputs are required for real PnR mapping; software and hardware candidates were discovered",
    }


def tool_candidates() -> list[Path]:
    env_tool = os.environ.get("LOOM_PNR_MAP")
    candidates = []
    if env_tool:
        candidates.append(Path(env_tool))
    candidates.extend(
        [
            ROOT / "build/tools/loom-pnr-map/loom-pnr-map",
            ROOT / "build/bin/loom-pnr-map",
        ]
    )
    return candidates


def find_tool() -> Path | None:
    for candidate in tool_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def config_tool_candidates() -> list[Path]:
    env_tool = os.environ.get("LOOM_CONFIG_TEST")
    candidates = []
    if env_tool:
        candidates.append(Path(env_tool))
    candidates.extend(
        [
            ROOT / "build/tools/loom-config-test/loom-config-test",
            ROOT / "build/bin/loom-config-test",
        ]
    )
    return candidates


def find_config_tool() -> Path | None:
    for candidate in config_tool_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def run_config_tool(*argv: str) -> str:
    tool = find_config_tool()
    if tool is None:
        raise RuntimeError("missing loom-config-test; build the config tool first")
    result = subprocess.run(
        [str(tool), *argv],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "loom-config-test failed")
    return result.stdout.strip()


def escaped_identity_part(value: str) -> str:
    escaped = []
    for byte in value.encode("utf-8"):
        char = chr(byte)
        if char.isalnum() or char == "_":
            escaped.append(char)
        else:
            escaped.append(f"%{byte:02X}")
    return "".join(escaped)


def mapping_id(workload: str, graph: str, hardware: str) -> str:
    return "__".join(
        (
            escaped_identity_part(workload),
            escaped_identity_part(graph),
            escaped_identity_part(hardware),
        )
    )


def is_system_root(args: argparse.Namespace) -> bool:
    return args.hardware_root_kind in {"system", "fabric.system"}


def hardware_identity(args: argparse.Namespace) -> str:
    if is_system_root(args):
        return f"{args.hardware}::{args.acc_core}"
    return args.hardware


def find_matching_brace(text: str, open_brace: int) -> int:
    depth = 0
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def system_region(text: str, system: str) -> str:
    match = re.search(rf"fabric\.system\s+@{re.escape(system)}\b", text)
    if match is None:
        raise RuntimeError(f"hardware MLIR lacks fabric.system @{system}")
    open_brace = text.find("{", match.end())
    if open_brace < 0:
        raise RuntimeError(f"fabric.system @{system} lacks a body")
    close_brace = find_matching_brace(text, open_brace)
    if close_brace < 0:
        raise RuntimeError(f"fabric.system @{system} has an unterminated body")
    return text[open_brace + 1 : close_brace]


def resolve_spatialcore_template(args: argparse.Namespace) -> str:
    if not args.hardware_mlir:
        raise RuntimeError("system-root unsupported mapping requires --hardware-mlir")
    if not args.acc_core:
        raise RuntimeError("system-root unsupported mapping requires --acc-core")
    text = Path(args.hardware_mlir).read_text()
    region = system_region(text, args.hardware)
    node_match = re.search(
        rf"fabric\.node\s+@{re.escape(args.acc_core)}\b(?P<body>.*?)(?=\n\s*fabric\.(?:node|link)\b|\Z)",
        region,
        flags=re.S,
    )
    if node_match is None:
        raise RuntimeError(f"fabric.system @{args.hardware} lacks acc_core @{args.acc_core}")
    body = node_match.group("body")
    if 'kind = "acc_core"' not in body:
        raise RuntimeError(f"fabric.node @{args.acc_core} is not an acc_core")
    spatial_match = re.search(r"\bspatial\s*=\s*@([A-Za-z_.$][A-Za-z0-9_.$-]*)", body)
    if spatial_match is None:
        raise RuntimeError(f"fabric.node @{args.acc_core} lacks spatial template metadata")
    return spatial_match.group(1)


def unsupported_graph_operation(output: str) -> str | None:
    match = re.search(
        r"graph contains unsupported operation for PnR mapping:\s*([^\r\n]+)",
        output,
    )
    if match is None:
        return None
    return match.group(1).strip()


def unsupported_mapping_diagnostic(output: str) -> str | None:
    operation = unsupported_graph_operation(output)
    if operation:
        return f"unsupported PnR graph operation: {operation}"
    match = re.search(
        r"(graph returns unsupported pointer value for PnR mapping)",
        output,
    )
    if match:
        return match.group(1)
    return None


def write_unsupported_mapping(args: argparse.Namespace, diagnostic: str) -> None:
    hardware = hardware_identity(args)
    map_id = mapping_id(args.workload, args.graph, hardware)
    row = {
        "workload": args.workload,
        "hardware": hardware,
        "mapping_id": map_id,
        "placed_records": "",
        "routed_edges": "",
        "unrouted_edges": "",
        "unplaced_records": "",
        "status": "unsupported",
        "diagnostic": diagnostic,
    }
    intermediate_artifacts.write_csv_rows(
        "pnr_mapping",
        intermediate_artifacts.output_path(args.output),
        [row],
    )
    if not args.artifact:
        return
    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "kind": "pnr_mapping",
        "workload": args.workload,
        "hardware": hardware,
        "graph": args.graph,
        "mapping_id": map_id,
        "config_id": "loom.default",
        "config_fingerprint": run_config_tool("--resolved-fingerprint"),
        "component_config_view": "pnr.mapping.v1",
        "component_config_fingerprint": run_config_tool(
            "--component-fingerprint",
            "--component-view",
            "pnr.mapping.v1",
        ),
        "status": "unsupported",
        "placed_records": 0,
        "routed_edges": 0,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 0,
        "placements": [],
        "routes": [],
        "unrouted_edge_details": [],
        "config_bitstream": [],
        "diagnostics": [diagnostic],
    }
    if is_system_root(args):
        spatialcore_template = resolve_spatialcore_template(args)
        artifact.update(
            {
                "hardware_root_kind": "fabric.system",
                "hardware_system": args.hardware,
                "selected_acc_core": args.acc_core or "",
                "spatialcore_template": spatialcore_template,
            }
        )
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def explicit_mapper_args(args: argparse.Namespace) -> bool:
    explicit = [
        args.dfg_mlir,
        args.graph,
        args.hardware_mlir,
        args.hardware,
        args.workload,
    ]
    optional_explicit = [args.hardware_root_kind, args.acc_core]
    if (any(explicit) or any(optional_explicit)) and not all(explicit):
        missing = [
            name
            for name, value in (
                ("--dfg-mlir", args.dfg_mlir),
                ("--graph", args.graph),
                ("--hardware-mlir", args.hardware_mlir),
                ("--hardware", args.hardware),
                ("--workload", args.workload),
            )
            if not value
        ]
        raise SystemExit(f"explicit mapper mode is missing {', '.join(missing)}")
    if args.acc_core and args.hardware_root_kind not in {"system", "fabric.system"}:
        raise SystemExit("--acc-core requires --hardware-root-kind system")
    return all(explicit)


def run_explicit_mapper(args: argparse.Namespace) -> int:
    tool = find_tool()
    if tool is None:
        sys.stderr.write("missing loom-pnr-map; build the mapper tool first\n")
        return 1
    command = [
        str(tool),
        "--dfg-mlir",
        str(Path(args.dfg_mlir)),
        "--graph",
        args.graph,
        "--hardware-mlir",
        str(Path(args.hardware_mlir)),
        "--hardware",
        args.hardware,
        "--workload",
        args.workload,
        "--output",
        args.output,
    ]
    if args.artifact:
        command.extend(["--artifact", args.artifact])
    if args.hardware_root_kind:
        command.extend(["--hardware-root-kind", args.hardware_root_kind])
    if args.acc_core:
        command.extend(["--acc-core", args.acc_core])
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        combined_output = result.stdout + result.stderr
        diagnostic = unsupported_mapping_diagnostic(combined_output)
        if diagnostic:
            try:
                write_unsupported_mapping(args, diagnostic)
            except RuntimeError as exc:
                sys.stderr.write(str(exc) + "\n")
                return 1
            return 0
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    return result.returncode


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if explicit_mapper_args(args):
        return run_explicit_mapper(args)

    output = Path(args.output)
    ignore_standard = os.environ.get("LOOM_IGNORE_STANDARD_ARTIFACTS") == "1"
    primitive_path = (
        Path(args.primitive_coverage)
        if args.primitive_coverage
        else Path()
        if ignore_standard
        else ROOT / "temp/dataflow-primitive-coverage.csv"
    )
    hardware_path = (
        Path(args.hardware_summary)
        if args.hardware_summary
        else Path()
        if ignore_standard
        else ROOT / "temp/adg-hardware-summary.csv"
    )

    workloads = candidate_summary_common.workloads_from_primitive_coverage(primitive_path)
    hardware = candidate_summary_common.hardware_from_summary(hardware_path)
    if not workloads or not hardware:
        intermediate_artifacts.write_csv("pnr_mapping", intermediate_artifacts.output_path(args.output))
        return 0

    rows = [mapping_row(workload, candidate) for workload in workloads for candidate in hardware]
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("pnr_mapping", output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
