#!/usr/bin/env python3
"""Emit ADG hardware summary rows from checked-in Fabric MLIR templates."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


DEFAULT_INPUTS = (
    ROOT / "test" / "fabric" / "unit" / "pe" / "valid.mlir",
    ROOT / "test" / "pnr" / "minimal_spatial_adg.mlir.inc",
    ROOT / "test" / "pnr" / "minimal_temporal_adg.mlir.inc",
    ROOT / "test" / "pnr" / "shared_reduction_adg.mlir",
)
ADG_BUILDER_RECIPES = {
    "test/pnr/minimal_spatial_adg.mlir.inc": "adg-builder::minimal-spatial",
    "test/pnr/minimal_temporal_adg.mlir.inc": "adg-builder::minimal-temporal",
    "test/pnr/shared_reduction_adg.mlir": "adg-builder::shared-reduction",
}
SYMBOL_PATTERN = r'"(?:\\.|[^"\\])*"|[A-Za-z_.$-][A-Za-z0-9_.$-]*'
MODULE_RE = re.compile(rf"^\s*fabric\.module @(?P<name>{SYMBOL_PATTERN})")
NODE_RE = re.compile(r"\bfabric\.(pe|switch|mem|fifo|instantiate)\b")
TILE_RE = re.compile(r"\bfabric\.(pe|switch|mem)\b")
SCHEDULE_RE = re.compile(r"\[(spatial|temporal)\]")
LINK_RE = re.compile(r"\bfabric\.link\b")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", action="append", dest="inputs", default=[])
    return parser.parse_args(argv)


def loom_tool() -> str | None:
    value = os.environ.get("LOOM")
    if value:
        return value
    built = ROOT / "build" / "tools" / "loom" / "loom"
    if built.is_file():
        return str(built)
    return shutil.which("loom")


def relative_id(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def adg_builder_recipe_identity(path: Path) -> str:
    return ADG_BUILDER_RECIPES.get(relative_id(path), "")


def first_diagnostic(result: subprocess.CompletedProcess[str]) -> str:
    lines = (result.stderr.strip() or result.stdout.strip()).splitlines()
    return lines[0] if lines else f"loom verifier exited {result.returncode}"


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


def iter_module_bodies(text: str) -> list[tuple[str, list[str]]]:
    modules: list[tuple[str, list[str]]] = []
    active_name: str | None = None
    active_body: list[str] = []
    brace_depth = 0

    for line in text.splitlines():
        if active_name is None:
            match = MODULE_RE.match(line)
            if match is None:
                continue
            active_name = symbol_name(match.group("name"))
            active_body = [line]
            brace_depth = line.count("{") - line.count("}")
            if brace_depth == 0:
                modules.append((active_name, active_body))
                active_name = None
                active_body = []
            continue

        active_body.append(line)
        brace_depth += line.count("{") - line.count("}")
        if brace_depth == 0:
            modules.append((active_name, active_body))
            active_name = None
            active_body = []

    return modules


def summarize_module(input_path: Path, name: str, body: list[str]) -> dict[str, str]:
    node_count = sum(1 for line in body if NODE_RE.search(line))
    link_count = sum(1 for line in body if LINK_RE.search(line))
    tile_kinds: set[str] = set()
    schedule_kinds: set[str] = set()
    for line in body:
        tile_match = TILE_RE.search(line)
        schedule_match = SCHEDULE_RE.search(line)
        if tile_match is None or schedule_match is None:
            continue
        tile_kinds.add(tile_match.group(1))
        schedule_kinds.add(schedule_match.group(1))
    return {
        "hardware": f"{relative_id(input_path)}::{name}",
        "topology_class": "fabric_module_template",
        "node_count": str(node_count),
        "link_count": str(link_count),
        "verify_status": "pass",
        "diagnostic": "fabric.module template verified; link_count counts explicit fabric.link records only",
        "tile_kinds": ";".join(sorted(tile_kinds)),
        "schedule_kinds": ";".join(sorted(schedule_kinds)),
        "adg_builder_recipe_identity": adg_builder_recipe_identity(input_path),
    }


def failed_row(input_path: Path, diagnostic: str) -> dict[str, str]:
    return {
        "hardware": relative_id(input_path),
        "topology_class": "fabric_module_template",
        "node_count": "0",
        "link_count": "0",
        "verify_status": "fail",
        "diagnostic": diagnostic,
        "tile_kinds": "",
        "schedule_kinds": "",
        "adg_builder_recipe_identity": adg_builder_recipe_identity(input_path),
    }


def summarize_input(tool: str, input_path: Path) -> tuple[list[dict[str, str]], bool]:
    text, diagnostic = run_loom(tool, input_path)
    if text is None:
        return [failed_row(input_path, diagnostic)], False

    modules = iter_module_bodies(text)
    if not modules:
        return [failed_row(input_path, "verified file contains no fabric.module hardware template")], False
    return [summarize_module(input_path, name, body) for name, body in modules], True


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    tool = loom_tool()
    if tool is None:
        intermediate_artifacts.write_csv("adg_hardware", intermediate_artifacts.output_path(args.output))
        return 0

    inputs = [Path(value) for value in args.inputs] if args.inputs else list(DEFAULT_INPUTS)
    rows: list[dict[str, str]] = []
    ok = True
    for input_path in inputs:
        summary_rows, input_ok = summarize_input(tool, input_path)
        rows.extend(summary_rows)
        ok = ok and input_ok

    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows("adg_hardware", output, rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
