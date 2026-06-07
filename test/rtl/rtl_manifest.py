#!/usr/bin/env python3
"""Emit a deterministic RTL manifest and minimal SystemVerilog source."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hardware-summary")
    return parser.parse_args(argv)


def module_name(hardware_identity: str) -> str:
    raw_name = hardware_identity.rsplit("::", 1)[-1]
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)
    if not sanitized:
        return "loom_rtl_top"
    if sanitized[0].isdigit():
        return f"loom_{sanitized}"
    return sanitized


def pass_hardware_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("verify_status") != "pass" or not row.get("hardware"):
                continue
            parsed_counts = []
            for key in ("node_count", "link_count"):
                try:
                    value = int(row.get(key, ""))
                except ValueError:
                    parsed_counts = []
                    break
                if value < 0:
                    parsed_counts = []
                    break
                parsed_counts.append((key, str(value)))
            if len(parsed_counts) != 2:
                continue
            normalized = dict(row)
            for key, value in parsed_counts:
                normalized[key] = value
            rows.append(normalized)
    return rows


def sv_source(module: str, node_count: str, link_count: str) -> str:
    return (
        "`timescale 1ns/1ps\n"
        f"module {module}(\n"
        "  input logic clk,\n"
        "  input logic rst_n\n"
        ");\n"
        f"  localparam int LOOM_NODE_COUNT = {node_count};\n"
        f"  localparam int LOOM_LINK_COUNT = {link_count};\n"
        "endmodule\n"
    )


def build_manifest(output: Path, hardware_row: dict[str, str]) -> dict[str, object]:
    hardware_identity = hardware_row["hardware"]
    top_module = module_name(hardware_identity)
    source_relative = Path("rtl") / f"{top_module}.sv"
    source_path = output.parent / source_relative
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        sv_source(
            top_module,
            hardware_row.get("node_count", "0") or "0",
            hardware_row.get("link_count", "0") or "0",
        )
    )
    return {
        "schema_version": 1,
        "kind": "rtl_manifest",
        "manifest_id": f"rtl-manifest::{top_module}",
        "source_fabric_adg_identity": hardware_identity,
        "mapping_artifact_identity": "",
        "lowering_configuration": {
            "lowering_kind": "architecture_rtl",
            "source_root_kind": "fabric_adg",
            "systemverilog_profile": "behavioral_shell_v1",
        },
        "emitted_source_files": [
            {
                "path": source_relative.as_posix(),
                "language": "systemverilog",
                "fingerprint": intermediate_artifacts.artifact_fingerprint(source_path),
            }
        ],
        "top_level_modules": [top_module],
        "generated_packages": [],
        "generated_interfaces": [],
        "black_box_modules": [],
        "behavioral_models": ["behavioral_fabric_module_shell"],
        "required_tool_capability_classes": ["rtl_lint"],
        "required_library_profile_classes": [],
        "constraints": [],
        "activity_hooks": [
            {
                "source": "rtl_signal_names",
                "top_level_module": top_module,
            }
        ],
        "diagnostics": [],
        "status": "pass",
    }


def scaffold(output: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "rtl_manifest",
        "manifest_id": "rtl-manifest::blocked",
        "source_fabric_adg_identity": "",
        "mapping_artifact_identity": "",
        "lowering_configuration": {},
        "emitted_source_files": [],
        "top_level_modules": [],
        "generated_packages": [],
        "generated_interfaces": [],
        "black_box_modules": [],
        "behavioral_models": [],
        "required_tool_capability_classes": [],
        "required_library_profile_classes": [],
        "constraints": [],
        "activity_hooks": [],
        "diagnostics": [
            {
                "diagnostic_class": "missing_fabric_adg",
                "message": "no passing ADG hardware summary row was provided",
            }
        ],
        "status": "blocked",
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hardware_summary = Path(args.hardware_summary) if args.hardware_summary else Path()
    rows = pass_hardware_rows(hardware_summary)
    manifest = build_manifest(output, sorted(rows, key=lambda row: row["hardware"])[0]) if rows else scaffold(output)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
