#!/usr/bin/env python3
"""Emit a deterministic RTL manifest and minimal SystemVerilog source."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


SYMBOL_PATTERN = r'"(?:\\.|[^"\\])*"|[A-Za-z_.$-][A-Za-z0-9_.$-]*'


@dataclass(frozen=True)
class ScalarPort:
    name: str
    direction: str
    fabric_type: str
    width: int

    @property
    def systemverilog_type(self) -> str:
        if self.width == 1:
            return "logic"
        return f"logic [{self.width - 1}:0]"

    def manifest_entry(self) -> dict[str, object]:
        return {
            "name": self.name,
            "direction": self.direction,
            "fabric_type": self.fabric_type,
            "systemverilog_type": self.systemverilog_type,
        }


@dataclass(frozen=True)
class MappingEvidence:
    identity: str
    workload: str
    hardware: str
    mapping_id: str


class InterfaceLoweringError(ValueError):
    pass


class MappingArtifactError(ValueError):
    def __init__(self, diagnostic_class: str, message: str):
        super().__init__(message)
        self.diagnostic_class = diagnostic_class


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hardware-summary")
    parser.add_argument("--mapping-artifact")
    return parser.parse_args(argv)


def module_name(hardware_identity: str) -> str:
    raw_name = hardware_identity.rsplit("::", 1)[-1]
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)
    if not sanitized:
        return "loom_rtl_top"
    if sanitized[0].isdigit():
        return f"loom_{sanitized}"
    return sanitized


def string_field(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    return value if isinstance(value, str) else ""


def hardware_identity_matches(candidate: str, hardware: str) -> bool:
    return intermediate_artifacts.hardware_identity_matches(candidate, hardware)


def hardware_matches_mapping_record(hardware_identity: str, mapping: dict[str, object]) -> bool:
    hardware = string_field(mapping, "hardware")
    return hardware_identity_matches(
        hardware_identity,
        hardware,
    ) or intermediate_artifacts.mapping_system_child_matches(hardware_identity, mapping)


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


def source_root(hardware_identity: str, source_base: Path = ROOT) -> tuple[Path, str]:
    if "::" not in hardware_identity:
        raise InterfaceLoweringError("hardware identity must include source path and module symbol")
    raw_path, symbol = hardware_identity.rsplit("::", 1)
    if not raw_path or not symbol:
        raise InterfaceLoweringError("hardware identity must include source path and module symbol")
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [source_base / path, ROOT / path]
    resolved_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if resolved_path is None:
        raise InterfaceLoweringError(f"Fabric source file does not exist: {raw_path}")
    return resolved_path, symbol


def split_top_level_commas(raw: str) -> list[str]:
    entries: list[str] = []
    start = 0
    angle_depth = 0
    paren_depth = 0
    bracket_depth = 0
    for index, char in enumerate(raw):
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
        elif char == "," and not angle_depth and not paren_depth and not bracket_depth:
            entry = raw[start:index].strip()
            if entry:
                entries.append(entry)
            start = index + 1
    tail = raw[start:].strip()
    if tail:
        entries.append(tail)
    return entries


def sanitize_port_name(raw: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if not base:
        base = "port"
    if base[0].isdigit():
        base = f"port_{base}"
    if base in {"clk", "rst_n", "module", "input", "output", "logic", "endmodule"}:
        base = f"{base}_port"
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def scalar_bits_width(fabric_type: str) -> int | None:
    match = re.fullmatch(r"!fabric\.bits<([0-9]+)>", fabric_type.strip())
    if not match:
        return None
    width = int(match.group(1))
    if width <= 0:
        return None
    return width


def unsupported_boundary_diagnostic(name: str, direction: str, fabric_type: str) -> dict[str, str]:
    return {
        "diagnostic_class": "unsupported_rtl_boundary_type",
        "message": (
            f"{direction} boundary port {name} has unsupported RTL boundary type {fabric_type}"
        ),
    }


def mlir_symbol_name(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1]
    return raw


def find_module_signature(text: str, symbol: str) -> tuple[str, str]:
    pattern = re.compile(
        r"fabric\.module\s+@(?P<symbol>"
        + SYMBOL_PATTERN
        + r")"
        + r"\s*\((?P<inputs>.*?)\)\s*(?:->\s*(?P<outputs>\([^)]*\)|[^{\s]+))?\s*\{",
        re.S,
    )
    for match in pattern.finditer(text):
        if mlir_symbol_name(match.group("symbol")) == symbol:
            return match.group("inputs") or "", match.group("outputs") or ""
    raise InterfaceLoweringError(f"Fabric module symbol not found: {symbol}")


def find_system_body(text: str, symbol: str) -> None:
    pattern = re.compile(
        r"fabric\.system\s+@(?P<symbol>"
        + SYMBOL_PATTERN
        + r")\s+memory_model\s*=",
        re.S,
    )
    for match in pattern.finditer(text):
        if mlir_symbol_name(match.group("symbol")) == symbol:
            return
    raise InterfaceLoweringError(f"Fabric system symbol not found: {symbol}")


def classify_boundary_port(
    raw_name: str,
    direction: str,
    fabric_type: str,
    used: set[str],
) -> tuple[ScalarPort | None, dict[str, str] | None]:
    width = scalar_bits_width(fabric_type)
    if width is None:
        return None, unsupported_boundary_diagnostic(raw_name, direction, fabric_type)
    return (
        ScalarPort(
            name=sanitize_port_name(raw_name, used),
            direction=direction,
            fabric_type=fabric_type,
            width=width,
        ),
        None,
    )


def parse_input_ports(raw_inputs: str, used: set[str]) -> tuple[list[ScalarPort], list[dict[str, str]]]:
    ports: list[ScalarPort] = []
    diagnostics: list[dict[str, str]] = []
    for entry in split_top_level_commas(raw_inputs):
        match = re.fullmatch(r"%([A-Za-z_][A-Za-z0-9_$]*)\s*:\s*(.+)", entry.strip(), re.S)
        if not match:
            raise InterfaceLoweringError(f"unsupported Fabric module input syntax: {entry}")
        raw_name, fabric_type = match.groups()
        fabric_type = " ".join(fabric_type.split())
        port, diagnostic = classify_boundary_port(raw_name, "input", fabric_type, used)
        if port is not None:
            ports.append(port)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return ports, diagnostics


def parse_output_ports(raw_outputs: str, used: set[str]) -> tuple[list[ScalarPort], list[dict[str, str]]]:
    if not raw_outputs.strip():
        return [], []
    output_text = raw_outputs.strip()
    if output_text.startswith("(") and output_text.endswith(")"):
        output_text = output_text[1:-1]
    ports: list[ScalarPort] = []
    diagnostics: list[dict[str, str]] = []
    for index, entry in enumerate(split_top_level_commas(output_text)):
        fabric_type = " ".join(entry.split())
        port, diagnostic = classify_boundary_port(f"out_{index}", "output", fabric_type, used)
        if port is not None:
            ports.append(port)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return ports, diagnostics


def module_ports(
    hardware_identity: str,
    source_base: Path = ROOT,
) -> tuple[list[ScalarPort], list[dict[str, str]]]:
    path, symbol = source_root(hardware_identity, source_base)
    raw_inputs, raw_outputs = find_module_signature(path.read_text(), symbol)
    used = {"clk", "rst_n"}
    input_ports, input_diagnostics = parse_input_ports(raw_inputs, used)
    output_ports, output_diagnostics = parse_output_ports(raw_outputs, used)
    return input_ports + output_ports, input_diagnostics + output_diagnostics


def system_ports(
    hardware_identity: str,
    source_base: Path = ROOT,
) -> tuple[list[ScalarPort], list[dict[str, str]]]:
    path, symbol = source_root(hardware_identity, source_base)
    find_system_body(path.read_text(), symbol)
    return [], []


def read_mapping_evidence(path: Path, hardware_identity: str) -> MappingEvidence:
    if not path.is_file():
        raise MappingArtifactError(
            "missing_mapping_artifact",
            f"mapping artifact does not exist: {path}",
        )
    try:
        raw_data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise MappingArtifactError(
            "invalid_mapping_artifact",
            f"mapping artifact is not JSON: {error}",
        ) from error
    if not isinstance(raw_data, dict):
        raise MappingArtifactError("invalid_mapping_artifact", "mapping artifact must be a JSON object")
    if raw_data.get("kind") != "pnr_mapping":
        raise MappingArtifactError(
            "invalid_mapping_artifact",
            "mapping artifact kind must be pnr_mapping",
        )
    if raw_data.get("status") != "pass":
        raise MappingArtifactError(
            "mapping_artifact_failure",
            "mapping artifact must be passing for mapped workload RTL",
        )
    workload = string_field(raw_data, "workload")
    hardware = string_field(raw_data, "hardware")
    mapping_id = string_field(raw_data, "mapping_id")
    missing = [
        key
        for key, value in (
            ("workload", workload),
            ("hardware", hardware),
            ("mapping_id", mapping_id),
        )
        if not value
    ]
    if missing:
        raise MappingArtifactError(
            "invalid_mapping_artifact",
            f"mapping artifact lacks {', '.join(missing)}",
        )
    if not hardware_matches_mapping_record(hardware_identity, raw_data):
        raise MappingArtifactError(
            "mapping_hardware_mismatch",
            f"mapping hardware {hardware} does not match RTL hardware {hardware_identity}",
        )
    return MappingEvidence(
        identity=intermediate_artifacts.artifact_id_for_path(path),
        workload=workload,
        hardware=hardware,
        mapping_id=mapping_id,
    )


def read_mapping_artifact_object(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    try:
        raw_data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw_data, dict):
        return {}
    return raw_data


def mapping_selection_blocked_manifest(
    mapping_artifact: Path,
    diagnostic_class: str,
    message: str,
) -> dict[str, object]:
    return blocked_manifest(
        "",
        diagnostic_class,
        message,
        mode="mapped_workload_rtl",
        mapping_artifact_identity=intermediate_artifacts.artifact_id_for_path(mapping_artifact),
    )


def interface_manifest(top_module: str, ports: list[ScalarPort]) -> list[dict[str, object]]:
    if not ports:
        return []
    return [
        {
            "interface_id": f"interface::{top_module}::scalar_bits_top_ports",
            "interface_kind": "scalar_bits_top_ports",
            "ports": [port.manifest_entry() for port in ports],
        }
    ]


def sv_source(module: str, node_count: str, link_count: str, ports: list[ScalarPort]) -> str:
    port_lines = ["  input logic clk", "  input logic rst_n"]
    port_lines.extend(
        f"  {port.direction} {port.systemverilog_type} {port.name}" for port in ports
    )
    rendered_ports = ",\n".join(port_lines)
    return (
        "`timescale 1ns/1ps\n"
        f"module {module}(\n"
        f"{rendered_ports}\n"
        ");\n"
        f"  localparam int LOOM_NODE_COUNT = {node_count};\n"
        f"  localparam int LOOM_LINK_COUNT = {link_count};\n"
        "endmodule\n"
    )


def blocked_manifest(
    hardware_identity: str,
    diagnostic_class: str,
    message: str,
    *,
    mode: str = "architecture_rtl",
    mapping_artifact_identity: str = "",
) -> dict[str, object]:
    top_module = module_name(hardware_identity) if hardware_identity else "blocked"
    return {
        "schema_version": 1,
        "kind": "rtl_manifest",
        "manifest_id": f"rtl-manifest::{top_module}",
        "mode": mode,
        "source_hardware_root": hardware_identity,
        "source_fabric_adg_identity": hardware_identity,
        "mapping_artifact_identity": mapping_artifact_identity,
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
                "diagnostic_class": diagnostic_class,
                "message": message,
            }
        ],
        "status": "blocked",
    }


def build_manifest(
    output: Path,
    hardware_row: dict[str, str],
    mapping_artifact: Path | None = None,
    source_base: Path = ROOT,
) -> dict[str, object]:
    hardware_identity = hardware_row["hardware"]
    topology_class = hardware_row.get("topology_class", "")
    top_module = module_name(hardware_identity)
    mapping_evidence = None
    if mapping_artifact is not None:
        try:
            mapping_evidence = read_mapping_evidence(mapping_artifact, hardware_identity)
        except MappingArtifactError as error:
            return blocked_manifest(
                hardware_identity,
                error.diagnostic_class,
                str(error),
                mode="mapped_workload_rtl",
                mapping_artifact_identity=intermediate_artifacts.artifact_id_for_path(mapping_artifact),
            )
    try:
        if topology_class == "fabric_system":
            ports, diagnostics = system_ports(hardware_identity, source_base)
        else:
            ports, diagnostics = module_ports(hardware_identity, source_base)
    except InterfaceLoweringError as error:
        return blocked_manifest(
            hardware_identity,
            "unsupported_rtl_interface",
            str(error),
            mode="mapped_workload_rtl" if mapping_evidence else "architecture_rtl",
            mapping_artifact_identity=mapping_evidence.identity if mapping_evidence else "",
        )
    source_relative = Path("rtl") / f"{top_module}.sv"
    source_path = output.parent / source_relative
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        sv_source(
            top_module,
            hardware_row.get("node_count", "0") or "0",
            hardware_row.get("link_count", "0") or "0",
            ports,
        )
    )
    mode = "mapped_workload_rtl" if mapping_evidence else "architecture_rtl"
    source_root_kind = "fabric_system" if topology_class == "fabric_system" else "fabric_adg"
    behavioral_model = (
        "behavioral_fabric_system_shell"
        if topology_class == "fabric_system"
        else "behavioral_fabric_module_shell"
    )
    lowering_configuration: dict[str, object] = {
        "lowering_kind": mode,
        "source_root_kind": source_root_kind,
        "systemverilog_profile": "behavioral_shell_v1",
        "node_count": int(hardware_row.get("node_count", "0") or "0"),
        "link_count": int(hardware_row.get("link_count", "0") or "0"),
    }
    constraints: list[dict[str, object]] = []
    activity_hooks = [
        {
            "source": "rtl_signal_names",
            "top_level_module": top_module,
        }
    ]
    mapping_artifact_identity = ""
    if mapping_evidence is not None:
        mapping_artifact_identity = mapping_evidence.identity
        lowering_configuration.update(
            {
                "mapping_artifact_identity": mapping_evidence.identity,
                "mapping_id": mapping_evidence.mapping_id,
                "mapping_hardware": mapping_evidence.hardware,
                "workload": mapping_evidence.workload,
            }
        )
        constraints.append(
            {
                "constraint_kind": "pnr_mapping_binding",
                "mapping_artifact_identity": mapping_evidence.identity,
                "mapping_id": mapping_evidence.mapping_id,
                "workload": mapping_evidence.workload,
            }
        )
        activity_hooks.append(
            {
                "source": "pnr_mapping_activity",
                "mapping_artifact_identity": mapping_evidence.identity,
                "mapping_id": mapping_evidence.mapping_id,
            }
        )
    return {
        "schema_version": 1,
        "kind": "rtl_manifest",
        "manifest_id": f"rtl-manifest::{top_module}",
        "mode": mode,
        "source_hardware_root": hardware_identity,
        "source_fabric_adg_identity": hardware_identity,
        "mapping_artifact_identity": mapping_artifact_identity,
        "lowering_configuration": lowering_configuration,
        "emitted_source_files": [
            {
                "path": source_relative.as_posix(),
                "language": "systemverilog",
                "fingerprint": intermediate_artifacts.artifact_fingerprint(source_path),
            }
        ],
        "top_level_modules": [top_module],
        "generated_packages": [],
        "generated_interfaces": interface_manifest(top_module, ports),
        "black_box_modules": [],
        "behavioral_models": [behavioral_model],
        "required_tool_capability_classes": ["rtl_lint"],
        "required_library_profile_classes": [],
        "constraints": constraints,
        "activity_hooks": activity_hooks,
        "diagnostics": diagnostics,
        "status": "pass",
    }


def scaffold(output: Path) -> dict[str, object]:
    return blocked_manifest(
        "",
        "missing_fabric_adg",
        "no passing ADG hardware summary row was provided",
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hardware_summary = Path(args.hardware_summary) if args.hardware_summary else Path()
    mapping_artifact = Path(args.mapping_artifact) if args.mapping_artifact else None
    rows = pass_hardware_rows(hardware_summary)
    sorted_rows = sorted(rows, key=lambda row: row["hardware"])
    source_base = hardware_summary.parent.resolve() if args.hardware_summary else ROOT
    mapping_data = read_mapping_artifact_object(mapping_artifact)
    hardware_hint = string_field(mapping_data, "hardware")
    if hardware_hint:
        matching_rows = [
            row
            for row in sorted_rows
            if hardware_matches_mapping_record(row["hardware"], mapping_data)
        ]
        if len(matching_rows) == 1:
            sorted_rows = matching_rows
        elif len(matching_rows) > 1 and mapping_artifact is not None:
            manifest = mapping_selection_blocked_manifest(
                mapping_artifact,
                "ambiguous_mapping_hardware",
                f"mapping hardware {hardware_hint} matches multiple passing hardware rows",
            )
            output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            return 1
        elif mapping_artifact is not None:
            manifest = mapping_selection_blocked_manifest(
                mapping_artifact,
                "mapping_hardware_mismatch",
                f"mapping hardware {hardware_hint} does not match any passing hardware row",
            )
            output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            return 1
    if not sorted_rows and mapping_artifact is not None:
        manifest = mapping_selection_blocked_manifest(
            mapping_artifact,
            "missing_fabric_adg",
            "no passing ADG hardware summary row was provided",
        )
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return 1
    manifest = (
        build_manifest(output, sorted_rows[0], mapping_artifact, source_base)
        if sorted_rows
        else scaffold(output)
    )
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
