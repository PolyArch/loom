#!/usr/bin/env python3
"""Regression test for deterministic RTL manifest artifacts."""

from __future__ import annotations

import json
import csv
import sys
from pathlib import Path

import artifact_test_common


REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "manifest_id",
    "mode",
    "source_hardware_root",
    "source_fabric_adg_identity",
    "mapping_artifact_identity",
    "lowering_configuration",
    "emitted_source_files",
    "top_level_modules",
    "generated_packages",
    "generated_interfaces",
    "black_box_modules",
    "behavioral_models",
    "required_tool_capability_classes",
    "required_library_profile_classes",
    "constraints",
    "activity_hooks",
    "diagnostics",
    "status",
}


def write_filtered_hardware_summary(source: Path, output: Path, hardware_identity: str) -> None:
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("hardware") == hardware_identity]
        fieldnames = reader.fieldnames
    if len(rows) != 1 or fieldnames is None:
        raise AssertionError(f"expected one hardware row for {hardware_identity}: {rows}")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-rtl-manifest-") as tmp:
        out_dir = Path(tmp)
        _, hardware = artifact_test_common.prepare_candidate_inputs(repo, out_dir)
        manifest = out_dir / "rtl-manifest.json"

        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_manifest.sh",
                "--hardware-summary",
                str(hardware),
                "--output",
                str(manifest),
            ],
            "RTL manifest",
        )

        data = json.loads(manifest.read_text())
        missing = REQUIRED_KEYS - set(data)
        if missing:
            raise AssertionError(f"RTL manifest missing keys: {sorted(missing)}")
        if data["kind"] != "rtl_manifest" or data["status"] != "pass":
            raise AssertionError(f"unexpected RTL manifest status: {data}")
        if data["mode"] != "architecture_rtl":
            raise AssertionError(f"unexpected RTL manifest mode: {data}")
        if data["source_hardware_root"] != "test/fabric/unit/pe/valid.mlir::pe_2x2":
            raise AssertionError(f"unexpected RTL source hardware root: {data}")
        if data["source_fabric_adg_identity"] != "test/fabric/unit/pe/valid.mlir::pe_2x2":
            raise AssertionError(f"unexpected Fabric ADG identity: {data}")
        if data["mapping_artifact_identity"] != "":
            raise AssertionError(f"architecture RTL manifest should not claim a mapping input: {data}")
        if data["top_level_modules"] != ["pe_2x2"]:
            raise AssertionError(f"unexpected top-level module list: {data}")
        if data["black_box_modules"] != []:
            raise AssertionError(f"minimal architecture RTL should not claim black boxes: {data}")
        if data["behavioral_models"] != ["behavioral_fabric_module_shell"]:
            raise AssertionError(f"manifest should identify behavioral model use: {data}")
        if data["required_tool_capability_classes"] != ["rtl_lint"]:
            raise AssertionError(f"manifest should declare lint capability requirement: {data}")
        expected_interface = {
            "interface_id": "interface::pe_2x2::scalar_bits_top_ports",
            "interface_kind": "scalar_bits_top_ports",
            "ports": [
                {
                    "name": "a",
                    "direction": "input",
                    "fabric_type": "!fabric.bits<32>",
                    "systemverilog_type": "logic [31:0]",
                },
                {
                    "name": "b",
                    "direction": "input",
                    "fabric_type": "!fabric.bits<32>",
                    "systemverilog_type": "logic [31:0]",
                },
            ],
        }
        if data["generated_interfaces"] != [expected_interface]:
            raise AssertionError(f"manifest should record lowered scalar top ports: {data}")
        sources = data["emitted_source_files"]
        if len(sources) != 1:
            raise AssertionError(f"expected one emitted source file: {data}")
        source = sources[0]
        if source.get("path") != "rtl/pe_2x2.sv":
            raise AssertionError(f"unexpected source path: {source}")
        source_path = manifest.parent / source["path"]
        if not source_path.is_file():
            raise AssertionError(f"manifest source file does not exist: {source_path}")
        source_text = source_path.read_text()
        if "module pe_2x2" not in source_text or "endmodule" not in source_text:
            raise AssertionError(f"unexpected SystemVerilog source: {source_text}")
        for snippet in (
            "input logic clk",
            "input logic rst_n",
            "input logic [31:0] a",
            "input logic [31:0] b",
        ):
            if snippet not in source_text:
                raise AssertionError(f"SystemVerilog source lacks lowered port {snippet}: {source_text}")
        if source.get("fingerprint") != artifact_test_common.fingerprint(source_path):
            raise AssertionError(f"source fingerprint does not match file: {source}")
        hooks = data["activity_hooks"]
        if hooks != [{"source": "rtl_signal_names", "top_level_module": "pe_2x2"}]:
            raise AssertionError(f"unexpected activity hooks: {data}")

        audit = out_dir / "rtl-manifest-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit),
                str(manifest),
            ],
            "RTL manifest audit",
        )

        stale_manifest = out_dir / "stale-rtl-manifest.json"
        stale_data = json.loads(manifest.read_text())
        stale_data["emitted_source_files"][0]["fingerprint"] = "0" * 64
        stale_manifest.write_text(json.dumps(stale_data, indent=2, sort_keys=True) + "\n")
        stale_audit = out_dir / "stale-rtl-manifest-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(stale_audit),
                str(stale_manifest),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("RTL manifest with stale source fingerprint unexpectedly passed audit")

        missing_source_manifest = out_dir / "missing-source-rtl-manifest.json"
        missing_source_data = json.loads(manifest.read_text())
        missing_source_data["emitted_source_files"][0]["path"] = "rtl/missing.sv"
        missing_source_manifest.write_text(
            json.dumps(missing_source_data, indent=2, sort_keys=True) + "\n"
        )
        missing_source_audit = out_dir / "missing-source-rtl-manifest-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_source_audit),
                str(missing_source_manifest),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("RTL manifest with missing source file unexpectedly passed audit")

        missing_mode_manifest = out_dir / "missing-mode-rtl-manifest.json"
        missing_mode_data = json.loads(manifest.read_text())
        del missing_mode_data["mode"]
        missing_mode_manifest.write_text(json.dumps(missing_mode_data, indent=2, sort_keys=True) + "\n")
        missing_mode_audit = out_dir / "missing-mode-rtl-manifest-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(missing_mode_audit),
                str(missing_mode_manifest),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("RTL manifest without mode unexpectedly passed audit")

        mapped_without_mapping = out_dir / "mapped-without-mapping-rtl-manifest.json"
        mapped_without_mapping_data = json.loads(manifest.read_text())
        mapped_without_mapping_data["mode"] = "mapped_workload_rtl"
        mapped_without_mapping_data["mapping_artifact_identity"] = ""
        mapped_without_mapping.write_text(
            json.dumps(mapped_without_mapping_data, indent=2, sort_keys=True) + "\n"
        )
        mapped_without_mapping_audit = out_dir / "mapped-without-mapping-rtl-manifest-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(mapped_without_mapping_audit),
                str(mapped_without_mapping),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("mapped-workload RTL manifest without mapping unexpectedly passed audit")

        architecture_with_mapping = out_dir / "architecture-with-mapping-rtl-manifest.json"
        architecture_with_mapping_data = json.loads(manifest.read_text())
        architecture_with_mapping_data["mode"] = "architecture_rtl"
        architecture_with_mapping_data["mapping_artifact_identity"] = "pnr-mapping"
        architecture_with_mapping.write_text(
            json.dumps(architecture_with_mapping_data, indent=2, sort_keys=True) + "\n"
        )
        architecture_with_mapping_audit = out_dir / "architecture-with-mapping-rtl-manifest-audit-summary.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(architecture_with_mapping_audit),
                str(architecture_with_mapping),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("architecture RTL manifest with mapping unexpectedly passed audit")

        shared_hardware = out_dir / "shared-reduction-adg-hardware-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/fabric/run_adg_hardware_summary.sh",
                "--input",
                "test/pnr/shared_reduction_adg.mlir",
                "--output",
                str(shared_hardware),
            ],
            "shared reduction ADG hardware summary",
        )
        shared_manifest = out_dir / "shared-reduction-rtl-manifest.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_manifest.sh",
                "--hardware-summary",
                str(shared_hardware),
                "--output",
                str(shared_manifest),
            ],
            "shared reduction RTL manifest",
        )
        shared_data = json.loads(shared_manifest.read_text())
        if shared_data.get("status") != "pass":
            raise AssertionError(f"shared reduction manifest should remain pass: {shared_data}")
        shared_interfaces = shared_data.get("generated_interfaces")
        if not isinstance(shared_interfaces, list) or len(shared_interfaces) != 1:
            raise AssertionError(f"shared reduction manifest should record one scalar interface: {shared_data}")
        shared_ports = shared_interfaces[0].get("ports")
        if not isinstance(shared_ports, list):
            raise AssertionError(f"shared reduction interface lacks ports: {shared_data}")
        shared_port_names = {port.get("name") for port in shared_ports if isinstance(port, dict)}
        for name in ("i64a", "i64b", "i64c", "i32a", "i32b", "i32c", "i32d"):
            if name not in shared_port_names:
                raise AssertionError(f"shared reduction scalar port {name} was not lowered: {shared_data}")
        for name in ("mgr", "ctrl"):
            if name in shared_port_names:
                raise AssertionError(f"unsupported shared reduction port {name} should not be lowered: {shared_data}")
        shared_diagnostic_classes = {
            diagnostic.get("diagnostic_class")
            for diagnostic in shared_data.get("diagnostics", [])
            if isinstance(diagnostic, dict)
        }
        if "unsupported_rtl_boundary_type" not in shared_diagnostic_classes:
            raise AssertionError(f"shared reduction manifest should report unsupported boundary types: {shared_data}")
        shared_diagnostic_pairs = {
            (diagnostic.get("diagnostic_class"), diagnostic.get("message"))
            for diagnostic in shared_data.get("diagnostics", [])
            if isinstance(diagnostic, dict)
        }
        expected_shared_diagnostics = {
            (
                "unsupported_rtl_boundary_type",
                "input boundary port mgr has unsupported RTL boundary type memref<?x!fabric.bits<32>>",
            ),
            (
                "unsupported_rtl_boundary_type",
                "input boundary port ctrl has unsupported RTL boundary type !fabric.bits<0>",
            ),
        }
        if shared_diagnostic_pairs != expected_shared_diagnostics:
            raise AssertionError(f"unexpected shared reduction diagnostics: {shared_data}")
        shared_source = shared_manifest.parent / shared_data["emitted_source_files"][0]["path"]
        shared_source_text = shared_source.read_text()
        for snippet in (
            "input logic [63:0] i64a",
            "input logic [31:0] i32a",
        ):
            if snippet not in shared_source_text:
                raise AssertionError(f"shared reduction source lacks lowered port {snippet}: {shared_source_text}")
        for snippet in ("mgr", "ctrl"):
            if snippet in shared_source_text:
                raise AssertionError(f"shared reduction source should not fake unsupported port {snippet}")
        shared_audit = out_dir / "shared-reduction-rtl-manifest-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(shared_audit),
                str(shared_manifest),
            ],
            "shared reduction RTL manifest audit",
        )

        module_hardware = out_dir / "module-adg-hardware-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/fabric/run_adg_hardware_summary.sh",
                "--input",
                "test/fabric/unit/module/valid.mlir",
                "--output",
                str(module_hardware),
            ],
            "module ADG hardware summary",
        )
        output_hardware = out_dir / "output-module-adg-hardware-summary.csv"
        write_filtered_hardware_summary(
            module_hardware,
            output_hardware,
            "test/fabric/unit/module/valid.mlir::m_with_outputs",
        )
        output_manifest = out_dir / "output-module-rtl-manifest.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_manifest.sh",
                "--hardware-summary",
                str(output_hardware),
                "--output",
                str(output_manifest),
            ],
            "output module RTL manifest",
        )
        output_data = json.loads(output_manifest.read_text())
        output_ports = output_data["generated_interfaces"][0]["ports"]
        expected_output_ports = [
            ("a", "input", "logic [31:0]"),
            ("b", "input", "logic [31:0]"),
            ("out_0", "output", "logic [31:0]"),
            ("out_1", "output", "logic [31:0]"),
        ]
        actual_output_ports = [
            (port["name"], port["direction"], port["systemverilog_type"]) for port in output_ports
        ]
        if actual_output_ports != expected_output_ports:
            raise AssertionError(f"unexpected output module top ports: {output_data}")
        output_source = output_manifest.parent / output_data["emitted_source_files"][0]["path"]
        output_source_text = output_source.read_text()
        for snippet in (
            "input logic [31:0] a",
            "input logic [31:0] b",
            "output logic [31:0] out_0",
            "output logic [31:0] out_1",
        ):
            if snippet not in output_source_text:
                raise AssertionError(f"output module source lacks lowered port {snippet}: {output_source_text}")

        quoted_input = out_dir / "quoted-named-pe.mlir"
        quoted_input.write_text(
            """fabric.module @\"quoted module\"(%a : !fabric.bits<32>) {
  fabric.pe @\"ALU 0\" [spatial] (!fabric.bits<32>) -> (!fabric.bits<32>) {
  ^bb0(%pa: !fabric.bits<32>):
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
    fabric.yield %pa : !fabric.bits<32>
  }
  fabric.yield
}
"""
        )
        quoted_hardware = out_dir / "quoted-adg-hardware-summary.csv"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/fabric/run_adg_hardware_summary.sh",
                "--input",
                str(quoted_input),
                "--output",
                str(quoted_hardware),
            ],
            "quoted ADG hardware summary",
        )
        quoted_manifest = out_dir / "quoted-rtl-manifest.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_manifest.sh",
                "--hardware-summary",
                str(quoted_hardware),
                "--output",
                str(quoted_manifest),
            ],
            "quoted RTL manifest",
        )
        quoted_data = json.loads(quoted_manifest.read_text())
        if quoted_data["source_fabric_adg_identity"].rsplit("::", 1)[-1] != "quoted module":
            raise AssertionError(f"quoted symbol identity was not preserved: {quoted_data}")
        quoted_source = quoted_manifest.parent / quoted_data["emitted_source_files"][0]["path"]
        quoted_source_text = quoted_source.read_text()
        if "module quoted_module" not in quoted_source_text or "input logic [31:0] a" not in quoted_source_text:
            raise AssertionError(f"quoted module source lacks lowered scalar port: {quoted_source_text}")

        missing_source_hardware = out_dir / "missing-source-adg-hardware-summary.csv"
        missing_source_hardware.write_text(
            "hardware,verify_status,node_count,link_count,diagnostic\n"
            "temp/test-runs/does-not-exist.mlir::missing,pass,0,0,synthetic missing source\n"
        )
        missing_source_manifest = out_dir / "missing-source-identity-rtl-manifest.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_manifest.sh",
                "--hardware-summary",
                str(missing_source_hardware),
                "--output",
                str(missing_source_manifest),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("RTL manifest with missing Fabric source unexpectedly passed")
        missing_source_data = json.loads(missing_source_manifest.read_text())
        if (
            missing_source_data.get("status") != "blocked"
            or missing_source_data.get("diagnostics", [{}])[0].get("diagnostic_class")
            != "unsupported_rtl_interface"
        ):
            raise AssertionError(f"missing source should produce blocked manifest: {missing_source_data}")

        missing_symbol_hardware = out_dir / "missing-symbol-adg-hardware-summary.csv"
        missing_symbol_hardware.write_text(
            "hardware,verify_status,node_count,link_count,diagnostic\n"
            "test/fabric/unit/pe/valid.mlir::missing_symbol,pass,0,0,synthetic missing symbol\n"
        )
        missing_symbol_manifest = out_dir / "missing-symbol-rtl-manifest.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_manifest.sh",
                "--hardware-summary",
                str(missing_symbol_hardware),
                "--output",
                str(missing_symbol_manifest),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("RTL manifest with missing Fabric symbol unexpectedly passed")
        missing_symbol_data = json.loads(missing_symbol_manifest.read_text())
        if (
            missing_symbol_data.get("status") != "blocked"
            or missing_symbol_data.get("diagnostics", [{}])[0].get("diagnostic_class")
            != "unsupported_rtl_interface"
        ):
            raise AssertionError(f"missing symbol should produce blocked manifest: {missing_symbol_data}")

        malformed_hardware = out_dir / "malformed-adg-hardware-summary.csv"
        malformed_hardware.write_text(
            "hardware,verify_status,node_count,link_count,diagnostic\n"
            "bad_fabric,pass,not-an-int,1,synthetic malformed hardware summary\n"
        )
        malformed_manifest = out_dir / "malformed-rtl-manifest.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_manifest.sh",
                "--hardware-summary",
                str(malformed_hardware),
                "--output",
                str(malformed_manifest),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("RTL manifest with malformed hardware counts unexpectedly passed")
        malformed_data = json.loads(malformed_manifest.read_text())
        if malformed_data.get("status") != "blocked" or malformed_data.get("emitted_source_files") != []:
            raise AssertionError(f"malformed hardware counts should produce blocked manifest: {malformed_data}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
