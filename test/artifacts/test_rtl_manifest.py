#!/usr/bin/env python3
"""Regression test for deterministic RTL manifest artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "manifest_id",
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
