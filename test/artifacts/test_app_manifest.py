#!/usr/bin/env python3
"""Regression test for the app corpus manifest CLI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import artifact_test_common


def load_ir_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("loom_test_ir_runner", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load IR runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_case(root: Path, name: str, tiers: list[str]) -> dict[str, object]:
    case_dir = root / name
    case_dir.mkdir(parents=True)
    (case_dir / "main.c").write_text("int main(void) { return 0; }\n")
    (case_dir / "expected.txt").write_text("")
    return {
        "case": name,
        "language": "c",
        "sources": ["main.c"],
        "expected_stdout": "expected.txt",
        "tiers": tiers,
        "compiler_flags": [],
        "link_flags": [],
        "expected_executables": [name],
        "feature_tags": ["fixture"],
    }


def write_manifest(path: Path, cases: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "cases": cases}) + "\n")


def check_dfg_symbol_mismatch(ir_runner: ModuleType, root: Path) -> None:
    dfg_ir = root / "symbol-mismatch.mlir"
    dfg_ir.write_text(
        "dataflow.graph.func private @unrelated_kernel() {\n"
        "  dataflow.graph.return\n"
        "}\n"
    )
    case = ir_runner.CaseSpec(
        case="symbol_mismatch",
        case_dir=root,
        language="c",
        source=root / "main.c",
        compiler_flags=(),
        dfg_symbol="expected_kernel",
    )
    try:
        ir_runner.validate_dfg_ir(dfg_ir, case)
    except ir_runner.RunnerExecutionError as exc:
        if "no dataflow definition for expected_kernel" not in str(exc):
            raise AssertionError(f"unexpected DFG symbol diagnostic: {exc}") from exc
    else:
        raise AssertionError("DFG validation accepted an unrelated symbol")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    cli = [sys.executable, "test/app/app_manifest.py"]
    ir_runner = load_ir_runner(repo / "test" / "app" / "ir_runner.py")

    artifact_test_common.require_success(
        repo,
        [*cli, "validate"],
        "checked-in app manifest validation",
    )

    with artifact_test_common.repo_temp_dir(repo, "loom-app-manifest-") as tmp:
        root = Path(tmp)
        valid_manifest = root / "valid" / "manifest.json"
        write_manifest(
            valid_manifest,
            [
                create_case(valid_manifest.parent, "run_case", ["run"]),
                create_case(valid_manifest.parent, "raise_case", ["raise"]),
            ],
        )
        result = artifact_test_common.require_success(
            repo,
            [*cli, "list", "--manifest", str(valid_manifest), "--tier", "run"],
            "app manifest run-tier list",
        )
        if result.stdout != "run_case\n":
            raise AssertionError(f"unexpected run-tier list: {result.stdout!r}")

        root_list_manifest = root / "root-list.json"
        root_list_manifest.write_text(json.dumps([]) + "\n")
        result = artifact_test_common.run_command(
            repo,
            [*cli, "validate", "--manifest", str(root_list_manifest)],
        )
        if result.returncode == 0:
            raise AssertionError("manifest with a list root unexpectedly passed")
        if "manifest root must be an object" not in result.stderr:
            raise AssertionError(
                f"unexpected manifest root diagnostic: {result.stderr}"
            )

        unsafe_manifest = root / "unsafe" / "manifest.json"
        unsafe_case = create_case(unsafe_manifest.parent, "unsafe_case", ["run"])
        unsafe_case["sources"] = ["../outside.c"]
        (unsafe_manifest.parent / "outside.c").write_text(
            "int main(void) { return 0; }\n"
        )
        write_manifest(unsafe_manifest, [unsafe_case])

        result = artifact_test_common.run_command(
            repo,
            [*cli, "validate", "--manifest", str(unsafe_manifest)],
        )
        if result.returncode == 0:
            raise AssertionError("manifest source escaped the case directory")
        if "sources entries must be file names" not in result.stderr:
            raise AssertionError(f"missing unsafe source diagnostic: {result.stderr}")

        check_dfg_symbol_mismatch(ir_runner, root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
