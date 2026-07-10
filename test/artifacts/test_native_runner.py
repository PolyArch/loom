#!/usr/bin/env python3
"""Regression tests for the manifest-driven app native runner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import artifact_test_common


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_native_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("loom_test_native_runner", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load native runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_case(
    app_root: Path,
    name: str,
    language: str,
    source: str,
    expected: bytes,
    compiler_flags: list[str] | None = None,
) -> dict[str, object]:
    case_dir = app_root / name
    case_dir.mkdir(parents=True)
    source_name = "main.c" if language == "c" else "main.cpp"
    (case_dir / source_name).write_text(source)
    (case_dir / "expected.txt").write_bytes(expected)
    return {
        "case": name,
        "language": language,
        "sources": [source_name],
        "expected_stdout": "expected.txt",
        "tiers": ["run"],
        "compiler_flags": compiler_flags or [],
        "link_flags": [],
        "expected_executables": [name],
        "feature_tags": ["native-runner-fixture"],
    }


def write_fixture(root: Path) -> Path:
    app_root = root / "app"
    cases = [
        add_case(
            app_root,
            "c_ok",
            "c",
            (
                "#include <stdio.h>\n"
                "#ifndef CASE_VALUE\n"
                "#error CASE_VALUE is required\n"
                "#endif\n"
                'int main(void) { printf("%d\\n", CASE_VALUE); return 0; }\n'
            ),
            b"7\n",
            ["-DCASE_VALUE=7"],
        ),
        add_case(
            app_root,
            "cxx_ok",
            "cxx",
            '#include <iostream>\nint main() { std::cout << "cxx-ok\\n"; }\n',
            b"cxx-ok\n",
        ),
        add_case(
            app_root,
            "mismatch",
            "c",
            '#include <stdio.h>\nint main(void) { puts("actual"); return 0; }\n',
            b"expected\n",
        ),
    ]
    manifest = app_root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": cases}))
    return manifest


def run_cli(
    runner: Path,
    manifest: Path,
    build_root: Path,
    cases: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(runner),
        "--manifest",
        str(manifest),
        "--build-root",
        str(build_root),
        "--cc",
        "gcc",
        "--cxx",
        "g++",
    ]
    for case in cases:
        command.extend(["--case", case])
    return subprocess.run(
        command,
        cwd=manifest.parent.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def check_cli_success(runner: Path, manifest: Path, fixture_root: Path) -> None:
    result = run_cli(
        runner,
        manifest,
        fixture_root / "cli-build",
        ("c_ok", "cxx_ok"),
    )
    require(
        result.returncode == 0,
        f"native runner CLI failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    for case in ("c_ok", "cxx_ok"):
        require(
            f"[{case}] PASS" in result.stdout, f"missing {case} result: {result.stdout}"
        )


def check_cli_failure(runner: Path, manifest: Path, fixture_root: Path) -> None:
    result = run_cli(
        runner,
        manifest,
        fixture_root / "cli-mismatch-build",
        ("mismatch",),
    )
    require(
        result.returncode != 0, f"mismatch CLI unexpectedly passed: {result.stdout}"
    )
    require(
        "[mismatch]" in result.stderr and "stdout mismatch" in result.stderr,
        f"missing mismatch CLI diagnostic: {result.stderr}",
    )


def check_api_failure(runner: ModuleType, manifest: Path, fixture_root: Path) -> None:
    results = runner.run_cases(
        manifest_path=manifest,
        case_names=["mismatch"],
        build_root=fixture_root / "api-build",
        caller_cwd=fixture_root,
    )
    require(
        len(results) == 1 and not results[0].passed, f"expected mismatch: {results}"
    )
    require(
        any("stdout mismatch" in diagnostic for diagnostic in results[0].diagnostics),
        f"missing stdout mismatch diagnostic: {results[0]}",
    )


def check_overlapping_build_root(
    runner: ModuleType, manifest: Path, fixture_root: Path
) -> None:
    source = manifest.parent / "c_ok" / "main.c"
    try:
        runner.run_cases(
            manifest_path=manifest,
            case_names=["c_ok"],
            build_root=manifest.parent,
            caller_cwd=fixture_root,
        )
    except runner.RunnerConfigurationError:
        pass
    else:
        raise AssertionError("runner accepted a build root overlapping app sources")
    require(source.is_file(), f"runner removed app source after rejection: {source}")


def check_compiler_symlink_name(runner: ModuleType, fixture_root: Path) -> None:
    bin_dir = fixture_root / "toolchain" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "loom-cc").write_text("")
    (bin_dir / "loom-c++").symlink_to("loom-cc")

    resolved = runner.resolve_compiler("toolchain/bin/loom-c++", fixture_root)
    require(
        resolved == str(bin_dir / "loom-c++"),
        f"compiler resolution changed the driver invocation name: {resolved}",
    )


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    runner_path = repo / "test" / "app" / "native_runner.py"
    runner = load_native_runner(runner_path)
    with artifact_test_common.repo_temp_dir(repo, "loom-native-runner-") as tmp:
        fixture_root = Path(tmp)
        manifest = write_fixture(fixture_root)
        check_cli_success(runner_path, manifest, fixture_root)
        check_cli_failure(runner_path, manifest, fixture_root)
        check_api_failure(runner, manifest, fixture_root)
        check_overlapping_build_root(runner, manifest, fixture_root)
        check_compiler_symlink_name(runner, fixture_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
