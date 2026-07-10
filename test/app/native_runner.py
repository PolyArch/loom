#!/usr/bin/env python3
"""Compile and run app cases described by the app manifest."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


APP_ROOT = Path(__file__).resolve().parent
ROOT = APP_ROOT.parents[1]
sys.path.insert(0, str(APP_ROOT))

import app_manifest  # noqa: E402


DEFAULT_BUILD_ROOT = ROOT / "build" / "test-runs" / "native-runner"
DEFAULT_FLAGS = {
    "c": ("-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"),
    "cxx": ("-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"),
}


class RunnerConfigurationError(ValueError):
    """Raised when a requested native run cannot be configured."""


@dataclass(frozen=True)
class CaseSpec:
    case: str
    case_dir: Path
    language: str
    sources: tuple[Path, ...]
    expected_stdout: Path
    compiler_flags: tuple[str, ...]
    link_flags: tuple[str, ...]
    expected_executables: tuple[str, ...]


@dataclass(frozen=True)
class CaseResult:
    case: str
    passed: bool
    diagnostics: tuple[str, ...]

    @property
    def diagnostic(self) -> str:
        if self.passed:
            return "all executables passed"
        return "; ".join(self.diagnostics)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", action="append", dest="cases", metavar="NAME")
    selection.add_argument("--all", action="store_true", dest="run_all")
    parser.add_argument("--manifest", default=str(app_manifest.DEFAULT_MANIFEST))
    parser.add_argument("--build-root", default=str(DEFAULT_BUILD_ROOT))
    parser.add_argument("--jobs", type=positive_int)
    parser.add_argument("--cc")
    parser.add_argument("--cxx")
    return parser.parse_args(argv)


def resolve_from_caller(path: Path | str, caller_cwd: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = caller_cwd / resolved
    return resolved.resolve()


def resolve_compiler(compiler: str, caller_cwd: Path) -> str:
    expanded = os.path.expanduser(compiler)
    if os.sep in expanded and not os.path.isabs(expanded):
        return os.path.abspath(caller_cwd / expanded)
    return expanded


def env_positive_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RunnerConfigurationError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise RunnerConfigurationError(f"{name} must be a positive integer")
    return parsed


def worker_count(requested: int | None, case_count: int) -> int:
    if requested is not None and requested < 1:
        raise RunnerConfigurationError("jobs must be a positive integer")
    budget = (
        requested
        or env_positive_int("LOOM_NATIVE_RUNNER_JOBS")
        or env_positive_int("LOOM_TEST_JOBS")
        or env_positive_int("JOBS")
        or (os.cpu_count() or 1)
    )
    return max(1, min(case_count, budget))


def safe_filename(value: str, context: str) -> None:
    if value in {".", ".."} or Path(value).name != value:
        raise RunnerConfigurationError(f"{context} must be a single file name: {value!r}")


def load_case_specs(manifest_path: Path, case_names: Sequence[str] | None) -> list[CaseSpec]:
    data, diagnostics = app_manifest.validate_manifest(manifest_path)
    if diagnostics:
        raise RunnerConfigurationError("\n".join(diagnostics))

    entries = data.get("cases")
    if not isinstance(entries, list):
        raise RunnerConfigurationError("manifest cases must be a list")

    selected: set[str] | None = None
    if case_names is not None:
        if not case_names:
            raise RunnerConfigurationError("at least one case must be selected")
        duplicates = sorted({name for name in case_names if case_names.count(name) > 1})
        if duplicates:
            raise RunnerConfigurationError(f"duplicate --case values: {', '.join(duplicates)}")
        selected = set(case_names)

    manifest_names = {
        str(entry["case"])
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("case"), str)
    }
    if selected is not None:
        unknown = [name for name in case_names if name not in manifest_names]
        if unknown:
            raise RunnerConfigurationError(f"unknown case(s): {', '.join(unknown)}")

    specs: list[CaseSpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        case = str(entry["case"])
        tiers = entry["tiers"]
        if selected is not None and case not in selected:
            continue
        if "run" not in tiers:
            if selected is not None:
                raise RunnerConfigurationError(f"{case}: run tier is not enabled")
            continue
        safe_filename(case, "case")
        executables = tuple(str(value) for value in entry["expected_executables"])
        for executable in executables:
            safe_filename(executable, f"{case}: expected executable")
        case_dir = manifest_path.parent / case
        specs.append(
            CaseSpec(
                case=case,
                case_dir=case_dir,
                language=str(entry["language"]),
                sources=tuple(case_dir / str(value) for value in entry["sources"]),
                expected_stdout=case_dir / str(entry["expected_stdout"]),
                compiler_flags=tuple(str(value) for value in entry["compiler_flags"]),
                link_flags=tuple(str(value) for value in entry["link_flags"]),
                expected_executables=executables,
            )
        )

    if not specs:
        raise RunnerConfigurationError("no run-tier cases selected")
    return specs


def command_detail(output: bytes) -> str:
    lines = output.decode(errors="replace").strip().splitlines()
    return lines[0] if lines else "no diagnostic output"


def bytes_detail(value: bytes, limit: int = 240) -> str:
    if len(value) <= limit:
        return repr(value)
    return f"{value[:limit]!r}... ({len(value)} bytes total)"


def run_case(
    spec: CaseSpec,
    build_root: Path,
    cc: str,
    cxx: str,
    command_env: dict[str, str],
) -> CaseResult:
    case_build = build_root / spec.case
    diagnostics: list[str] = []
    try:
        shutil.rmtree(case_build, ignore_errors=True)
        case_build.mkdir(parents=True, exist_ok=True)
        expected = spec.expected_stdout.read_bytes()
    except OSError as exc:
        return CaseResult(spec.case, False, (f"cannot prepare build directory: {exc}",))

    compiler = cc if spec.language == "c" else cxx
    defaults = DEFAULT_FLAGS[spec.language]
    for source, executable_name in zip(spec.sources, spec.expected_executables, strict=True):
        executable = case_build / executable_name
        command = [
            compiler,
            *defaults,
            *spec.compiler_flags,
            str(source),
            "-o",
            str(executable),
            *spec.link_flags,
        ]
        try:
            compiled = subprocess.run(
                command,
                cwd=spec.case_dir,
                env=command_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            diagnostics.append(f"{executable_name}: compiler launch failed: {exc}")
            continue
        if compiled.returncode != 0:
            detail = command_detail(compiled.stderr or compiled.stdout)
            diagnostics.append(
                f"{executable_name}: compile failed with exit {compiled.returncode}: {detail}"
            )
            continue

        try:
            executed = subprocess.run(
                [str(executable)],
                cwd=spec.case_dir,
                env=command_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            diagnostics.append(f"{executable_name}: execution failed: {exc}")
            continue
        if executed.returncode != 0:
            detail = command_detail(executed.stderr)
            diagnostics.append(
                f"{executable_name}: exited {executed.returncode}: {detail}"
            )
        if executed.stdout != expected:
            diagnostics.append(
                f"{executable_name}: stdout mismatch: expected {bytes_detail(expected)}, "
                f"got {bytes_detail(executed.stdout)}"
            )

    return CaseResult(spec.case, not diagnostics, tuple(diagnostics))


def run_cases(
    *,
    manifest_path: Path | str = app_manifest.DEFAULT_MANIFEST,
    case_names: Sequence[str] | None = None,
    build_root: Path | str = DEFAULT_BUILD_ROOT,
    jobs: int | None = None,
    cc: str = "gcc",
    cxx: str = "g++",
    caller_cwd: Path | str | None = None,
) -> list[CaseResult]:
    caller = resolve_from_caller(caller_cwd or Path.cwd(), Path.cwd())
    manifest = resolve_from_caller(manifest_path, caller)
    output_root = resolve_from_caller(build_root, caller)
    specs = load_case_specs(manifest, case_names)
    for spec in specs:
        if (
            output_root == spec.case_dir
            or output_root in spec.case_dir.parents
            or spec.case_dir in output_root.parents
        ):
            raise RunnerConfigurationError(
                f"build root must not overlap app source directories: {spec.case_dir}"
            )

    resolved_cc = resolve_compiler(cc, caller)
    resolved_cxx = resolve_compiler(cxx, caller)
    output_root.mkdir(parents=True, exist_ok=True)
    command_env = os.environ.copy()
    command_env["LC_ALL"] = "C"

    results: list[CaseResult | None] = [None] * len(specs)
    with ThreadPoolExecutor(max_workers=worker_count(jobs, len(specs))) as executor:
        futures = {
            executor.submit(
                run_case,
                spec,
                output_root,
                resolved_cc,
                resolved_cxx,
                command_env,
            ): index
            for index, spec in enumerate(specs)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                results[index] = CaseResult(
                    specs[index].case,
                    False,
                    (f"internal runner error: {exc}",),
                )

    return [result for result in results if result is not None]


def report_results(results: Sequence[CaseResult]) -> int:
    failed = 0
    for result in results:
        if result.passed:
            print(f"[{result.case}] PASS")
            continue
        failed += 1
        for diagnostic in result.diagnostics:
            print(f"[{result.case}] {diagnostic}", file=sys.stderr)
    if failed:
        print(f"{failed} of {len(results)} case(s) failed", file=sys.stderr)
        return 1
    print(f"all {len(results)} case(s) passed")
    return 0


def main(argv: Sequence[str]) -> int:
    caller_cwd = Path.cwd()
    args = parse_args(argv)
    cc = args.cc or os.environ.get("CC", "gcc")
    cxx = args.cxx or os.environ.get("CXX", "g++")
    try:
        results = run_cases(
            manifest_path=args.manifest,
            case_names=None if args.run_all else args.cases,
            build_root=args.build_root,
            jobs=args.jobs,
            cc=cc,
            cxx=cxx,
            caller_cwd=caller_cwd,
        )
    except (OSError, RunnerConfigurationError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return report_results(results)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
