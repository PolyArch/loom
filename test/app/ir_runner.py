#!/usr/bin/env python3
"""Run manifest-selected LoomBench sources through Loom IR pipelines."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


APP_ROOT = Path(__file__).resolve().parent
ROOT = APP_ROOT.parents[1]
sys.path.insert(0, str(APP_ROOT))

import app_manifest  # noqa: E402


DEFAULT_BUILD_ROOT = ROOT / "build" / "test-runs" / "app-ir-runner"
CXX_EXTENSIONS = {".C", ".cc", ".cpp", ".cxx"}
MAIN_DEFINITION = re.compile(
    r"^\s*func\.func\s+(?:public\s+)?@main\b[^\n]*\{\s*$",
    re.MULTILINE,
)
SCF_OPERATION = re.compile(r"\bscf\.[A-Za-z_][A-Za-z0-9_.]*\b")


class RunnerConfigurationError(ValueError):
    """Raised when the requested run cannot be configured."""


class RunnerExecutionError(RuntimeError):
    """Raised when a pipeline command or artifact check fails."""


@dataclass(frozen=True)
class CaseSpec:
    case: str
    case_dir: Path
    language: str
    source: Path
    compiler_flags: tuple[str, ...]


@dataclass(frozen=True)
class Toolchain:
    cc: str
    cxx: str
    raise_tool: str
    lower: str
    raise_opt: str


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("raise", "dfg"), required=True)
    parser.add_argument("--case", action="append", dest="cases", metavar="NAME")
    parser.add_argument("--manifest", default=str(app_manifest.DEFAULT_MANIFEST))
    parser.add_argument("--build-root", default=str(DEFAULT_BUILD_ROOT))
    return parser.parse_args(argv)


def resolve_from_caller(path: Path | str, caller_cwd: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = caller_cwd / resolved
    return resolved.resolve()


def resolve_tool(value: str, caller_cwd: Path) -> str:
    expanded = os.path.expanduser(value)
    if os.sep in expanded and not os.path.isabs(expanded):
        return str((caller_cwd / expanded).resolve())
    return expanded


def tool_from_environment(
    environment_name: str,
    executable_name: str,
    caller_cwd: Path,
) -> str:
    fallback = ROOT / "build" / "bin" / executable_name
    value = os.environ.get(environment_name) or str(fallback)
    return resolve_tool(value, caller_cwd)


def load_toolchain(caller_cwd: Path) -> Toolchain:
    return Toolchain(
        cc=tool_from_environment("LOOM_CC", "loom-cc", caller_cwd),
        cxx=tool_from_environment("LOOM_CXX", "loom-c++", caller_cwd),
        raise_tool=tool_from_environment("LOOM_RAISE", "loom-raise", caller_cwd),
        lower=tool_from_environment("LOOM_LOWER", "loom-lower", caller_cwd),
        raise_opt=tool_from_environment(
            "LOOM_RAISE_OPT", "loom-raise-opt", caller_cwd
        ),
    )


def duplicate_values(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def source_language(source: Path) -> str:
    if source.suffix == ".c":
        return "c"
    if source.suffix in CXX_EXTENSIONS:
        return "cxx"
    raise RunnerConfigurationError(
        f"unsupported main_func source extension for {source}: {source.suffix!r}"
    )


def case_spec(entry: dict[object, object], manifest_path: Path) -> CaseSpec:
    case = str(entry["case"])
    case_dir = manifest_path.parent / case
    sources = [str(value) for value in entry["sources"]]
    main_sources = [source for source in sources if Path(source).stem == "main_func"]
    if len(main_sources) != 1:
        raise RunnerConfigurationError(
            f"{case}: expected exactly one source whose stem is main_func, "
            f"found {len(main_sources)}"
        )

    source = case_dir / main_sources[0]
    language = str(entry["language"])
    inferred_language = source_language(source)
    if language != inferred_language:
        raise RunnerConfigurationError(
            f"{case}: manifest language {language!r} does not match "
            f"source extension {source.suffix!r}"
        )

    return CaseSpec(
        case=case,
        case_dir=case_dir,
        language=language,
        source=source,
        compiler_flags=tuple(str(value) for value in entry["compiler_flags"]),
    )


def load_case_specs(
    manifest_path: Path,
    stage: str,
    case_names: Sequence[str] | None,
) -> list[CaseSpec]:
    data, diagnostics = app_manifest.validate_manifest(manifest_path)
    if diagnostics:
        raise RunnerConfigurationError("\n".join(diagnostics))

    entries = data.get("cases")
    if not isinstance(entries, list):
        raise RunnerConfigurationError("manifest cases must be a list")

    entries_by_name = {
        str(entry["case"]): entry for entry in entries if isinstance(entry, dict)
    }
    if case_names is not None:
        duplicates = duplicate_values(case_names)
        if duplicates:
            raise RunnerConfigurationError(
                f"duplicate --case values: {', '.join(duplicates)}"
            )
        unknown = [name for name in case_names if name not in entries_by_name]
        if unknown:
            raise RunnerConfigurationError(f"unknown case(s): {', '.join(unknown)}")
        selected_entries = [entries_by_name[name] for name in case_names]
    else:
        selected_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict) and stage in entry.get("tiers", [])
        ]

    if not selected_entries:
        raise RunnerConfigurationError(f"no cases selected for {stage} stage")
    return [case_spec(entry, manifest_path) for entry in selected_entries]


def command_output(result: subprocess.CompletedProcess[str]) -> str:
    sections: list[str] = []
    if result.stderr.strip():
        sections.append(f"stderr:\n{result.stderr.rstrip()}")
    if result.stdout.strip():
        sections.append(f"stdout:\n{result.stdout.rstrip()}")
    return "\n".join(sections) if sections else "no diagnostic output"


def run_command(command: Sequence[str], cwd: Path, context: str) -> None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env={**os.environ, "LC_ALL": "C"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise RunnerExecutionError(
            f"{context}: failed to launch command\n"
            f"command: {shlex.join(command)}\n"
            f"working directory: {cwd}\n"
            f"error: {exc}"
        ) from exc
    if result.returncode != 0:
        raise RunnerExecutionError(
            f"{context}: command failed with exit {result.returncode}\n"
            f"command: {shlex.join(command)}\n"
            f"working directory: {cwd}\n"
            f"{command_output(result)}"
        )


def remove_stale_output(path: Path, context: str) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise RunnerExecutionError(f"{context}: cannot remove stale output {path}: {exc}") from exc


def require_nonempty(path: Path, context: str) -> None:
    try:
        nonempty = path.is_file() and path.stat().st_size > 0
    except OSError as exc:
        raise RunnerExecutionError(f"{context}: cannot inspect output {path}: {exc}") from exc
    if not nonempty:
        raise RunnerExecutionError(f"{context}: expected nonempty output {path}")


def read_ir(path: Path, context: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RunnerExecutionError(f"{context}: cannot read generated IR {path}: {exc}") from exc


def reparse_mlir(path: Path, spec: CaseSpec, tools: Toolchain) -> None:
    run_command(
        [tools.raise_opt, str(path), "-o", os.devnull],
        spec.case_dir,
        f"{spec.case}: reparse {path.name}",
    )


def validate_raise_ir(path: Path, spec: CaseSpec) -> None:
    text = read_ir(path, f"{spec.case}: raised IR validation")
    if MAIN_DEFINITION.search(text) is None:
        raise RunnerExecutionError(
            f"{spec.case}: {path} has no public func.func @main definition with a body"
        )
    if SCF_OPERATION.search(text) is None:
        raise RunnerExecutionError(f"{spec.case}: {path} has no scf operation")


def require_dfg_artifact(path: Path, case: str) -> None:
    text = read_ir(path, f"{case}: DFG artifact check")
    if "dataflow.graph " not in text:
        raise RunnerExecutionError(
            f"{case}: {path} has no dataflow.graph definition"
        )
    if "dataflow.graph.launch " not in text:
        raise RunnerExecutionError(f"{case}: {path} has no dataflow.graph.launch")


def compiler_for(spec: CaseSpec, tools: Toolchain) -> str:
    return tools.cc if spec.language == "c" else tools.cxx


def run_case(
    spec: CaseSpec,
    stage: str,
    build_root: Path,
    tools: Toolchain,
) -> None:
    case_build = build_root / spec.case
    try:
        case_build.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RunnerExecutionError(
            f"{spec.case}: cannot create build directory {case_build}: {exc}"
        ) from exc

    llvm_ir = case_build / "main_func.ll"
    raised_ir = case_build / "main_func.scf.mlir"
    dfg_ir = case_build / "main_func.dfg.mlir"
    for output in (llvm_ir, raised_ir, dfg_ir):
        remove_stale_output(output, spec.case)

    compile_command = [
        compiler_for(spec, tools),
        "-emit-llvm",
        "-O1",
        "-S",
        *spec.compiler_flags,
        str(spec.source),
        "-o",
        str(llvm_ir),
    ]
    run_command(compile_command, spec.case_dir, f"{spec.case}: compile main_func")
    require_nonempty(llvm_ir, f"{spec.case}: LLVM IR generation")

    run_command(
        [tools.raise_tool, str(llvm_ir), "-o", str(raised_ir)],
        spec.case_dir,
        f"{spec.case}: raise LLVM IR",
    )
    require_nonempty(raised_ir, f"{spec.case}: SCF MLIR generation")
    reparse_mlir(raised_ir, spec, tools)
    validate_raise_ir(raised_ir, spec)

    if stage == "dfg":
        lower_command = [tools.lower, str(raised_ir), "-o", str(dfg_ir)]
        lower_context = f"{spec.case}: lower SCF MLIR"
        run_command(lower_command, spec.case_dir, lower_context)
        require_nonempty(dfg_ir, f"{spec.case}: DFG MLIR generation")
        reparse_mlir(dfg_ir, spec, tools)
        require_dfg_artifact(dfg_ir, spec.case)


def reject_overlapping_build_root(build_root: Path, specs: Sequence[CaseSpec]) -> None:
    for spec in specs:
        if (
            build_root == spec.case_dir
            or build_root in spec.case_dir.parents
            or spec.case_dir in build_root.parents
        ):
            raise RunnerConfigurationError(
                f"build root must not overlap app source directory: {spec.case_dir}"
            )


def main(argv: Sequence[str]) -> int:
    caller_cwd = Path.cwd()
    args = parse_args(argv)
    try:
        manifest_path = resolve_from_caller(args.manifest, caller_cwd)
        build_root = resolve_from_caller(args.build_root, caller_cwd)
        specs = load_case_specs(manifest_path, args.stage, args.cases)
        reject_overlapping_build_root(build_root, specs)
        build_root.mkdir(parents=True, exist_ok=True)
        tools = load_toolchain(caller_cwd)
        for spec in specs:
            run_case(spec, args.stage, build_root, tools)
            print(f"PASS  {spec.case}  {args.stage}")
    except RunnerConfigurationError as exc:
        print(f"[app-ir-runner] configuration error: {exc}", file=sys.stderr)
        return 2
    except (OSError, RunnerExecutionError) as exc:
        print(f"[app-ir-runner] ERROR: {exc}", file=sys.stderr)
        return 1

    print("[app-ir-runner] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
