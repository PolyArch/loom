#!/usr/bin/env python3
"""Unified corpus gate for the canonical high-level source corpus.

Runs every case selected from test/corpus_inventory.py (the sole inventory
authority) through the production compiler tools and checks the resulting
artifacts:

- stage ``llvm``: loom-cc/loom-c++ compiles each source to LLVM IR for the
  exact builtin Fabric InstructionCore target; the emitted module must carry
  the exact target triple and DataLayout.
- stage ``s0``: additionally loom-raise produces the initial Structured
  Program candidate and loom-raise-opt parses and verifies it; the S0 module
  must carry the exact target triple attribute.
- stage ``d0``: additionally loom-pre-mapping runs the production
  pre-Mapping compilation library (Structured and Canonical Dataflow
  finalizers included) against the exact builtin Fabric target resolved
  through a per-case ArtifactStore; the finalized Canonical Dataflow module
  must carry the exact target triple attribute and its structured
  graph/actor counts must parse, so a graph-free whole-program result is
  distinguishable from a nonempty Spatial graph.

A source whose feature-guarded body is empty under the exact target still
produces a valid module: a case passes on real compiler exit status,
parse/verify success, and exact target facts, never on the presence of a
callable. There are no source skips, no Unsupported-as-pass, and no output
placeholders; every requested case either passes these checks or fails with
an honest category.

Compilation uses a real configured RISC-V cross sysroot and GCC toolchain,
given explicitly (--sysroot / --gcc-toolchain or the LOOM_CORPUS_SYSROOT /
LOOM_CORPUS_GCC_TOOLCHAIN environment variables) or derived from a real
riscv64-unknown-elf-gcc. When neither is available the gate refuses to run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))

import corpus_inventory  # noqa: E402


# The exact builtin Fabric InstructionCore target profile. Every corpus
# source is compiled with exactly these flags and every emitted module is
# checked for exactly these target facts. The .ll triple and DataLayout are
# the pinned LLVM forms produced by clang for this target; the MLIR attribute
# is the triple carried through the LLVMIR import into the Structured and
# Canonical Dataflow modules.
TARGET_TRIPLE = "riscv64-unknown-elf"
TARGET_MARCH = "rv64im"
TARGET_MABI = "lp64"
LLVM_TRIPLE_LINE = 'target triple = "riscv64-unknown-unknown-elf"'
LLVM_DATALAYOUT_LINE = (
    'target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"'
)
MLIR_TRIPLE_ATTRIBUTE = 'llvm.target_triple = "riscv64-unknown-unknown-elf"'

# The one exact builtin Fabric target preset resolved by the d0 stage through
# loom-pre-mapping.
BUILTIN_TARGET_PRESET = "small"

STAGES = ("llvm", "s0", "d0")

# Honest failure categories. "compile"/"raise"/"verify"/"pre-mapping" are
# nonzero exits from the production tools; "*-artifact" categories mean the
# tool exited zero but the emitted module is missing, empty, or does not
# carry the exact target facts (a fabricated or truncated artifact), or its
# structured counts are missing or malformed; "timeout" means the per-case
# deadline killed the process group; "internal" means the gate itself hit an
# unexpected error and never counts as a case pass.
CATEGORY_COMPILE = "compile"
CATEGORY_LLVM_ARTIFACT = "llvm-artifact"
CATEGORY_RAISE = "raise"
CATEGORY_S0_ARTIFACT = "s0-artifact"
CATEGORY_VERIFY = "verify"
CATEGORY_PRE_MAPPING = "pre-mapping"
CATEGORY_D0_ARTIFACT = "d0-artifact"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_INTERNAL = "internal"

CXX_SUFFIXES = {".cc", ".cpp", ".cxx"}
RISCV_GCC_NAME = "riscv64-unknown-elf-gcc"

ENV_SYSROOT = "LOOM_CORPUS_SYSROOT"
ENV_GCC_TOOLCHAIN = "LOOM_CORPUS_GCC_TOOLCHAIN"
ENV_TOOL_NAMES = {
    "cc": "LOOM_CC",
    "cxx": "LOOM_CXX",
    "raise": "LOOM_RAISE",
    "raise_opt": "LOOM_RAISE_OPT",
    "pre_mapping": "LOOM_PRE_MAPPING",
}
TOOL_FILE_NAMES = {
    "cc": "loom-cc",
    "cxx": "loom-c++",
    "raise": "loom-raise",
    "raise_opt": "loom-raise-opt",
    "pre_mapping": "loom-pre-mapping",
}

DEFAULT_CASE_TIMEOUT_SECONDS = 120.0


class GateConfigError(ValueError):
    """Raised when the gate cannot be configured honestly."""


@dataclass(frozen=True)
class Toolchain:
    cc: str
    cxx: str
    raise_tool: str
    raise_opt: str
    pre_mapping: str
    sysroot: Path
    gcc_toolchain: Path


@dataclass(frozen=True)
class StepFailure:
    category: str
    detail: str


@dataclass(frozen=True)
class CaseResult:
    case: corpus_inventory.CorpusCase
    passed: bool
    category: str | None
    detail: str | None
    duration_seconds: float
    graphs: int | None = None
    actors: int | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "case": self.case.case,
            "category": self.category,
            "detail": self.detail,
            "duration_seconds": round(self.duration_seconds, 3),
            "identity": self.case.identity,
            "sources": len(self.case.sources),
            "status": "pass" if self.passed else "fail",
            "suite": self.case.suite,
        }
        # Present only for a passed d0 case: the structured counts that
        # distinguish a graph-free whole-program result from a nonempty
        # Spatial graph.
        if self.graphs is not None:
            payload["graphs"] = self.graphs
        if self.actors is not None:
            payload["actors"] = self.actors
        return payload


def run_quiet(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            env={**os.environ, "LC_ALL": "C"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise GateConfigError(f"cannot run {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or "unknown error"
        raise GateConfigError(
            f"{shlex.join(command)} failed: {diagnostic}"
        )
    return completed.stdout


def derive_from_riscv_gcc(gcc: str) -> tuple[Path, Path]:
    """Derive (sysroot, gcc toolchain root) from a real cross compiler."""
    sysroot_text = run_quiet([gcc, "-print-sysroot"]).strip()
    search_dirs = run_quiet([gcc, "-print-search-dirs"])
    install_line = next(
        (line for line in search_dirs.splitlines() if line.startswith("install: ")),
        None,
    )
    if install_line is None:
        raise GateConfigError(f"{gcc}: no install line in -print-search-dirs")
    install = Path(install_line.removeprefix("install: ").strip()).resolve()
    # install is <root>/lib/gcc/<machine>/<version>.
    if install.parents[1].name != "gcc" or install.parents[2].name != "lib":
        raise GateConfigError(
            f"{gcc}: unexpected GCC install layout: {install}"
        )
    toolchain_root = install.parents[3]
    if sysroot_text:
        sysroot = Path(sysroot_text).resolve()
    else:
        # Bare-metal builds may report no sysroot; the target system root
        # then lives under the toolchain root.
        sysroot = toolchain_root / TARGET_TRIPLE
    return sysroot, toolchain_root


def resolve_toolchain(args: argparse.Namespace) -> Toolchain:
    sysroot: Path | None = None
    gcc_toolchain: Path | None = None
    if args.sysroot:
        sysroot = Path(args.sysroot).expanduser().resolve()
    elif os.environ.get(ENV_SYSROOT):
        sysroot = Path(os.environ[ENV_SYSROOT]).expanduser().resolve()
    if args.gcc_toolchain:
        gcc_toolchain = Path(args.gcc_toolchain).expanduser().resolve()
    elif os.environ.get(ENV_GCC_TOOLCHAIN):
        gcc_toolchain = Path(os.environ[ENV_GCC_TOOLCHAIN]).expanduser().resolve()

    if sysroot is None or gcc_toolchain is None:
        gcc = args.riscv_gcc or shutil.which(RISCV_GCC_NAME)
        if gcc is not None:
            derived_sysroot, derived_toolchain = derive_from_riscv_gcc(gcc)
            if sysroot is None:
                sysroot = derived_sysroot
            if gcc_toolchain is None:
                gcc_toolchain = derived_toolchain

    missing = []
    if sysroot is None:
        missing.append(
            f"RISC-V cross sysroot: pass --sysroot, set {ENV_SYSROOT}, or put "
            f"{RISCV_GCC_NAME} on PATH"
        )
    if gcc_toolchain is None:
        missing.append(
            f"RISC-V GCC toolchain: pass --gcc-toolchain, set "
            f"{ENV_GCC_TOOLCHAIN}, or put {RISCV_GCC_NAME} on PATH"
        )
    if missing:
        raise GateConfigError("; ".join(missing))
    assert sysroot is not None and gcc_toolchain is not None

    tools: dict[str, str] = {}
    for key, file_name in TOOL_FILE_NAMES.items():
        candidate = os.environ.get(ENV_TOOL_NAMES[key]) or str(
            ROOT / "build" / "bin" / file_name
        )
        if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
            raise GateConfigError(
                f"{file_name} not found or not executable at: {candidate} "
                f"(override with {ENV_TOOL_NAMES[key]})"
            )
        tools[key] = candidate

    if not (sysroot / "include" / "stdint.h").is_file():
        raise GateConfigError(
            f"not a configured RISC-V cross sysroot (missing include/stdint.h): "
            f"{sysroot}"
        )
    if not (gcc_toolchain / "lib" / "gcc" / TARGET_TRIPLE).is_dir():
        raise GateConfigError(
            f"not a {TARGET_TRIPLE} GCC toolchain (missing "
            f"lib/gcc/{TARGET_TRIPLE}): {gcc_toolchain}"
        )
    return Toolchain(
        cc=tools["cc"],
        cxx=tools["cxx"],
        raise_tool=tools["raise"],
        raise_opt=tools["raise_opt"],
        pre_mapping=tools["pre_mapping"],
        sysroot=sysroot,
        gcc_toolchain=gcc_toolchain,
    )


def target_flags(toolchain: Toolchain) -> list[str]:
    return [
        f"--target={TARGET_TRIPLE}",
        f"-march={TARGET_MARCH}",
        f"-mabi={TARGET_MABI}",
        f"--sysroot={toolchain.sysroot}",
        f"--gcc-toolchain={toolchain.gcc_toolchain}",
    ]


def suite_compile_flags(suite: str, external_root: Path) -> list[str]:
    """Uniform per-suite source configuration; never per-source overrides."""
    if suite == "loombench":
        return []
    if suite == "cmsis-dsp":
        root = external_root / "cmsis-dsp"
        # __GNUC_PYTHON__ is the official CMSIS-DSP non-Arm scalar source
        # configuration; it disables the CMSIS-Core dependency.
        return [
            "-D__GNUC_PYTHON__",
            f"-I{root / 'Include'}",
            f"-I{root / 'PrivateInclude'}",
        ]
    if suite == "cmsis-nn":
        return [f"-I{external_root / 'cmsis-nn' / 'Include'}"]
    raise GateConfigError(f"unknown suite: {suite}")


def resolve_source(repo_relative: str, external_root: Path) -> Path:
    path = Path(repo_relative)
    if not path.parts:
        raise GateConfigError(f"empty corpus source path: {repo_relative!r}")
    if path.parts[0] == "externals":
        return external_root.joinpath(*path.parts[1:])
    return ROOT / path


def compiler_for(toolchain: Toolchain, source: Path) -> str:
    if source.suffix == ".c":
        return toolchain.cc
    if source.suffix in CXX_SUFFIXES:
        return toolchain.cxx
    raise GateConfigError(f"unsupported corpus source extension: {source}")


def compile_command(
    toolchain: Toolchain,
    suite_flags: Sequence[str],
    source: Path,
    output: Path,
) -> list[str]:
    return [
        compiler_for(toolchain, source),
        *target_flags(toolchain),
        *suite_flags,
        "-emit-llvm",
        "-S",
        "-O1",
        str(source),
        "-o",
        str(output),
    ]


def raise_command(toolchain: Toolchain, llvm_ir: Path, output: Path) -> list[str]:
    return [toolchain.raise_tool, str(llvm_ir), "-o", str(output)]


def verify_command(toolchain: Toolchain, s0_module: Path) -> list[str]:
    return [toolchain.raise_opt, str(s0_module), "-o", os.devnull]


def pre_mapping_command(
    toolchain: Toolchain,
    llvm_ir: Path,
    store_dir: Path,
    d0_module: Path,
    counts: Path,
) -> list[str]:
    return [
        toolchain.pre_mapping,
        f"--builtin={BUILTIN_TARGET_PRESET}",
        f"--artifact-store={store_dir}",
        f"--counts={counts}",
        str(llvm_ir),
        "-o",
        str(d0_module),
    ]


def llvm_ir_defect(path: Path) -> str | None:
    """Return why a .ll artifact is not a real exact-target module."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"cannot read LLVM IR {path}: {exc}"
    if not text.strip():
        return f"empty LLVM IR: {path}"
    lines = text.splitlines()
    if LLVM_TRIPLE_LINE not in lines:
        return f"LLVM IR lacks exact target triple {LLVM_TRIPLE_LINE!r}: {path}"
    if LLVM_DATALAYOUT_LINE not in lines:
        return f"LLVM IR lacks exact target DataLayout {LLVM_DATALAYOUT_LINE!r}: {path}"
    return None


def s0_module_defect(path: Path) -> str | None:
    """Return why an S0 artifact is not a real exact-target module."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"cannot read S0 module {path}: {exc}"
    if not text.strip():
        return f"empty S0 module: {path}"
    if MLIR_TRIPLE_ATTRIBUTE not in text:
        return f"S0 module lacks exact target triple {MLIR_TRIPLE_ATTRIBUTE!r}: {path}"
    return None


def d0_module_defect(path: Path) -> str | None:
    """Return why a D0 artifact is not a real exact-target module."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"cannot read D0 module {path}: {exc}"
    if not text.strip():
        return f"empty D0 module: {path}"
    if MLIR_TRIPLE_ATTRIBUTE not in text:
        return f"D0 module lacks exact target triple {MLIR_TRIPLE_ATTRIBUTE!r}: {path}"
    return None


def parse_d0_counts(path: Path) -> tuple[dict[str, int] | None, str | None]:
    """Parse the structured graph/actor counts emitted by loom-pre-mapping.

    Returns (counts, None) for a well-formed object, or (None, defect) when
    the file is missing, unreadable, or not exactly the structured counts
    the production tool emits.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        return None, f"cannot read pre-Mapping counts {path}: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"malformed pre-Mapping counts {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"pre-Mapping counts are not a JSON object: {path}"
    expected_fields = {"graphs", "actors"}
    if set(payload) != expected_fields:
        return None, (
            "pre-Mapping counts contain missing or unexpected fields "
            f"(expected {sorted(expected_fields)}): {path}"
        )
    counts: dict[str, int] = {}
    for key in ("graphs", "actors"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None, (
                f"pre-Mapping counts lack a non-negative integer {key!r}: {path}"
            )
        counts[key] = value
    return counts, None


def run_step(
    command: Sequence[str],
    log_path: Path,
    deadline: float,
    category: str,
) -> StepFailure | None:
    """Run one pipeline step in its own process group under a deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return StepFailure(
            CATEGORY_TIMEOUT,
            f"case deadline exceeded before running: {shlex.join(command)}",
        )
    try:
        with open(log_path, "wb") as log:
            try:
                process = subprocess.Popen(
                    list(command),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "LC_ALL": "C"},
                )
            except OSError as exc:
                return StepFailure(
                    category, f"cannot launch {shlex.join(command)}: {exc}"
                )
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                return StepFailure(
                    CATEGORY_TIMEOUT,
                    f"deadline exceeded; killed process group {process.pid}: "
                    f"{shlex.join(command)}",
                )
    except OSError as exc:
        return StepFailure(category, f"cannot write log {log_path}: {exc}")
    if process.returncode != 0:
        return StepFailure(
            category,
            f"exit {process.returncode}: {shlex.join(command)}; see {log_path}",
        )
    return None


def case_out_dir(out_root: Path, case: corpus_inventory.CorpusCase) -> Path:
    return out_root / case.suite / case.case.removesuffix(".c")


def run_case(
    case: corpus_inventory.CorpusCase,
    stage: str,
    toolchain: Toolchain,
    external_root: Path,
    out_root: Path,
    case_timeout: float,
) -> CaseResult:
    started = time.monotonic()
    deadline = started + case_timeout
    graphs = 0
    actors = 0

    def finish(category: str | None, detail: str | None) -> CaseResult:
        passed = category is None
        return CaseResult(
            case=case,
            passed=passed,
            category=category,
            detail=detail,
            duration_seconds=time.monotonic() - started,
            graphs=graphs if passed and stage == "d0" else None,
            actors=actors if passed and stage == "d0" else None,
        )

    try:
        case_dir = case_out_dir(out_root, case)
        case_dir.mkdir(parents=True, exist_ok=True)
        store_dir = case_dir / "artifact-store"
        if stage == "d0":
            store_dir.mkdir(parents=True, exist_ok=True)
        suite_flags = suite_compile_flags(case.suite, external_root)
        for repo_relative in case.sources:
            source = resolve_source(repo_relative, external_root)
            stem = source.stem
            llvm_ir = case_dir / f"{stem}.ll"
            failure = run_step(
                compile_command(toolchain, suite_flags, source, llvm_ir),
                case_dir / f"{stem}.compile.log",
                deadline,
                CATEGORY_COMPILE,
            )
            if failure is not None:
                return finish(failure.category, f"{repo_relative}: {failure.detail}")
            defect = llvm_ir_defect(llvm_ir)
            if defect is not None:
                return finish(CATEGORY_LLVM_ARTIFACT, f"{repo_relative}: {defect}")
            if stage == "llvm":
                continue
            if stage == "d0":
                d0_module = case_dir / f"{stem}.dfg.mlir"
                counts_path = case_dir / f"{stem}.counts.json"
                failure = run_step(
                    pre_mapping_command(
                        toolchain, llvm_ir, store_dir, d0_module, counts_path
                    ),
                    case_dir / f"{stem}.pre-mapping.log",
                    deadline,
                    CATEGORY_PRE_MAPPING,
                )
                if failure is not None:
                    return finish(
                        failure.category, f"{repo_relative}: {failure.detail}"
                    )
                defect = d0_module_defect(d0_module)
                if defect is not None:
                    return finish(CATEGORY_D0_ARTIFACT, f"{repo_relative}: {defect}")
                counts, defect = parse_d0_counts(counts_path)
                if defect is not None:
                    return finish(CATEGORY_D0_ARTIFACT, f"{repo_relative}: {defect}")
                assert counts is not None
                graphs += counts["graphs"]
                actors += counts["actors"]
                continue
            s0_module = case_dir / f"{stem}.scf.mlir"
            failure = run_step(
                raise_command(toolchain, llvm_ir, s0_module),
                case_dir / f"{stem}.raise.log",
                deadline,
                CATEGORY_RAISE,
            )
            if failure is not None:
                return finish(failure.category, f"{repo_relative}: {failure.detail}")
            defect = s0_module_defect(s0_module)
            if defect is not None:
                return finish(CATEGORY_S0_ARTIFACT, f"{repo_relative}: {defect}")
            failure = run_step(
                verify_command(toolchain, s0_module),
                case_dir / f"{stem}.verify.log",
                deadline,
                CATEGORY_VERIFY,
            )
            if failure is not None:
                return finish(failure.category, f"{repo_relative}: {failure.detail}")
        return finish(None, None)
    except (GateConfigError, OSError) as exc:
        return finish(CATEGORY_INTERNAL, f"gate error: {exc}")
    except Exception:  # noqa: BLE001 - a runner bug must not pass or hang the gate
        return finish(CATEGORY_INTERNAL, traceback.format_exc().rstrip())


def run_cases(
    cases: Sequence[corpus_inventory.CorpusCase],
    stage: str,
    toolchain: Toolchain,
    external_root: Path,
    out_root: Path,
    jobs: int,
    case_timeout: float,
) -> list[CaseResult]:
    results: list[CaseResult | None] = [None] * len(cases)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                run_case, case, stage, toolchain, external_root, out_root, case_timeout
            ): index
            for index, case in enumerate(cases)
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    for result in results:
        if result is None:  # unreachable: run_case never raises
            raise AssertionError("missing case result")
    return [result for result in results if result is not None]


def default_jobs() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        choices=corpus_inventory.SUITE_ORDER,
        dest="suites",
        help="restrict to a suite (repeatable; default: all suites)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="cases",
        metavar="SUITE:CASE",
        help="restrict to an inventory case identity (repeatable)",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="d0",
        help="gate stage to check (default: d0)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="bounded parallel case workers (default: %(default)s)",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=DEFAULT_CASE_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="per-case wall-clock deadline; the case process group is killed "
        "when it expires (default: %(default)s)",
    )
    parser.add_argument(
        "--sysroot",
        help=f"RISC-V cross sysroot (or set {ENV_SYSROOT}; else derived from "
        f"{RISCV_GCC_NAME})",
    )
    parser.add_argument(
        "--gcc-toolchain",
        help=f"RISC-V GCC toolchain root (or set {ENV_GCC_TOOLCHAIN}; else "
        f"derived from {RISCV_GCC_NAME})",
    )
    parser.add_argument(
        "--riscv-gcc",
        help=f"path to {RISCV_GCC_NAME} used to derive sysroot/toolchain "
        "(default: search PATH)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="artifact root (default: build/test-runs/corpus-gate/<stage>)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="JSON summary path (default: <out-dir>/summary.json)",
    )
    return parser.parse_args(argv)


def render_human(
    results: Sequence[CaseResult],
    stage: str,
    toolchain: Toolchain,
    jobs: int,
    duration_seconds: float,
) -> str:
    lines = [
        f"[corpus-gate] stage={stage} target={TARGET_TRIPLE} "
        f"march={TARGET_MARCH} mabi={TARGET_MABI} "
        f"builtin={BUILTIN_TARGET_PRESET} "
        f"sysroot={toolchain.sysroot} gcc-toolchain={toolchain.gcc_toolchain}"
    ]
    for result in results:
        case = result.case
        label = "PASS" if result.passed else "FAIL"
        line = (
            f"{label}  {case.identity}  "
            f"({len(case.sources)} source(s), {result.duration_seconds:.2f}s)"
        )
        if result.passed and result.graphs is not None:
            line += f"  graphs={result.graphs} actors={result.actors}"
        if not result.passed:
            line += f"  [{result.category}] {result.detail}"
        lines.append(line)
    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    lines.append(
        f"[corpus-gate] {passed} passed, {failed} failed, {len(results)} total "
        f"in {duration_seconds:.1f}s (stage={stage}, jobs={jobs})"
    )
    categories: dict[str, int] = {}
    for result in results:
        if not result.passed and result.category is not None:
            categories[result.category] = categories.get(result.category, 0) + 1
    if categories:
        summary = ", ".join(
            f"{category}={categories[category]}" for category in sorted(categories)
        )
        lines.append(f"[corpus-gate] failures by category: {summary}")
    lines.append(f"[corpus-gate] {'PASS' if failed == 0 else 'FAIL'}")
    return "\n".join(lines) + "\n"


def render_json(
    results: Sequence[CaseResult],
    stage: str,
    toolchain: Toolchain,
    jobs: int,
    case_timeout: float,
    duration_seconds: float,
) -> str:
    passed = sum(1 for result in results if result.passed)
    suite_counts: dict[str, dict[str, int]] = {}
    categories: dict[str, int] = {}
    for result in results:
        suite = suite_counts.setdefault(result.case.suite, {"pass": 0, "fail": 0})
        suite["pass" if result.passed else "fail"] += 1
        if not result.passed and result.category is not None:
            categories[result.category] = categories.get(result.category, 0) + 1
    payload = {
        "case_count": len(results),
        "case_timeout_seconds": case_timeout,
        "cases": [result.as_dict() for result in results],
        "duration_seconds": round(duration_seconds, 3),
        "failed": len(results) - passed,
        "failure_categories": categories,
        "jobs": jobs,
        "passed": passed,
        "stage": stage,
        "suite_counts": suite_counts,
        "target": {
            "builtin_preset": BUILTIN_TARGET_PRESET,
            "datalayout": LLVM_DATALAYOUT_LINE,
            "gcc_toolchain": str(toolchain.gcc_toolchain),
            "mabi": TARGET_MABI,
            "march": TARGET_MARCH,
            "sysroot": str(toolchain.sysroot),
            "triple": TARGET_TRIPLE,
        },
        "tools": {
            "cc": toolchain.cc,
            "cxx": toolchain.cxx,
            "pre_mapping": toolchain.pre_mapping,
            "raise": toolchain.raise_tool,
            "raise_opt": toolchain.raise_opt,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.jobs < 1:
        print("[corpus-gate] configuration error: --jobs must be >= 1", file=sys.stderr)
        return 2
    if args.case_timeout <= 0:
        print(
            "[corpus-gate] configuration error: --case-timeout must be > 0",
            file=sys.stderr,
        )
        return 2
    try:
        inventory = corpus_inventory.load_inventory(ROOT)
        selected = corpus_inventory.select_cases(
            inventory, suite_names=args.suites, case_ids=args.cases
        )
        external_root = corpus_inventory.resolve_externals_root(ROOT)
        toolchain = resolve_toolchain(args)
    except (corpus_inventory.InventoryError, GateConfigError) as exc:
        print(f"[corpus-gate] configuration error: {exc}", file=sys.stderr)
        return 2

    out_root = args.out_dir or (
        ROOT / "build" / "test-runs" / "corpus-gate" / args.stage
    )
    out_root = out_root.expanduser().resolve()
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"[corpus-gate] configuration error: cannot create {out_root}: {exc}",
            file=sys.stderr,
        )
        return 2

    started = time.monotonic()
    results = run_cases(
        selected,
        args.stage,
        toolchain,
        external_root,
        out_root,
        args.jobs,
        args.case_timeout,
    )
    duration = time.monotonic() - started

    sys.stdout.write(
        render_human(results, args.stage, toolchain, args.jobs, duration)
    )
    json_path = args.json_path or (out_root / "summary.json")
    try:
        json_path.expanduser().resolve().write_text(
            render_json(
                results,
                args.stage,
                toolchain,
                args.jobs,
                args.case_timeout,
                duration,
            )
        )
    except OSError as exc:
        print(
            f"[corpus-gate] configuration error: cannot write {json_path}: {exc}",
            file=sys.stderr,
        )
        return 2
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
