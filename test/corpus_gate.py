#!/usr/bin/env python3
"""Unified source and linked-workload corpus gate.

The ``llvm`` stage consumes SourceTranslationUnitInventory rows. All later
stages consume ProgramWorkloadInventory rows. Both are derived by
test/corpus_inventory.py and run through production compiler tools:

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
  graph/actor counts must describe at least one nonempty Spatial graph.
- stage ``dfg-sim``: runs the production pre-Mapping path and typed DFG
  simulator in one invocation. Each exact Spatial invocation is compared with
  the source program under the workload-owned runtime input. Graph-free,
  empty, or malformed executions fail.

A source row whose feature-guarded body is empty under the exact target can
still pass the ``llvm`` stage. It cannot stand in for a linked workload or a
Canonical Dataflow result. There are no source skips, no Unsupported-as-pass,
and no output placeholders. A workload whose exact profile requires another
instruction-set family reports a distinct Unsupported outcome before provider
setup; it is not counted as a semantic pass. Every other requested row either
passes its selected stage or fails with an honest category.

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
import corpus_target_profile  # noqa: E402
from corpus_gate_outcome import CaseOutcome, CaseResult  # noqa: E402
from corpus_gate_report import (  # noqa: E402
    CorpusGateReportContext,
    render_human,
    render_json,
)
from corpus_simulation_report import (  # noqa: E402
    DfgSimulationMetrics,
    parse_dfg_simulation_report,
)
from corpus_link_ownership import (  # noqa: E402
    LinkedWorkloadModules,
    load_compilation_owners,
    resolve_selected_corpus_sources,
)
from corpus_gate_scheduler import (  # noqa: E402
    CaseResourceLimiter as _CaseResourceLimiter,
    case_resource_slots,
    default_case_timeout,
    default_jobs,
)
from corpus_workload_provider import (  # noqa: E402
    CmakeToolchain,
    CmsisDspHarness,
    ProducedWorkload,
    WorkloadProviderError,
    cmake_build_command,
    cmake_configure_command,
    materialize_cmsis_dsp_harness,
    materialize_cmsis_nn_harness,
    supports_cmsis_nn_harness,
    supports_cmsis_dsp_harness,
)


# The exact builtin Fabric InstructionCore target profile. Every corpus
# source is compiled with exactly these flags and every emitted module is
# checked for exactly these target facts. The .ll triple and DataLayout are
# the pinned LLVM forms produced by clang for this target; the MLIR attribute
# is the triple carried through the LLVMIR import into the Structured and
# Canonical Dataflow modules.
TARGET_TRIPLE = "riscv64-unknown-elf"
TARGET_SINGLE_LETTER_EXTENSIONS = ("m", "a", "f", "d", "c")
TARGET_MULTI_LETTER_EXTENSIONS = ("zicsr", "zifencei")
TARGET_MARCH = (
    "rv64i"
    + "".join(TARGET_SINGLE_LETTER_EXTENSIONS)
    + "_"
    + "_".join(TARGET_MULTI_LETTER_EXTENSIONS)
)
TARGET_LTO_MATTR = ",".join(
    f"+{extension}"
    for extension in (
        *TARGET_SINGLE_LETTER_EXTENSIONS,
        *TARGET_MULTI_LETTER_EXTENSIONS,
    )
)
TARGET_MABI = "lp64d"
TARGET_CODE_MODEL = "medany"
LLVM_TRIPLE_LINE = 'target triple = "riscv64-unknown-unknown-elf"'
LLVM_DATALAYOUT_LINE = 'target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"'
MLIR_TRIPLE_ATTRIBUTE = 'llvm.target_triple = "riscv64-unknown-unknown-elf"'

# The one exact builtin Fabric target preset resolved by the d0 stage through
# loom-pre-mapping.
BUILTIN_TARGET_PRESET = "small"

STAGES = ("llvm", "s0", "d0", "dfg-sim")
WORKLOAD_STAGES = frozenset({"s0", "d0", "dfg-sim"})
TARGET_PROFILE = corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
STANDARD_FLOAT16_TARGET_PROFILE = corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE

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
CATEGORY_FINAL_LINK = "final-link"
CATEGORY_FINAL_LINK_ARTIFACT = "final-link-artifact"
CATEGORY_PAYLOAD_IMPORT = "payload-import"
CATEGORY_LINKED_LLVM_ARTIFACT = "linked-llvm-artifact"
CATEGORY_DFG_SIM = "dfg-sim"
CATEGORY_DFG_SIM_ARTIFACT = "dfg-sim-artifact"
CATEGORY_SOURCE_COVERAGE = "source-coverage"
CATEGORY_TARGET_PROFILE_UNSUPPORTED = "target-profile-unsupported"
CATEGORY_WORKLOAD_PROVIDER_UNAVAILABLE = "workload-provider-unavailable"
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
    "dfg_run": "LOOM_DFG_RUN",
    "lld": "LOOM_LLD",
    "payload": "LOOM_PAYLOAD",
    "llvm_dis": "LOOM_LLVM_DIS",
}
TOOL_FILE_NAMES = {
    "cc": "loom-cc",
    "cxx": "loom-c++",
    "raise": "loom-raise",
    "raise_opt": "loom-raise-opt",
    "pre_mapping": "loom-pre-mapping",
    "dfg_run": "loom-dfg-run",
    "lld": "ld.lld",
    "payload": "loom-payload",
    "llvm_dis": "llvm-dis",
}
LLVM_TOOL_KEYS = frozenset({"lld", "llvm_dis"})

DEFAULT_DFG_SIMULATION_TIMEOUT_SECONDS = 15.0
DEFAULT_PROVIDER_SETUP_TIMEOUT_SECONDS = 120.0
DEFAULT_DFG_MAX_WAVEFRONT_STEPS = 1_000_000
DEFAULT_DFG_MAX_EVENT_COUNT = 10_000_000
DEFAULT_DFG_MAX_CAPTURE_BYTES = 256 * 1024 * 1024


class GateConfigError(ValueError):
    """Raised when the gate cannot be configured honestly."""


@dataclass(frozen=True)
class Toolchain:
    cc: str
    cxx: str
    raise_tool: str
    raise_opt: str
    pre_mapping: str
    dfg_run: str
    lld: str
    payload: str
    llvm_dis: str
    sysroot: Path
    gcc_toolchain: Path


@dataclass(frozen=True)
class DfgExecutionLimits:
    max_wavefront_steps: int
    max_event_count: int
    max_capture_bytes: int

    def as_dict(self) -> dict[str, int]:
        return {
            "max_capture_bytes": self.max_capture_bytes,
            "max_event_count": self.max_event_count,
            "max_wavefront_steps": self.max_wavefront_steps,
        }


@dataclass(frozen=True)
class StepFailure:
    category: str
    detail: str


def run_quiet(command: Sequence[str], input_text: str | None = None) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            env={**os.environ, "LC_ALL": "C"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            input=input_text,
        )
    except OSError as exc:
        raise GateConfigError(f"cannot run {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or "unknown error"
        raise GateConfigError(f"{shlex.join(command)} failed: {diagnostic}")
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
        raise GateConfigError(f"{gcc}: unexpected GCC install layout: {install}")
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
        default_root = (
            ROOT / "externals" / "llvm" / "build" / "bin"
            if key in LLVM_TOOL_KEYS
            else ROOT / "build" / "bin"
        )
        candidate = os.environ.get(ENV_TOOL_NAMES[key]) or str(default_root / file_name)
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
        dfg_run=tools["dfg_run"],
        lld=tools["lld"],
        payload=tools["payload"],
        llvm_dis=tools["llvm_dis"],
        sysroot=sysroot,
        gcc_toolchain=gcc_toolchain,
    )


def target_flags(toolchain: Toolchain) -> list[str]:
    return [
        f"--target={TARGET_TRIPLE}",
        f"-march={TARGET_MARCH}",
        f"-mabi={TARGET_MABI}",
        f"-mcmodel={TARGET_CODE_MODEL}",
        f"--sysroot={toolchain.sysroot}",
        f"--gcc-toolchain={toolchain.gcc_toolchain}",
    ]


def target_profile_compile_flags(suite: str, target_profile: str) -> list[str]:
    resolved = corpus_target_profile.resolve_target_profile(
        suite, target_profile, TARGET_TRIPLE
    )
    if (
        resolved.disposition
        is not corpus_target_profile.TargetProfileDisposition.RUNNABLE
    ):
        raise GateConfigError(resolved.detail)
    return list(resolved.compile_flags)


def suite_compile_flags(
    suite: str,
    external_root: Path,
    *,
    target_profile: str = TARGET_PROFILE,
) -> list[str]:
    """Uniform per-suite source configuration; never per-source overrides."""
    if suite == "loombench":
        flags: list[str] = []
    elif suite == "cmsis-dsp":
        root = external_root / "cmsis-dsp"
        # __GNUC_PYTHON__ is the official CMSIS-DSP non-Arm scalar source
        # configuration; it disables the CMSIS-Core dependency.
        flags = [
            "-D__GNUC_PYTHON__",
            f"-I{root / 'Include'}",
            f"-I{root / 'PrivateInclude'}",
        ]
    elif suite == "cmsis-nn":
        flags = [f"-I{external_root / 'cmsis-nn' / 'Include'}"]
    else:
        raise GateConfigError(f"unknown suite: {suite}")
    flags.extend(target_profile_compile_flags(suite, target_profile))
    return flags


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
        "-gline-tables-only",
        str(source),
        "-o",
        str(output),
    ]


def target_object_command(
    toolchain: Toolchain,
    suite_flags: Sequence[str],
    source: Path,
    output: Path,
) -> list[str]:
    return [
        compiler_for(toolchain, source),
        *target_flags(toolchain),
        *suite_flags,
        "-O1",
        "-gline-tables-only",
        "-flto=full",
        "-ffat-lto-objects",
        "-c",
        str(source),
        "-o",
        str(output),
    ]


def final_link_command(
    toolchain: Toolchain,
    sources: Sequence[Path],
    objects: Sequence[Path],
    link_flags: Sequence[str],
    output: Path,
) -> list[str]:
    driver = (
        toolchain.cxx
        if any(source.suffix in CXX_SUFFIXES for source in sources)
        else toolchain.cc
    )
    return [
        driver,
        *target_flags(toolchain),
        f"-fuse-ld={toolchain.lld}",
        "-O1",
        "-flto=full",
        "-Wl,--fat-lto-objects",
        "-Wl,--save-temps=resolution",
        "-Wl,--save-temps=precodegen",
        "-Wl,--lto-O1",
        "-Xlinker",
        f"--plugin-opt=-mattr={TARGET_LTO_MATTR}",
        *(str(path) for path in objects),
        *link_flags,
        "-o",
        str(output),
    ]


def payload_import_command(
    toolchain: Toolchain,
    executable: Path,
    bitcode_output: Path,
) -> list[str]:
    return [
        toolchain.payload,
        f"--resolution={executable}.resolution.txt",
        f"--linked-bitcode={executable}.0.5.precodegen.bc",
        f"--bitcode-output={bitcode_output}",
    ]


def disassemble_command(
    toolchain: Toolchain, bitcode: Path, llvm_ir: Path
) -> list[str]:
    return [toolchain.llvm_dis, str(bitcode), "-o", str(llvm_ir)]


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
    candidate_jobs: int,
    protocol_symbols: Sequence[str],
    config_path: Path | None = None,
) -> list[str]:
    command = [
        toolchain.pre_mapping,
        f"--builtin={BUILTIN_TARGET_PRESET}",
        f"--artifact-store={store_dir}",
        f"--counts={counts}",
        f"--candidate-jobs={candidate_jobs}",
        str(llvm_ir),
        "-o",
        str(d0_module),
    ]
    command[-2:-2] = [
        f"--operator-protocol-symbol={symbol}" for symbol in protocol_symbols
    ]
    if config_path is not None:
        command.insert(2, f"--config={config_path}")
    return command


def dfg_sim_command(
    toolchain: Toolchain,
    target_llvm_ir: Path,
    store_dir: Path,
    d0_module: Path,
    report: Path,
    candidate_jobs: int,
    limits: DfgExecutionLimits,
    simulation_timeout: float,
    protocol_symbols: Sequence[str],
    config_path: Path | None = None,
    expected_entry_result: int | None = None,
) -> list[str]:
    command = [
        toolchain.dfg_run,
        f"--builtin={BUILTIN_TARGET_PRESET}",
        f"--artifact-store={store_dir}",
        f"--canonical-output={d0_module}",
        f"--output={report}",
        f"--candidate-jobs={candidate_jobs}",
        f"--max-event-steps={limits.max_wavefront_steps}",
        f"--max-event-count={limits.max_event_count}",
        f"--max-capture-bytes={limits.max_capture_bytes}",
        f"--max-simulation-wall-seconds={simulation_timeout}",
        str(target_llvm_ir),
    ]
    command[-1:-1] = [
        f"--operator-protocol-symbol={symbol}" for symbol in protocol_symbols
    ]
    if expected_entry_result is not None:
        command.insert(
            -1 - len(protocol_symbols),
            f"--expected-entry-result={expected_entry_result}",
        )
    if config_path is not None:
        command.insert(2, f"--config={config_path}")
    return command


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
    *,
    cwd: Path | None = None,
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
                    cwd=cwd,
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


def case_out_dir(
    out_root: Path,
    case: corpus_inventory.SourceTranslationUnit | corpus_inventory.ProgramWorkload,
) -> Path:
    if isinstance(case, corpus_inventory.ProgramWorkload):
        operator_digest = case.operator_id.rsplit(":", 1)[-1]
        return out_root / case.suite / case.case / operator_digest
    return out_root / case.suite / case.case.removesuffix(".c")


def clear_artifacts(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def binary_artifact_defect(path: Path, role: str) -> str | None:
    try:
        if not path.is_file():
            return f"missing {role}: {path}"
        if path.stat().st_size == 0:
            return f"empty {role}: {path}"
    except OSError as exc:
        return f"cannot inspect {role} {path}: {exc}"
    return None


def prepare_linked_workload(
    case: corpus_inventory.ProgramWorkload,
    toolchain: Toolchain,
    external_root: Path,
    case_dir: Path,
    deadline: float,
) -> LinkedWorkloadModules | StepFailure:
    if not case.sources:
        return StepFailure(
            CATEGORY_FINAL_LINK,
            "direct-source workload has no source inputs",
        )

    suite_flags = suite_compile_flags(
        case.suite,
        external_root,
        target_profile=case.target_profile,
    )
    suite_flags.extend(case.compiler_flags)
    sources = [resolve_source(path, external_root) for path in case.sources]
    target_objects: list[Path] = []
    for ordinal, (repo_relative, source) in enumerate(
        zip(case.sources, sources, strict=True)
    ):
        target_object = case_dir / f"source-{ordinal:03d}.o"
        clear_artifacts(target_object)
        failure = run_step(
            target_object_command(toolchain, suite_flags, source, target_object),
            case_dir / f"source-{ordinal:03d}.compile.log",
            deadline,
            CATEGORY_COMPILE,
        )
        if failure is not None:
            return StepFailure(failure.category, f"{repo_relative}: {failure.detail}")
        defect = binary_artifact_defect(target_object, "target object")
        if defect is not None:
            return StepFailure(CATEGORY_LLVM_ARTIFACT, f"{repo_relative}: {defect}")
        target_objects.append(target_object)

    executable = case_dir / "program.elf"
    resolution = Path(f"{executable}.resolution.txt")
    precodegen = Path(f"{executable}.0.5.precodegen.bc")
    clear_artifacts(executable, resolution, precodegen)
    failure = run_step(
        final_link_command(
            toolchain, sources, target_objects, case.link_flags, executable
        ),
        case_dir / "final-link.log",
        deadline,
        CATEGORY_FINAL_LINK,
    )
    if failure is not None:
        return failure
    for path, role in (
        (executable, "linked executable"),
        (resolution, "LLD resolution report"),
        (precodegen, "LLD pre-code-generation bitcode"),
    ):
        defect = binary_artifact_defect(path, role)
        if defect is not None:
            return StepFailure(CATEGORY_FINAL_LINK_ARTIFACT, defect)

    linked_bitcode = case_dir / "program.bc"
    clear_artifacts(linked_bitcode)
    failure = run_step(
        payload_import_command(toolchain, executable, linked_bitcode),
        case_dir / "payload-import.log",
        deadline,
        CATEGORY_PAYLOAD_IMPORT,
    )
    if failure is not None:
        return failure
    defect = binary_artifact_defect(linked_bitcode, "validated linked bitcode")
    if defect is not None:
        return StepFailure(CATEGORY_LINKED_LLVM_ARTIFACT, defect)

    linked_llvm_ir = case_dir / "program.ll"
    clear_artifacts(linked_llvm_ir)
    failure = run_step(
        disassemble_command(toolchain, linked_bitcode, linked_llvm_ir),
        case_dir / "llvm-dis.log",
        deadline,
        CATEGORY_LINKED_LLVM_ARTIFACT,
    )
    if failure is not None:
        return failure
    defect = llvm_ir_defect(linked_llvm_ir)
    if defect is not None:
        return StepFailure(CATEGORY_LINKED_LLVM_ARTIFACT, defect)

    return LinkedWorkloadModules(
        target=linked_llvm_ir,
        resolution=resolution,
        link_root=case_dir,
        object_sources=tuple(
            (output.resolve(), resolve_source(source, external_root).resolve())
            for output, source in zip(target_objects, case.sources, strict=True)
        ),
    )


def import_produced_workload(
    produced: ProducedWorkload,
    toolchain: Toolchain,
    case_dir: Path,
    deadline: float,
) -> LinkedWorkloadModules | StepFailure:
    try:
        relative_executable = produced.target_executable.relative_to(
            produced.target_build_dir
        )
    except ValueError:
        return StepFailure(
            CATEGORY_FINAL_LINK_ARTIFACT,
            "produced executable is outside its owning build directory",
        )
    resolution = Path(f"{produced.target_executable}.resolution.txt")
    precodegen = Path(f"{produced.target_executable}.0.5.precodegen.bc")
    compilation_database = produced.target_build_dir / "compile_commands.json"
    for path, role in (
        (produced.target_executable, "linked executable"),
        (resolution, "LLD resolution report"),
        (precodegen, "LLD pre-code-generation bitcode"),
        (compilation_database, "exact compilation database"),
    ):
        defect = binary_artifact_defect(path, role)
        if defect is not None:
            return StepFailure(CATEGORY_FINAL_LINK_ARTIFACT, defect)

    linked_bitcode = case_dir / "program.bc"
    clear_artifacts(linked_bitcode)
    failure = run_step(
        payload_import_command(toolchain, relative_executable, linked_bitcode),
        case_dir / "payload-import.log",
        deadline,
        CATEGORY_PAYLOAD_IMPORT,
        cwd=produced.target_build_dir,
    )
    if failure is not None:
        return failure
    defect = binary_artifact_defect(linked_bitcode, "validated linked bitcode")
    if defect is not None:
        return StepFailure(CATEGORY_LINKED_LLVM_ARTIFACT, defect)

    linked_llvm_ir = case_dir / "program.ll"
    clear_artifacts(linked_llvm_ir)
    failure = run_step(
        disassemble_command(toolchain, linked_bitcode, linked_llvm_ir),
        case_dir / "llvm-dis.log",
        deadline,
        CATEGORY_LINKED_LLVM_ARTIFACT,
    )
    if failure is not None:
        return failure
    defect = llvm_ir_defect(linked_llvm_ir)
    if defect is not None:
        return StepFailure(CATEGORY_LINKED_LLVM_ARTIFACT, defect)

    object_sources, defect = load_compilation_owners(compilation_database)
    if defect is not None:
        return StepFailure(CATEGORY_FINAL_LINK_ARTIFACT, defect)
    assert object_sources is not None
    return LinkedWorkloadModules(
        target=linked_llvm_ir,
        resolution=resolution,
        link_root=produced.target_build_dir,
        object_sources=object_sources,
        inline_definition_sources=produced.inline_definition_sources,
    )


def _cmsis_cmake_toolchain(
    toolchain: Toolchain,
    suite: str,
    target_profile: str,
) -> CmakeToolchain:
    llvm_bin = Path(toolchain.llvm_dis).parent
    compiler_flags = [
        *target_flags(toolchain),
        *target_profile_compile_flags(suite, target_profile),
        "-gline-tables-only",
        "-flto=full",
        "-ffat-lto-objects",
    ]
    linker_flags = [
        f"-fuse-ld={toolchain.lld}",
        "-flto=full",
        "-Wl,--fat-lto-objects",
        "-Wl,--save-temps=precodegen",
        "-Wl,--save-temps=resolution",
        "-Wl,--lto-O1",
        "-Xlinker",
        f"--plugin-opt=-mattr={TARGET_LTO_MATTR}",
    ]
    return CmakeToolchain(
        c_compiler=toolchain.cc,
        cxx_compiler=toolchain.cxx,
        archiver=llvm_bin / "llvm-ar",
        ranlib=llvm_bin / "llvm-ranlib",
        compiler_flags=tuple(compiler_flags),
        linker_flags=tuple(linker_flags),
        system_name="Generic",
    )


def prepare_workload_providers(
    cases: Sequence[corpus_inventory.ProgramWorkload],
    toolchain: Toolchain,
    external_root: Path,
    out_root: Path,
    jobs: int,
    timeout: float,
) -> dict[str, ProducedWorkload | StepFailure]:
    results: dict[str, ProducedWorkload | StepFailure] = {}
    profile_resolutions = {
        case.identity: corpus_target_profile.resolve_target_profile(
            case.suite, case.target_profile, TARGET_TRIPLE
        )
        for case in cases
    }
    runnable_profiles = {
        identity
        for identity, resolved in profile_resolutions.items()
        if resolved.disposition
        is corpus_target_profile.TargetProfileDisposition.RUNNABLE
    }
    cmsis_nn = [
        case
        for case in cases
        if isinstance(
            case.producer,
            (
                corpus_inventory.CmsisNnWorkloadProducer,
                corpus_inventory.CmsisNnGeneratedWorkloadProducer,
            ),
        )
        and case.identity in runnable_profiles
        and supports_cmsis_nn_harness(case)
    ]
    cmsis_dsp = [
        case
        for case in cases
        if isinstance(
            case.producer,
            (
                corpus_inventory.CmsisDspWorkloadProducer,
                corpus_inventory.CmsisDspGeneratedWorkloadProducer,
            ),
        )
        and case.identity in runnable_profiles
        and supports_cmsis_dsp_harness(case)
    ]
    implemented = {case.identity for case in (*cmsis_nn, *cmsis_dsp)}
    for case in cases:
        resolved = profile_resolutions[case.identity]
        if (
            resolved.disposition
            is corpus_target_profile.TargetProfileDisposition.INCOMPATIBLE_ISA
        ):
            continue
        if (
            resolved.disposition
            is corpus_target_profile.TargetProfileDisposition.PROVIDER_UNAVAILABLE
        ):
            results[case.identity] = StepFailure(
                CATEGORY_WORKLOAD_PROVIDER_UNAVAILABLE,
                resolved.detail,
            )
        elif case.producer.kind != "direct-source" and case.identity not in implemented:
            results[case.identity] = StepFailure(
                CATEGORY_WORKLOAD_PROVIDER_UNAVAILABLE,
                "no linked-workload builder is available for producer kind "
                f"{case.producer.kind}",
            )

    for provider_name, all_provider_cases in (
        ("cmsis-nn", cmsis_nn),
        ("cmsis-dsp", cmsis_dsp),
    ):
        profiles = sorted({case.target_profile for case in all_provider_cases})
        for target_profile in profiles:
            provider_cases = [
                case
                for case in all_provider_cases
                if case.target_profile == target_profile
            ]
            provider_root = out_root / "_providers" / provider_name
            if target_profile != TARGET_PROFILE:
                provider_root /= target_profile

            def fail_provider(failure: StepFailure) -> None:
                results.update({case.identity: failure for case in provider_cases})

            try:
                if provider_root.exists():
                    shutil.rmtree(provider_root)
                provider_root.mkdir(parents=True)
                if provider_name == "cmsis-nn":
                    harness = materialize_cmsis_nn_harness(
                        provider_cases, external_root, provider_root / "harness"
                    )
                else:
                    harness = materialize_cmsis_dsp_harness(
                        provider_cases, external_root, provider_root / "harness"
                    )
            except (OSError, WorkloadProviderError) as exc:
                fail_provider(StepFailure(CATEGORY_FINAL_LINK, str(exc)))
                continue

            target_build = provider_root / "target"
            failure = run_step(
                cmake_configure_command(
                    harness,
                    external_root / provider_name,
                    target_build,
                    _cmsis_cmake_toolchain(
                        toolchain,
                        provider_name,
                        target_profile,
                    ),
                ),
                provider_root / "target-configure.log",
                time.monotonic() + timeout,
                CATEGORY_FINAL_LINK,
            )
            if failure is not None:
                fail_provider(failure)
                continue
            failure = run_step(
                cmake_build_command(target_build, harness.targets, jobs),
                provider_root / "target-build.log",
                time.monotonic() + timeout,
                CATEGORY_FINAL_LINK,
            )
            for case in provider_cases:
                target_executable = harness.executable(target_build, case.executable)
                if target_executable.is_file():
                    try:
                        protocol_symbols = harness.protocol_symbols(case.executable)
                        if not protocol_symbols:
                            if not isinstance(harness, CmsisDspHarness):
                                raise WorkloadProviderError(
                                    "CMSIS-NN provider produced an empty protocol"
                                )
                            if len(case.protocol) != 1:
                                raise WorkloadProviderError(
                                    "multi-call CMSIS-DSP workload requires an "
                                    "atomic protocol wrapper"
                                )
                            protocol_symbols = (case.protocol[0].symbol,)
                    except (GateConfigError, WorkloadProviderError) as exc:
                        results[case.identity] = StepFailure(
                            CATEGORY_FINAL_LINK_ARTIFACT, str(exc)
                        )
                        continue
                    results[case.identity] = ProducedWorkload(
                        target_build_dir=target_build,
                        target_executable=target_executable,
                        protocol_symbols=protocol_symbols,
                        expected_entry_result=harness.expected_entry_result(
                            case.executable
                        ),
                        inline_definition_sources=harness.inline_definitions(
                            case.executable
                        ),
                    )
                    continue
                if failure is None:
                    results[case.identity] = StepFailure(
                        CATEGORY_FINAL_LINK,
                        f"provider did not produce target {case.executable}",
                    )
                    continue
                results[case.identity] = StepFailure(
                    failure.category,
                    f"provider did not produce target {case.executable}: "
                    f"{failure.detail}",
                )
    return results


def run_case(
    case: corpus_inventory.SourceTranslationUnit | corpus_inventory.ProgramWorkload,
    stage: str,
    toolchain: Toolchain,
    external_root: Path,
    out_root: Path,
    case_timeout: float,
    candidate_jobs: int,
    config_path: Path | None,
    dfg_limits: DfgExecutionLimits,
    dfg_simulation_timeout: float,
    provider_results: dict[str, ProducedWorkload | StepFailure],
    allowed_sources: frozenset[str],
) -> CaseResult:
    started = time.monotonic()
    deadline = started + case_timeout
    graphs = 0
    actors = 0
    dfg_totals = DfgSimulationMetrics.zero()
    selected_sources: tuple[str, ...] | None = None

    def finish(
        category: str | None,
        detail: str | None,
        outcome: CaseOutcome | None = None,
    ) -> CaseResult:
        if outcome is None:
            outcome = CaseOutcome.PASS if category is None else CaseOutcome.FAIL
        passed = outcome is CaseOutcome.PASS
        return CaseResult(
            case=case,
            outcome=outcome,
            category=category,
            detail=detail,
            duration_seconds=time.monotonic() - started,
            graphs=graphs if passed and stage in {"d0", "dfg-sim"} else None,
            actors=actors if passed and stage in {"d0", "dfg-sim"} else None,
            dfg_simulation=(dfg_totals if passed and stage == "dfg-sim" else None),
            selected_sources=(
                selected_sources if passed and stage == "dfg-sim" else None
            ),
        )

    try:
        if stage in WORKLOAD_STAGES:
            if not isinstance(case, corpus_inventory.ProgramWorkload):
                return finish(
                    CATEGORY_INTERNAL,
                    "whole-program stage received a source translation-unit row",
                )
        case_dir = case_out_dir(out_root, case)
        case_dir.mkdir(parents=True, exist_ok=True)
        store_dir = case_dir / "artifact-store"
        if stage in {"d0", "dfg-sim"}:
            store_dir.mkdir(parents=True, exist_ok=True)

        if stage == "llvm":
            suite_flags = suite_compile_flags(case.suite, external_root)
            for ordinal, repo_relative in enumerate(case.sources):
                source = resolve_source(repo_relative, external_root)
                llvm_ir = case_dir / f"source-{ordinal:03d}-{source.stem}.ll"
                clear_artifacts(llvm_ir)
                failure = run_step(
                    compile_command(toolchain, suite_flags, source, llvm_ir),
                    case_dir / f"source-{ordinal:03d}.compile.log",
                    deadline,
                    CATEGORY_COMPILE,
                )
                if failure is not None:
                    return finish(
                        failure.category, f"{repo_relative}: {failure.detail}"
                    )
                defect = llvm_ir_defect(llvm_ir)
                if defect is not None:
                    return finish(CATEGORY_LLVM_ARTIFACT, f"{repo_relative}: {defect}")
            return finish(None, None)

        assert isinstance(case, corpus_inventory.ProgramWorkload)
        resolved_profile = corpus_target_profile.resolve_target_profile(
            case.suite, case.target_profile, TARGET_TRIPLE
        )
        if (
            resolved_profile.disposition
            is corpus_target_profile.TargetProfileDisposition.INCOMPATIBLE_ISA
        ):
            return finish(
                CATEGORY_TARGET_PROFILE_UNSUPPORTED,
                resolved_profile.detail,
                CaseOutcome.UNSUPPORTED,
            )
        if (
            resolved_profile.disposition
            is corpus_target_profile.TargetProfileDisposition.PROVIDER_UNAVAILABLE
        ):
            return finish(
                CATEGORY_WORKLOAD_PROVIDER_UNAVAILABLE,
                resolved_profile.detail,
            )
        expected_entry_result: int | None = None
        if case.producer.kind == "direct-source":
            protocol_symbols = tuple(call.symbol for call in case.protocol)
            prepared = prepare_linked_workload(
                case,
                toolchain,
                external_root,
                case_dir,
                deadline,
            )
        else:
            produced = provider_results.get(case.identity)
            if produced is None:
                return finish(
                    CATEGORY_WORKLOAD_PROVIDER_UNAVAILABLE,
                    "no linked-workload builder is available for producer kind "
                    f"{case.producer.kind}",
                )
            if isinstance(produced, StepFailure):
                return finish(produced.category, produced.detail)
            protocol_symbols = produced.protocol_symbols
            expected_entry_result = produced.expected_entry_result
            prepared = import_produced_workload(
                produced,
                toolchain,
                case_dir,
                deadline,
            )
        if isinstance(prepared, StepFailure):
            return finish(prepared.category, prepared.detail)

        if stage == "dfg-sim":
            d0_module = case_dir / "program.dfg.mlir"
            report_path = case_dir / "program.dfg-sim.json"
            clear_artifacts(d0_module, report_path)
            failure = run_step(
                dfg_sim_command(
                    toolchain,
                    prepared.target,
                    store_dir,
                    d0_module,
                    report_path,
                    candidate_jobs,
                    dfg_limits,
                    dfg_simulation_timeout,
                    protocol_symbols,
                    config_path,
                    expected_entry_result,
                ),
                case_dir / "dfg-sim.log",
                deadline,
                CATEGORY_DFG_SIM,
            )
            if failure is not None:
                return finish(failure.category, failure.detail)
            defect = d0_module_defect(d0_module)
            if defect is not None:
                return finish(CATEGORY_D0_ARTIFACT, defect)
            report, defect = parse_dfg_simulation_report(report_path)
            if defect is not None:
                return finish(CATEGORY_DFG_SIM_ARTIFACT, defect)
            assert report is not None
            selected_sources, defect = resolve_selected_corpus_sources(
                prepared,
                report.selected_source_files,
                external_root,
                ROOT,
                allowed_sources,
            )
            if defect is not None:
                return finish(CATEGORY_SOURCE_COVERAGE, defect)
            graphs = report.graphs
            actors = report.actors
            dfg_totals = report
            return finish(None, None)

        if stage == "d0":
            d0_module = case_dir / "program.dfg.mlir"
            counts_path = case_dir / "program.counts.json"
            clear_artifacts(d0_module, counts_path)
            failure = run_step(
                pre_mapping_command(
                    toolchain,
                    prepared.target,
                    store_dir,
                    d0_module,
                    counts_path,
                    candidate_jobs,
                    protocol_symbols,
                    config_path,
                ),
                case_dir / "pre-mapping.log",
                deadline,
                CATEGORY_PRE_MAPPING,
            )
            if failure is not None:
                return finish(failure.category, failure.detail)
            defect = d0_module_defect(d0_module)
            if defect is not None:
                return finish(CATEGORY_D0_ARTIFACT, defect)
            counts, defect = parse_d0_counts(counts_path)
            if defect is not None:
                return finish(CATEGORY_D0_ARTIFACT, defect)
            assert counts is not None
            graphs = counts["graphs"]
            actors = counts["actors"]
            if graphs == 0 or actors == 0:
                return finish(
                    CATEGORY_PRE_MAPPING,
                    "pre-Mapping selected no nonempty Spatial graph",
                )
            return finish(None, None)

        s0_module = case_dir / "program.scf.mlir"
        clear_artifacts(s0_module)
        failure = run_step(
            raise_command(toolchain, prepared.target, s0_module),
            case_dir / "raise.log",
            deadline,
            CATEGORY_RAISE,
        )
        if failure is not None:
            return finish(failure.category, failure.detail)
        defect = s0_module_defect(s0_module)
        if defect is not None:
            return finish(CATEGORY_S0_ARTIFACT, defect)
        failure = run_step(
            verify_command(toolchain, s0_module),
            case_dir / "verify.log",
            deadline,
            CATEGORY_VERIFY,
        )
        if failure is not None:
            return finish(failure.category, failure.detail)
        return finish(None, None)
    except (GateConfigError, OSError) as exc:
        return finish(CATEGORY_INTERNAL, f"gate error: {exc}")
    except Exception:  # noqa: BLE001 - a runner bug must not pass or hang the gate
        return finish(CATEGORY_INTERNAL, traceback.format_exc().rstrip())


def run_cases(
    cases: Sequence[
        corpus_inventory.SourceTranslationUnit | corpus_inventory.ProgramWorkload
    ],
    stage: str,
    toolchain: Toolchain,
    external_root: Path,
    out_root: Path,
    jobs: int,
    case_timeout: float,
    candidate_jobs: int,
    config_path: Path | None,
    dfg_limits: DfgExecutionLimits,
    dfg_simulation_timeout: float,
) -> list[CaseResult]:
    source_rows = (
        corpus_inventory.load_source_inventory(ROOT) if stage == "dfg-sim" else ()
    )
    allowed_sources_by_suite: dict[str, frozenset[str]] = {
        suite: frozenset(
            source
            for row in source_rows
            if row.suite == suite
            for source in row.sources
        )
        for suite in corpus_inventory.SUITE_ORDER
    }
    workload_cases = [
        case for case in cases if isinstance(case, corpus_inventory.ProgramWorkload)
    ]
    provider_results = prepare_workload_providers(
        workload_cases,
        toolchain,
        external_root,
        out_root,
        jobs,
        max(DEFAULT_PROVIDER_SETUP_TIMEOUT_SECONDS, case_timeout),
    )
    results: list[CaseResult | None] = [None] * len(cases)
    limiter = _CaseResourceLimiter(jobs)

    def execute(case):
        slots = case_resource_slots(case, stage, jobs)
        limiter.acquire(slots)
        try:
            return run_case(
                case,
                stage,
                toolchain,
                external_root,
                out_root,
                case_timeout,
                candidate_jobs,
                config_path,
                dfg_limits,
                dfg_simulation_timeout,
                provider_results,
                allowed_sources_by_suite[case.suite],
            )
        finally:
            limiter.release(slots)

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(execute, case): index for index, case in enumerate(cases)
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    for result in results:
        if result is None:  # unreachable: run_case never raises
            raise AssertionError("missing case result")
    return [result for result in results if result is not None]


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
        default=None,
        metavar="SECONDS",
        help="per-case wall-clock deadline; the case process group is killed "
        "when it expires (default: 30 for dfg-sim, 120 otherwise)",
    )
    parser.add_argument(
        "--dfg-simulation-timeout",
        type=float,
        default=DEFAULT_DFG_SIMULATION_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="aggregate DFG event-replay wall-time limit per evaluated "
        "candidate (default: %(default)s)",
    )
    parser.add_argument(
        "--candidate-jobs",
        type=int,
        default=1,
        help="bounded ownership-candidate workers within each d0/dfg-sim case "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--dfg-max-wavefront-steps",
        type=int,
        default=DEFAULT_DFG_MAX_WAVEFRONT_STEPS,
        help="maximum aggregate DFG wavefront steps before an incomplete "
        "execution (default: %(default)s)",
    )
    parser.add_argument(
        "--dfg-max-event-count",
        type=int,
        default=DEFAULT_DFG_MAX_EVENT_COUNT,
        help="maximum aggregate DFG events before an incomplete execution "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--dfg-max-capture-bytes",
        type=int,
        default=DEFAULT_DFG_MAX_CAPTURE_BYTES,
        help="maximum retained source-backed capture bytes before an "
        "incomplete execution (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="resolved semantic configuration forwarded to pre-Mapping tools",
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
    args = parser.parse_args(argv)
    if args.case_timeout is None:
        args.case_timeout = default_case_timeout(args.stage)
    return args


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
    if args.dfg_simulation_timeout <= 0:
        print(
            "[corpus-gate] configuration error: --dfg-simulation-timeout must be > 0",
            file=sys.stderr,
        )
        return 2
    if args.candidate_jobs < 1:
        print(
            "[corpus-gate] configuration error: --candidate-jobs must be >= 1",
            file=sys.stderr,
        )
        return 2
    if (
        args.dfg_max_wavefront_steps < 1
        or args.dfg_max_event_count < 1
        or args.dfg_max_capture_bytes < 1
    ):
        print(
            "[corpus-gate] configuration error: DFG execution limits must be positive",
            file=sys.stderr,
        )
        return 2
    dfg_limits = DfgExecutionLimits(
        args.dfg_max_wavefront_steps,
        args.dfg_max_event_count,
        args.dfg_max_capture_bytes,
    )
    try:
        inventory = (
            corpus_inventory.load_workload_inventory(ROOT)
            if args.stage in WORKLOAD_STAGES
            else corpus_inventory.load_source_inventory(ROOT)
        )
        selected = corpus_inventory.select_rows(
            inventory, suite_names=args.suites, case_ids=args.cases
        )
        external_root = corpus_inventory.resolve_externals_root(ROOT)
        toolchain = resolve_toolchain(args)
        config_path = None
        if args.config is not None:
            config_path = args.config.expanduser().resolve()
            if not config_path.is_file():
                raise GateConfigError(
                    f"resolved configuration is not a file: {config_path}"
                )
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
        args.candidate_jobs,
        config_path,
        dfg_limits,
        args.dfg_simulation_timeout,
    )
    duration = time.monotonic() - started
    report_context = CorpusGateReportContext(
        stage=args.stage,
        jobs=args.jobs,
        candidate_jobs=args.candidate_jobs,
        case_timeout_seconds=args.case_timeout,
        dfg_simulation_timeout_seconds=args.dfg_simulation_timeout,
        dfg_execution_limits=dfg_limits.as_dict(),
        config=str(config_path) if config_path is not None else None,
        duration_seconds=duration,
        human_header=(
            f"[corpus-gate] stage={args.stage} target={TARGET_TRIPLE} "
            f"march={TARGET_MARCH} mabi={TARGET_MABI} "
            f"code-model={TARGET_CODE_MODEL} builtin={BUILTIN_TARGET_PRESET} "
            f"candidate-jobs={args.candidate_jobs} "
            f"dfg-limits={dfg_limits.max_wavefront_steps}/"
            f"{dfg_limits.max_event_count}/{dfg_limits.max_capture_bytes} "
            f"dfg-wall-time={args.dfg_simulation_timeout:g}s "
            f"config={config_path if config_path is not None else '<default>'} "
            f"sysroot={toolchain.sysroot} "
            f"gcc-toolchain={toolchain.gcc_toolchain}"
        ),
        target={
            "builtin_preset": BUILTIN_TARGET_PRESET,
            "code_model": TARGET_CODE_MODEL,
            "datalayout": LLVM_DATALAYOUT_LINE,
            "gcc_toolchain": str(toolchain.gcc_toolchain),
            "mabi": TARGET_MABI,
            "march": TARGET_MARCH,
            "sysroot": str(toolchain.sysroot),
            "triple": TARGET_TRIPLE,
        },
        tools={
            "cc": toolchain.cc,
            "cxx": toolchain.cxx,
            "dfg_run": toolchain.dfg_run,
            "lld": toolchain.lld,
            "llvm_dis": toolchain.llvm_dis,
            "payload": toolchain.payload,
            "pre_mapping": toolchain.pre_mapping,
            "raise": toolchain.raise_tool,
            "raise_opt": toolchain.raise_opt,
        },
    )

    sys.stdout.write(render_human(results, report_context))
    json_path = args.json_path or (out_root / "summary.json")
    try:
        json_path.expanduser().resolve().write_text(
            render_json(results, report_context)
        )
    except OSError as exc:
        print(
            f"[corpus-gate] configuration error: cannot write {json_path}: {exc}",
            file=sys.stderr,
        )
        return 2
    return 0 if all(not result.failed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
