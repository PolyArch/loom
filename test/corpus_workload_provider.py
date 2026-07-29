#!/usr/bin/env python3
"""Ephemeral builders for corpus-owned linked program workloads."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import corpus_inventory


class WorkloadProviderError(ValueError):
    """Raised when an owned workload cannot be materialized exactly."""


@dataclass(frozen=True)
class CmsisNnHarness:
    source_dir: Path
    unity_source: Path
    targets: tuple[str, ...]

    def executable(self, build_dir: Path, target: str) -> Path:
        if target not in self.targets:
            raise WorkloadProviderError(f"unknown CMSIS-NN harness target: {target}")
        return build_dir / "workloads" / target


@dataclass(frozen=True)
class ProducedWorkload:
    target_build_dir: Path
    target_executable: Path
    native_bitcode: Path | None


@dataclass(frozen=True)
class CmakeToolchain:
    c_compiler: str
    cxx_compiler: str
    archiver: Path
    ranlib: Path
    compiler_flags: tuple[str, ...]
    linker_flags: tuple[str, ...]
    system_name: str | None = None


_UNITY_TEST = re.compile(
    r"(?m)^\s*void\s+(test_[A-Za-z0-9_]+)\s*\(\s*void\s*\)\s*\{"
)


def _unity_tests(source: Path) -> tuple[str, ...]:
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkloadProviderError(f"cannot read Unity wrapper {source}: {exc}") from exc
    tests = tuple(_UNITY_TEST.findall(text))
    if not tests:
        raise WorkloadProviderError(f"Unity wrapper defines no tests: {source}")
    if len(tests) != len(set(tests)):
        raise WorkloadProviderError(f"Unity wrapper repeats a test function: {source}")
    return tests


def _render_unity_runner(tests: Sequence[str]) -> str:
    declarations = "\n".join(f"void {test}(void);" for test in tests)
    invocations = "\n".join(f"  RUN_TEST({test});" for test in tests)
    return (
        '#include "unity.h"\n\n'
        "void setUp(void);\n"
        "void tearDown(void);\n"
        f"{declarations}\n\n"
        "int main(void) {\n"
        "  UNITY_BEGIN();\n"
        f"{invocations}\n"
        "  return UNITY_END();\n"
        "}\n"
    )


def _case_source(
    workload: corpus_inventory.ProgramWorkload, external_root: Path
) -> Path:
    definition = Path(workload.producer.definition)
    expected_prefix = Path("externals/cmsis-nn/Tests/UnitTest/TestCases")
    try:
        relative = definition.relative_to(expected_prefix)
    except ValueError as exc:
        raise WorkloadProviderError(
            f"CMSIS-NN producer definition escapes its owner: {definition}"
        ) from exc
    if relative.name != "CMakeLists.txt" or len(relative.parts) != 2:
        raise WorkloadProviderError(
            f"CMSIS-NN producer definition is not one exact case: {definition}"
        )
    return external_root / "cmsis-nn" / "Tests" / "UnitTest" / "TestCases" / relative.parent


def _render_harness_cmake(case_directories: Sequence[str], targets: Sequence[str]) -> str:
    add_cases = "\n".join(
        f'add_subdirectory("TestCases/{case}" "cases/{case}")'
        for case in case_directories
    )
    target_items = " ".join(targets)
    return f"""cmake_minimum_required(VERSION 3.20)
project(loom_cmsis_nn_workloads C CXX)

if(NOT DEFINED LOOM_CMSIS_NN_SOURCE OR NOT DEFINED LOOM_UNITY_SOURCE)
  message(FATAL_ERROR "CMSIS-NN and Unity source roots are required")
endif()

set(CMSISNN_BUILD_PYBIND OFF CACHE BOOL "" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/workloads")
add_subdirectory("${{LOOM_CMSIS_NN_SOURCE}}" cmsis-nn)
add_subdirectory("${{LOOM_UNITY_SOURCE}}" unity)

function(add_cmsis_nn_unit_test_executable)
  foreach(target ${{ARGV}})
    add_executable(${{target}})
  endforeach()
endfunction()

{add_cases}

foreach(target IN ITEMS {target_items})
  target_include_directories(${{target}} PRIVATE
    "${{CMAKE_CURRENT_SOURCE_DIR}}/TestCases/Utils")
  target_link_libraries(${{target}} PRIVATE unity cmsis-nn)
endforeach()
"""


def materialize_cmsis_nn_harness(
    workloads: Sequence[corpus_inventory.ProgramWorkload],
    external_root: Path,
    destination: Path,
) -> CmsisNnHarness:
    if not workloads:
        raise WorkloadProviderError("CMSIS-NN harness selection is empty")
    if destination.exists():
        raise WorkloadProviderError(
            f"CMSIS-NN harness destination already exists: {destination}"
        )

    unity_source = external_root / "unity"
    for required in (unity_source / "src" / "unity.c", unity_source / "src" / "unity.h"):
        if not required.is_file():
            raise WorkloadProviderError(f"pinned Unity source is unavailable: {required}")

    source_dir = destination / "source"
    test_cases = source_dir / "TestCases"
    test_cases.mkdir(parents=True)
    cmsis_shared = external_root / "cmsis-nn" / "Tests" / "UnitTest" / "TestCases"
    for name in ("Common", "TestData", "Utils"):
        shared = cmsis_shared / name
        if not shared.is_dir():
            raise WorkloadProviderError(f"CMSIS-NN shared test input is unavailable: {shared}")
        (test_cases / name).symlink_to(shared, target_is_directory=True)

    targets: list[str] = []
    case_directories: list[str] = []
    for workload in workloads:
        if workload.suite != "cmsis-nn" or workload.producer.kind != "cmsis-nn-unit-test":
            raise WorkloadProviderError(
                f"workload is not owned by the CMSIS-NN provider: {workload.identity}"
            )
        target = workload.producer.target
        if target != workload.executable or not re.fullmatch(r"[A-Za-z0-9_]+", target):
            raise WorkloadProviderError(
                f"CMSIS-NN workload has an invalid target: {workload.identity}"
            )
        case_source = _case_source(workload, external_root)
        case_name = case_source.name
        case_destination = test_cases / case_name
        if case_name in case_directories or target in targets:
            raise WorkloadProviderError(
                f"CMSIS-NN harness repeats a case or target: {workload.identity}"
            )
        try:
            shutil.copytree(case_source, case_destination)
        except OSError as exc:
            raise WorkloadProviderError(
                f"cannot stage CMSIS-NN case {case_source}: {exc}"
            ) from exc

        wrappers = tuple((case_destination / "Unity").glob("unity_test_arm*.c"))
        if len(wrappers) != 1:
            raise WorkloadProviderError(
                f"CMSIS-NN case must own one Unity wrapper: {case_source}"
            )
        wrapper = wrappers[0]
        runner = wrapper.parent / "TestRunner" / f"{wrapper.stem}_runner.c"
        runner.parent.mkdir()
        runner.write_text(_render_unity_runner(_unity_tests(wrapper)), encoding="utf-8")
        targets.append(target)
        case_directories.append(case_name)

    (source_dir / "CMakeLists.txt").write_text(
        _render_harness_cmake(case_directories, targets), encoding="utf-8"
    )
    return CmsisNnHarness(source_dir, unity_source, tuple(targets))


def cmake_configure_command(
    harness: CmsisNnHarness,
    cmsis_nn_source: Path,
    build_dir: Path,
    toolchain: CmakeToolchain,
) -> list[str]:
    command = [
        "cmake",
        "-S",
        str(harness.source_dir),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={toolchain.c_compiler}",
        f"-DCMAKE_CXX_COMPILER={toolchain.cxx_compiler}",
        f"-DCMAKE_AR={toolchain.archiver}",
        f"-DCMAKE_RANLIB={toolchain.ranlib}",
        f"-DLOOM_CMSIS_NN_SOURCE={cmsis_nn_source}",
        f"-DLOOM_UNITY_SOURCE={harness.unity_source}",
        f"-DCMAKE_C_FLAGS={' '.join(toolchain.compiler_flags)}",
        f"-DCMAKE_CXX_FLAGS={' '.join(toolchain.compiler_flags)}",
        f"-DCMAKE_EXE_LINKER_FLAGS={' '.join(toolchain.linker_flags)}",
    ]
    if toolchain.system_name is not None:
        command.append(f"-DCMAKE_SYSTEM_NAME={toolchain.system_name}")
    return command


def cmake_build_command(
    build_dir: Path, targets: Sequence[str], jobs: int
) -> list[str]:
    return [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        *targets,
        "-j",
        str(jobs),
    ]
