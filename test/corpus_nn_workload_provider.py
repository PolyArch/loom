#!/usr/bin/env python3
"""CMSIS-NN linked workload materialization."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import corpus_inventory
import corpus_nn_protocol
from corpus_workload_errors import WorkloadProviderError


@dataclass(frozen=True)
class CmsisNnHarness:
    source_dir: Path
    unity_source: Path | None
    targets: tuple[str, ...]
    protocol_symbol_sets: tuple[tuple[str, ...], ...]
    protocol_source_owners: tuple[tuple[Path, Path], ...]
    expected_entry_results: tuple[int, ...]

    def executable(self, build_dir: Path, target: str) -> Path:
        if target not in self.targets:
            raise WorkloadProviderError(f"unknown CMSIS-NN harness target: {target}")
        return build_dir / "workloads" / target

    def protocol_symbols(self, target: str) -> tuple[str, ...]:
        return self.protocol_symbol_sets[self.targets.index(self._target(target))]

    def protocol_source_owner(self, target: str) -> tuple[Path, Path]:
        return self.protocol_source_owners[self.targets.index(self._target(target))]

    def expected_entry_result(self, target: str) -> int:
        return self.expected_entry_results[self.targets.index(self._target(target))]

    def _target(self, target: str) -> str:
        if target not in self.targets:
            raise WorkloadProviderError(f"unknown CMSIS-NN harness target: {target}")
        return target


@dataclass(frozen=True)
class _CmakeTarget:
    target: str
    case_directory: str | None = None
    direct_source: Path | None = None
    operator_sources: tuple[Path, ...] = ()
    compiler_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.case_directory is None) == (self.direct_source is None):
            raise ValueError(
                "CMSIS-NN target must select one Unity case or direct source"
            )


def _render_unity_runner(test_function: str) -> str:
    return (
        '#include "unity.h"\n\n'
        "void setUp(void);\n"
        "void tearDown(void);\n"
        f"void {test_function}(void);\n\n"
        "int main(void) {\n"
        "  UNITY_BEGIN();\n"
        f"  RUN_TEST({test_function});\n"
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
    return (
        external_root
        / "cmsis-nn"
        / "Tests"
        / "UnitTest"
        / "TestCases"
        / relative.parent
    )


def _cmake_quote(path: Path) -> str:
    return path.as_posix().replace('"', '\\"')


def _render_harness_cmake(targets: Sequence[_CmakeTarget]) -> str:
    unity_targets = tuple(item for item in targets if item.case_directory is not None)
    add_cases = "\n".join(
        f'add_subdirectory("TestCases/{item.case_directory}" '
        f'"cases/{item.case_directory}")'
        for item in unity_targets
    )
    unity_target_items = " ".join(item.target for item in unity_targets)
    unity_compile_options = "\n".join(
        f"target_compile_options({item.target} PRIVATE "
        f"{' '.join(('-fno-inline-functions', *item.compiler_flags))})"
        for item in unity_targets
    )
    unity_setup = ""
    if unity_targets:
        unity_setup = f"""add_subdirectory("${{LOOM_CMSIS_NN_SOURCE}}" cmsis-nn)
add_subdirectory("${{LOOM_UNITY_SOURCE}}" unity)

function(add_cmsis_nn_unit_test_executable)
  foreach(target ${{ARGV}})
    add_executable(${{target}})
  endforeach()
endfunction()

{add_cases}

foreach(target IN ITEMS {unity_target_items})
  target_include_directories(${{target}} PRIVATE
    "${{CMAKE_CURRENT_SOURCE_DIR}}/TestCases/Utils")
  target_link_libraries(${{target}} PRIVATE unity cmsis-nn)
endforeach()
{unity_compile_options}
"""

    direct_blocks: list[str] = []
    for item in targets:
        if item.direct_source is None:
            continue
        compile_options = " ".join(
            ("-fno-inline-functions", *item.compiler_flags)
        )
        operator_sources = "\n".join(
            f'  "${{LOOM_CMSIS_NN_SOURCE}}/{_cmake_quote(source)}"'
            for source in item.operator_sources
        )
        direct_blocks.append(
            f'''add_executable({item.target}
  "{_cmake_quote(item.direct_source)}"
{operator_sources})
target_include_directories({item.target} PRIVATE
  "${{LOOM_CMSIS_NN_SOURCE}}/Include"
  "${{CMAKE_CURRENT_SOURCE_DIR}}")
target_compile_options({item.target} PRIVATE {compile_options})
'''
        )

    unity_requirement = " OR NOT DEFINED LOOM_UNITY_SOURCE" if unity_targets else ""
    return f"""cmake_minimum_required(VERSION 3.20)
project(loom_cmsis_nn_workloads C CXX)

if(NOT DEFINED LOOM_CMSIS_NN_SOURCE{unity_requirement})
  message(FATAL_ERROR "CMSIS-NN workload source roots are required")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/workloads")
{unity_setup}
{"".join(direct_blocks)}
"""


def _rename_staged_cmake_target(
    cmake_path: Path, owner_target: str, workload_target: str
) -> None:
    try:
        text = cmake_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkloadProviderError(f"cannot read {cmake_path}: {exc}") from exc
    try:
        declared_target = corpus_inventory.load_cmsis_nn_case_target(cmake_path)
    except corpus_inventory.InventoryError as exc:
        raise WorkloadProviderError(str(exc)) from exc
    if declared_target != owner_target:
        raise WorkloadProviderError(
            f"CMSIS-NN case declares {declared_target}, expected {owner_target}"
        )
    token = re.compile(
        r"(?<![A-Za-z0-9_])" + re.escape(owner_target) + r"(?![A-Za-z0-9_])"
    )
    rewritten, replacements = token.subn(workload_target, text)
    if replacements < 2:
        raise WorkloadProviderError(
            f"CMSIS-NN case does not completely define target {owner_target}"
        )
    try:
        cmake_path.write_text(rewritten, encoding="utf-8")
    except OSError as exc:
        raise WorkloadProviderError(f"cannot write {cmake_path}: {exc}") from exc


def _operator_sources(
    workload: corpus_inventory.ProgramWorkload,
    external_root: Path,
    *,
    allow_empty: bool = False,
) -> tuple[Path, ...]:
    owner = Path("externals/cmsis-nn")
    sources: list[Path] = []
    for raw_source in workload.sources:
        source = Path(raw_source)
        try:
            relative = source.relative_to(owner)
        except ValueError as exc:
            raise WorkloadProviderError(
                f"CMSIS-NN operator source escapes its owner: {source}"
            ) from exc
        if not (external_root / "cmsis-nn" / relative).is_file():
            raise WorkloadProviderError(
                f"CMSIS-NN operator source is unavailable: {source}"
            )
        sources.append(relative)
    if not sources and not allow_empty:
        raise WorkloadProviderError(
            f"CMSIS-NN direct protocol has no implementation sources: "
            f"{workload.identity}"
        )
    return tuple(sources)


def materialize_cmsis_nn_harness(
    workloads: Sequence[corpus_inventory.ProgramWorkload],
    external_root: Path,
    destination: Path,
    protocol_symbol: str,
) -> CmsisNnHarness:
    if not workloads:
        raise WorkloadProviderError("CMSIS-NN harness selection is empty")
    if destination.exists():
        raise WorkloadProviderError(
            f"CMSIS-NN harness destination already exists: {destination}"
        )

    uses_unity = any(
        not isinstance(
            workload.producer,
            corpus_inventory.CmsisNnGeneratedWorkloadProducer,
        )
        for workload in workloads
    )
    unity_source = external_root / "unity" if uses_unity else None
    if unity_source is not None:
        for required in (
            unity_source / "src" / "unity.c",
            unity_source / "src" / "unity.h",
        ):
            if not required.is_file():
                raise WorkloadProviderError(
                    f"pinned Unity source is unavailable: {required}"
                )

    source_dir = destination / "source"
    test_cases = source_dir / "TestCases"
    test_cases.mkdir(parents=True)
    cmsis_shared = external_root / "cmsis-nn" / "Tests" / "UnitTest" / "TestCases"
    for name in ("Common", "TestData", "Utils"):
        shared = cmsis_shared / name
        if not shared.is_dir():
            raise WorkloadProviderError(
                f"CMSIS-NN shared test input is unavailable: {shared}"
            )
        (test_cases / name).symlink_to(shared, target_is_directory=True)

    targets: list[str] = []
    protocol_symbol_sets: list[tuple[str, ...]] = []
    protocol_source_owners: list[tuple[Path, Path]] = []
    expected_entry_results: list[int] = []
    cmake_targets: list[_CmakeTarget] = []
    for workload in workloads:
        if workload.suite != "cmsis-nn" or not isinstance(
            workload.producer,
            (
                corpus_inventory.CmsisNnWorkloadProducer,
                corpus_inventory.CmsisNnGeneratedWorkloadProducer,
            ),
        ):
            raise WorkloadProviderError(
                f"workload is not owned by the CMSIS-NN provider: "
                f"{workload.identity}"
            )
        target = workload.executable
        try:
            expected_target = corpus_inventory.operator_workload_target(
                workload.operator_id
            )
        except corpus_inventory.InventoryError as exc:
            raise WorkloadProviderError(str(exc)) from exc
        if target != expected_target:
            raise WorkloadProviderError(
                f"CMSIS-NN workload has an invalid target: {workload.identity}"
            )
        if target in targets:
            raise WorkloadProviderError(
                f"CMSIS-NN harness repeats a target: {workload.identity}"
            )

        if isinstance(
            workload.producer,
            corpus_inventory.CmsisNnGeneratedWorkloadProducer,
        ):
            projection = corpus_nn_protocol.render_generated_cmsis_nn_protocol(
                workload, external_root, protocol_symbol
            )
            generated = source_dir / "generated" / "targets" / target
            generated.mkdir(parents=True)
            direct_source = generated / "OperatorProtocol.c"
            direct_source.write_text(projection.source, encoding="utf-8")
            targets.append(target)
            protocol_symbol_sets.append((projection.protocol_symbol,))
            protocol_source_owners.append(
                (
                    projection.compiled_owner or direct_source,
                    projection.authoritative_owner,
                )
            )
            expected_entry_results.append(0)
            cmake_targets.append(
                _CmakeTarget(
                    target,
                    direct_source=direct_source,
                    operator_sources=_operator_sources(
                        workload,
                        external_root,
                        allow_empty=projection.compiled_owner is None,
                    ),
                    compiler_flags=workload.compiler_flags,
                )
            )
            continue

        owner_target = workload.producer.target
        case_source = _case_source(workload, external_root)
        case_destination = test_cases / target
        try:
            shutil.copytree(case_source, case_destination)
        except OSError as exc:
            raise WorkloadProviderError(
                f"cannot stage CMSIS-NN case {case_source}: {exc}"
            ) from exc

        _rename_staged_cmake_target(
            case_destination / "CMakeLists.txt", owner_target, target
        )
        wrappers = tuple((case_destination / "Unity").glob("unity_test_arm*.c"))
        if len(wrappers) != 1:
            raise WorkloadProviderError(
                f"CMSIS-NN case must own one Unity wrapper: {case_source}"
            )
        wrapper = wrappers[0]
        original_wrapper = case_source / "Unity" / wrapper.name
        if not original_wrapper.is_file():
            raise WorkloadProviderError(
                f"CMSIS-NN protocol owner is unavailable: {original_wrapper}"
            )
        try:
            test_functions = corpus_inventory.load_cmsis_nn_unity_test_functions(
                case_destination
            )
        except corpus_inventory.InventoryError as exc:
            raise WorkloadProviderError(str(exc)) from exc
        if workload.producer.test_function not in test_functions:
            raise WorkloadProviderError(
                f"CMSIS-NN workload selects an unknown test function: "
                f"{workload.identity}"
            )
        runner = wrapper.parent / "TestRunner" / f"{wrapper.stem}_runner.c"
        runner.parent.mkdir()
        runner.write_text(
            _render_unity_runner(workload.producer.test_function), encoding="utf-8"
        )
        targets.append(target)
        protocol_symbol_sets.append((workload.producer.test_function,))
        protocol_source_owners.append((wrapper, original_wrapper))
        expected_entry_results.append(0)
        cmake_targets.append(
            _CmakeTarget(
                target,
                case_directory=target,
                compiler_flags=workload.compiler_flags,
            )
        )

    (source_dir / "CMakeLists.txt").write_text(
        _render_harness_cmake(cmake_targets), encoding="utf-8"
    )
    return CmsisNnHarness(
        source_dir,
        unity_source,
        tuple(targets),
        tuple(protocol_symbol_sets),
        tuple(protocol_source_owners),
        tuple(expected_entry_results),
    )


def supports_cmsis_nn_harness(
    workload: corpus_inventory.ProgramWorkload,
) -> bool:
    if isinstance(workload.producer, corpus_inventory.CmsisNnWorkloadProducer):
        return True
    return corpus_nn_protocol.supports_generated_cmsis_nn_protocol(workload)
