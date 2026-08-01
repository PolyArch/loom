#!/usr/bin/env python3
"""CMake projection for materialized CMSIS-DSP corpus workloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CmakeTarget:
    target: str
    generated: Path
    shared: Path
    test_class: str | None = None
    source_group: str | None = None
    direct_source: Path | None = None
    direct_support_sources: tuple[Path, ...] = ()
    direct_include_directories: tuple[Path, ...] = ()
    compiler_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.source_group is None) == (self.direct_source is None):
            raise ValueError(
                "CMSIS-DSP target must select one descriptor or direct source"
            )
        if self.source_group is not None and self.test_class is None:
            raise ValueError("CMSIS-DSP descriptor target has no test class")


def _quote(path: Path) -> str:
    return path.as_posix().replace('"', '\\"')


def render_harness(
    targets: Sequence[CmakeTarget],
    support_sources: Sequence[Path],
    operator_compile_options: dict[Path, tuple[str, ...]],
    *,
    enable_float16: bool,
    enable_wrapper: bool,
) -> str:
    suite_libraries: dict[Path, tuple[str, str, str]] = {}
    for item in targets:
        if item.source_group is None:
            continue
        assert item.test_class is not None
        suite_libraries.setdefault(
            item.shared,
            (
                f"loom_dsp_{item.shared.name}",
                item.test_class,
                item.source_group,
            ),
        )

    suite_blocks = []
    for shared, (library, test_class, source_group) in suite_libraries.items():
        shared_include = _quote(shared / "GeneratedInclude")
        suite_blocks.append(
            f'''add_library({library} OBJECT
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/Source/{source_group}/{test_class}.cpp")
target_include_directories({library} PRIVATE
  "{shared_include}"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/Include/{source_group}"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkInclude")
target_compile_definitions({library} PRIVATE EMBEDDED NOTIMING)
target_compile_options({library} PRIVATE -fno-inline-functions)
target_link_libraries({library} PRIVATE CMSISDSP)
'''
        )

    target_blocks = []
    for item in targets:
        if item.direct_source is not None:
            compile_options = " ".join(("-fno-inline-functions", *item.compiler_flags))
            direct_sources = "".join(
                f'\n  "{_quote(source)}"' for source in item.direct_support_sources
            )
            direct_includes = "".join(
                f'\n  "{_quote(directory)}"'
                for directory in item.direct_include_directories
            )
            target_blocks.append(
                f'''add_executable({item.target}
  "{_quote(item.direct_source)}"{direct_sources})
target_include_directories({item.target} PRIVATE
  "${{LOOM_CMSIS_DSP_SOURCE}}/Include"{direct_includes})
target_compile_definitions({item.target} PRIVATE EMBEDDED NOTIMING)
target_compile_options({item.target} PRIVATE {compile_options})
target_link_libraries({item.target} PRIVATE CMSISDSP)
'''
            )
            continue
        assert item.source_group is not None
        library = suite_libraries[item.shared][0]
        generated_source = _quote(item.generated / "GeneratedSource")
        generated_include = _quote(item.generated / "GeneratedInclude")
        shared_include = _quote(item.shared / "GeneratedInclude")
        target_blocks.append(
            f'''add_executable({item.target}
  "${{CMAKE_CURRENT_SOURCE_DIR}}/OperatorMain.cpp"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/patterndata.c"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/testmain.cpp"
  "{generated_source}/TestDesc.cpp"
  $<TARGET_OBJECTS:{library}>)
target_include_directories({item.target} PRIVATE
  "{generated_include}"
  "{shared_include}"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkInclude"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/Include/{item.source_group}")
target_compile_definitions({item.target} PRIVATE EMBEDDED NOTIMING)
target_link_libraries({item.target} PRIVATE
  loom_cmsis_dsp_framework loom_cmsis_dsp_test_support CMSISDSP)
'''
        )

    framework_block = ""
    if suite_libraries:
        support_items = "\n".join(f'  "{_quote(source)}"' for source in support_sources)
        framework_block = f"""add_library(loom_cmsis_dsp_framework STATIC
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkSource/ArrayMemory.cpp"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkSource/Calibrate.cpp"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkSource/Error.cpp"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkSource/FPGA.cpp"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkSource/Generators.cpp"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkSource/IORunner.cpp"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkSource/Pattern.cpp"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkSource/PatternMgr.cpp"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkSource/Test.cpp"
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkSource/Timing.cpp")
target_include_directories(loom_cmsis_dsp_framework PUBLIC
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/FrameworkInclude")
target_compile_definitions(loom_cmsis_dsp_framework PRIVATE EMBEDDED NOTIMING)
target_link_libraries(loom_cmsis_dsp_framework PUBLIC CMSISDSP)

add_library(loom_cmsis_dsp_test_support STATIC
{support_items})
target_include_directories(loom_cmsis_dsp_test_support PRIVATE
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing/Include/Tests")
target_compile_definitions(loom_cmsis_dsp_test_support PRIVATE EMBEDDED NOTIMING)
target_link_libraries(loom_cmsis_dsp_test_support PRIVATE CMSISDSP)
"""
    operator_option_blocks = []
    for source, options in sorted(operator_compile_options.items()):
        for option in options:
            escaped_option = option.replace('"', '\\"')
            operator_option_blocks.append(
                "set_property(SOURCE "
                f'"${{LOOM_CMSIS_DSP_SOURCE}}/{_quote(source)}" '
                "TARGET_DIRECTORY CMSISDSP APPEND PROPERTY COMPILE_OPTIONS "
                f'"{escaped_option}")\n'
            )

    return f"""cmake_minimum_required(VERSION 3.20)
project(loom_cmsis_dsp_workloads C CXX)

if(NOT DEFINED LOOM_CMSIS_DSP_SOURCE)
  message(FATAL_ERROR "CMSIS-DSP source root is required")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/workloads")
set(CMSISDSP_INSTALL OFF CACHE BOOL "" FORCE)
set(DISABLEFLOAT16 {"OFF" if enable_float16 else "ON"} CACHE BOOL "" FORCE)
set(FASTBUILD OFF CACHE BOOL "" FORCE)
set(HELIUM OFF CACHE BOOL "" FORCE)
set(HOST ON CACHE BOOL "" FORCE)
set(MVEF OFF CACHE BOOL "" FORCE)
set(MVEI OFF CACHE BOOL "" FORCE)
set(NEON OFF CACHE BOOL "" FORCE)
set(NEONEXPERIMENTAL OFF CACHE BOOL "" FORCE)
set(WRAPPER {"ON" if enable_wrapper else "OFF"} CACHE BOOL "" FORCE)
add_subdirectory("${{LOOM_CMSIS_DSP_SOURCE}}/Source" cmsis-dsp)
target_compile_definitions(CMSISDSP PRIVATE ARM_DSP_CUSTOM_CONFIG)
target_compile_definitions(CMSISDSP PUBLIC ARM_DSP_TESTING)
target_include_directories(CMSISDSP PRIVATE
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing")
{"".join(operator_option_blocks)}

{framework_block}
{"".join(suite_blocks)}
{"".join(target_blocks)}
"""
