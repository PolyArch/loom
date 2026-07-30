#!/usr/bin/env python3
"""Ephemeral builders for corpus-owned linked program workloads."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import math
import re
import shutil
import struct
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import corpus_inventory


class WorkloadProviderError(ValueError):
    """Raised when an owned workload cannot be materialized exactly."""


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
class CmsisDspHarness:
    source_dir: Path
    targets: tuple[str, ...]
    shared_directories: tuple[Path, ...]
    protocol_methods: tuple[tuple[str, str], ...]
    protocol_symbol_sets: tuple[tuple[str, ...], ...]
    protocol_source_owners: tuple[tuple[Path, Path], ...]
    expected_entry_results: tuple[int | None, ...] = ()

    def generated_directory(self, target: str) -> Path:
        return self.source_dir / "generated" / "targets" / self._target(target)

    def shared_directory(self, target: str) -> Path:
        return self.shared_directories[self.targets.index(self._target(target))]

    def executable(self, build_dir: Path, target: str) -> Path:
        return build_dir / "workloads" / self._target(target)

    def protocol_method(self, target: str) -> tuple[str, str]:
        return self.protocol_methods[self.targets.index(self._target(target))]

    def protocol_symbols(self, target: str) -> tuple[str, ...]:
        return self.protocol_symbol_sets[self.targets.index(self._target(target))]

    def protocol_source_owner(self, target: str) -> tuple[Path, Path]:
        return self.protocol_source_owners[self.targets.index(self._target(target))]

    def expected_entry_result(self, target: str) -> int | None:
        ordinal = self.targets.index(self._target(target))
        if not self.expected_entry_results:
            return None
        return self.expected_entry_results[ordinal]

    def _target(self, target: str) -> str:
        if target not in self.targets:
            raise WorkloadProviderError(f"unknown CMSIS-DSP harness target: {target}")
        return target


@dataclass(frozen=True)
class ProducedWorkload:
    target_build_dir: Path
    target_executable: Path
    protocol_symbols: tuple[str, ...]
    protocol_source_owners: tuple[tuple[Path, Path], ...]
    expected_entry_result: int | None = None


@dataclass(frozen=True)
class CmakeToolchain:
    c_compiler: str
    cxx_compiler: str
    archiver: Path
    ranlib: Path
    compiler_flags: tuple[str, ...]
    linker_flags: tuple[str, ...]
    system_name: str | None = None


@dataclass(frozen=True)
class _CmsisDspCmakeTarget:
    target: str
    generated: Path
    shared: Path
    test_class: str
    source_group: str | None = None
    direct_source: Path | None = None

    def __post_init__(self) -> None:
        if (self.source_group is None) == (self.direct_source is None):
            raise ValueError(
                "CMSIS-DSP target must select one descriptor or direct source"
            )


@dataclass(frozen=True)
class _CmsisNnCmakeTarget:
    target: str
    case_directory: str | None = None
    direct_source: Path | None = None
    operator_sources: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if (self.case_directory is None) == (self.direct_source is None):
            raise ValueError(
                "CMSIS-NN target must select one Unity case or direct source"
            )
        if self.direct_source is not None and not self.operator_sources:
            raise ValueError("direct CMSIS-NN target has no operator sources")


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


def _render_harness_cmake(targets: Sequence[_CmsisNnCmakeTarget]) -> str:
    unity_targets = tuple(item for item in targets if item.case_directory is not None)
    add_cases = "\n".join(
        f'add_subdirectory("TestCases/{item.case_directory}" '
        f'"cases/{item.case_directory}")'
        for item in unity_targets
    )
    unity_target_items = " ".join(item.target for item in unity_targets)
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
  target_compile_options(${{target}} PRIVATE -fno-inline-functions)
  target_link_libraries(${{target}} PRIVATE unity cmsis-nn)
endforeach()
"""

    direct_blocks: list[str] = []
    for item in targets:
        if item.direct_source is None:
            continue
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
target_compile_options({item.target} PRIVATE -fno-inline-functions)
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


def _cmsis_nn_operator_sources(
    workload: corpus_inventory.ProgramWorkload, external_root: Path
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
    if not sources:
        raise WorkloadProviderError(
            f"CMSIS-NN direct protocol has no implementation sources: {workload.identity}"
        )
    return tuple(sources)


def _render_cmsis_nn_convolution_protocol() -> str:
    return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "TestCases/TestData/kernel1x1/test_data.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

static int oracle_matches(const int8_t *output)
{{
    for (size_t index = 0; index < KERNEL1X1_DST_SIZE; ++index)
    {{
        if (output[index] != kernel1x1_output_ref[index])
        {{
            return 0;
        }}
    }}
    return 1;
}}

LOOM_NOINLINE arm_cmsis_nn_status {_CORPUS_OPERATOR_PROTOCOL_SYMBOL}(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *biases,
    const int32_t *multipliers,
    const int32_t *shifts,
    void *scratch,
    size_t scratch_capacity,
    int8_t *output)
{{
    cmsis_nn_context context = {{0}};
    cmsis_nn_conv_params convolution = {{0}};
    cmsis_nn_per_channel_quant_params quantization = {{0}};
    cmsis_nn_dims input_dims = {{0}};
    cmsis_nn_dims filter_dims = {{0}};
    cmsis_nn_dims bias_dims = {{0}};
    cmsis_nn_dims output_dims = {{0}};

    input_dims.n = KERNEL1X1_INPUT_BATCHES;
    input_dims.h = KERNEL1X1_INPUT_H;
    input_dims.w = KERNEL1X1_INPUT_W;
    input_dims.c = KERNEL1X1_IN_CH;
    filter_dims.n = KERNEL1X1_OUT_CH;
    filter_dims.h = KERNEL1X1_FILTER_Y;
    filter_dims.w = KERNEL1X1_FILTER_X;
    filter_dims.c = KERNEL1X1_IN_CH;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = KERNEL1X1_OUT_CH;
    output_dims.n = KERNEL1X1_INPUT_BATCHES;
    output_dims.h = KERNEL1X1_OUTPUT_H;
    output_dims.w = KERNEL1X1_OUTPUT_W;
    output_dims.c = KERNEL1X1_OUT_CH;

    convolution.padding.h = KERNEL1X1_PAD_Y;
    convolution.padding.w = KERNEL1X1_PAD_X;
    convolution.stride.h = KERNEL1X1_STRIDE_Y;
    convolution.stride.w = KERNEL1X1_STRIDE_X;
    convolution.dilation.h = KERNEL1X1_DILATION_Y;
    convolution.dilation.w = KERNEL1X1_DILATION_X;
    convolution.input_offset = KERNEL1X1_INPUT_OFFSET;
    convolution.output_offset = KERNEL1X1_OUTPUT_OFFSET;
    convolution.activation.min = KERNEL1X1_OUT_ACTIVATION_MIN;
    convolution.activation.max = KERNEL1X1_OUT_ACTIVATION_MAX;
    quantization.multiplier = (int32_t *)multipliers;
    quantization.shift = (int32_t *)shifts;

    const int32_t required =
        arm_convolve_1x1_s8_fast_get_buffer_size(&input_dims);
    if (required < 0 || (size_t)required > scratch_capacity)
    {{
        return ARM_CMSIS_NN_ARG_ERROR;
    }}
    context.buf = required == 0 ? NULL : scratch;
    context.size = required;
    return arm_convolve_1x1_s8_fast(&context,
                                    &convolution,
                                    &quantization,
                                    &input_dims,
                                    input,
                                    &filter_dims,
                                    weights,
                                    &bias_dims,
                                    biases,
                                    &output_dims,
                                    output);
}}

int main(void)
{{
    int8_t output[KERNEL1X1_DST_SIZE] = {{0}};
    uint8_t scratch[2 * KERNEL1X1_IN_CH * sizeof(int16_t)] = {{0}};
    const arm_cmsis_nn_status status = {_CORPUS_OPERATOR_PROTOCOL_SYMBOL}(
        kernel1x1_input,
        kernel1x1_weights,
        kernel1x1_biases,
        kernel1x1_output_mult,
        kernel1x1_output_shift,
        scratch,
        sizeof(scratch),
        output);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {{
        return 1;
    }}
    return oracle_matches(output) ? 0 : 1;
}}
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

    uses_unity = any(
        workload.identity != _CMSIS_NN_DIRECT_CONVOLUTION_ID for workload in workloads
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
    cmake_targets: list[_CmsisNnCmakeTarget] = []
    for workload in workloads:
        if workload.suite != "cmsis-nn" or not isinstance(
            workload.producer, corpus_inventory.CmsisNnWorkloadProducer
        ):
            raise WorkloadProviderError(
                f"workload is not owned by the CMSIS-NN provider: {workload.identity}"
            )
        owner_target = workload.producer.target
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

        if workload.identity == _CMSIS_NN_DIRECT_CONVOLUTION_ID:
            protocol = tuple(
                (call.symbol, call.signature) for call in workload.protocol
            )
            if protocol != (
                (
                    "arm_convolve_1x1_s8_fast_get_buffer_size",
                    "int32_t (const cmsis_nn_dims *)",
                ),
                (
                    "arm_convolve_1x1_s8_fast",
                    "arm_cmsis_nn_status (const cmsis_nn_context *, const "
                    "cmsis_nn_conv_params *, const "
                    "cmsis_nn_per_channel_quant_params *, const cmsis_nn_dims "
                    "*, const int8_t *, const cmsis_nn_dims *, const int8_t *, "
                    "const cmsis_nn_dims *, const int32_t *, const cmsis_nn_dims "
                    "*, int8_t *)",
                ),
            ):
                raise WorkloadProviderError(
                    "CMSIS-NN convolution protocol no longer matches its manifest"
                )
            generated = source_dir / "generated" / "targets" / target
            generated.mkdir(parents=True)
            direct_source = generated / "OperatorProtocol.c"
            direct_source.write_text(
                _render_cmsis_nn_convolution_protocol(), encoding="utf-8"
            )
            authoritative_owner = (
                external_root / "cmsis-nn" / "Include" / "arm_nnfunctions.h"
            )
            if not authoritative_owner.is_file():
                raise WorkloadProviderError(
                    "CMSIS-NN convolution protocol declaration is unavailable"
                )
            targets.append(target)
            protocol_symbol_sets.append((_CORPUS_OPERATOR_PROTOCOL_SYMBOL,))
            protocol_source_owners.append((direct_source, authoritative_owner))
            expected_entry_results.append(0)
            cmake_targets.append(
                _CmsisNnCmakeTarget(
                    target,
                    direct_source=direct_source,
                    operator_sources=_cmsis_nn_operator_sources(
                        workload, external_root
                    ),
                )
            )
            continue

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
                f"CMSIS-NN workload selects an unknown test function: {workload.identity}"
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
        cmake_targets.append(_CmsisNnCmakeTarget(target, case_directory=target))

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


def _load_cmsis_dsp_codegen(testing_root: Path) -> tuple[object, object, object]:
    path = str(testing_root)
    sys.path.insert(0, path)
    try:
        import pyparsing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyparsing.PyparsingDeprecationWarning)
            parser = importlib.import_module("TestScripts.NewParser")
            codegen = importlib.import_module("TestScripts.CodeGen")
            tree = importlib.import_module("TestScripts.Parser")
    except ImportError as exc:
        raise WorkloadProviderError(
            f"cannot import pinned CMSIS-DSP test generator: {exc}"
        ) from exc
    finally:
        sys.path.remove(path)
    return parser, codegen, tree


def _cmsis_dsp_suite_chain(
    root: object, suite_kind: int, test_class: str
) -> list[object]:
    matches: list[list[object]] = []

    def visit(node: object, path: list[object]) -> None:
        current = [*path, node]
        data = node.data
        if node.kind == suite_kind and data.get("class") == test_class:
            matches.append(current)
        for child in node.children:
            visit(child, current)

    visit(root, [])
    if len(matches) != 1:
        raise WorkloadProviderError(
            f"CMSIS-DSP descriptor resolves {len(matches)} suites for {test_class}"
        )
    return matches[0]


def _select_cmsis_dsp_test_tree(
    root: object,
    suite_kind: int,
    test_kind: int,
    test_class: str,
    test_method: str,
    vector_ordinal: int,
) -> object:
    selected = copy.deepcopy(root)
    chain = _cmsis_dsp_suite_chain(selected, suite_kind, test_class)
    suite = chain[-1]
    matching = [
        child
        for child in suite.children
        if child.kind == test_kind and child.data.get("class") == test_method
    ]
    if vector_ordinal >= len(matching):
        raise WorkloadProviderError(
            "CMSIS-DSP descriptor has no selected vector for "
            f"{test_class}::{test_method}[{vector_ordinal}]"
        )
    for parent, child in zip(chain[:-1], chain[1:], strict=True):
        parent._children = [child]
    suite._children = [matching[vector_ordinal]]
    return selected


def _select_cmsis_dsp_suite_tree(
    root: object, suite_kind: int, test_class: str
) -> object:
    selected = copy.deepcopy(root)
    chain = _cmsis_dsp_suite_chain(selected, suite_kind, test_class)
    for parent, child in zip(chain[:-1], chain[1:], strict=True):
        parent._children = [child]
    return selected


def _cmsis_dsp_shared_name(descriptor: Path, test_class: str) -> str:
    preimage = f"{descriptor.as_posix()}\0{test_class}".encode()
    return f"suite_{hashlib.sha256(preimage).hexdigest()[:16]}"


def _generate_cmsis_dsp_tree(
    codegen_module: object,
    testing_root: Path,
    generated: Path,
    tree: object,
) -> None:
    (generated / "GeneratedSource").mkdir(parents=True)
    (generated / "GeneratedInclude").mkdir()
    with contextlib.chdir(generated):
        codegen_module.CodeGen(
            str(testing_root / "Patterns"),
            str(testing_root / "Parameters"),
            True,
        ).genCodeForTree(str(generated), tree, False)


def _cmake_quote(path: Path) -> str:
    return path.as_posix().replace('"', '\\"')


_CORPUS_OPERATOR_PROTOCOL_SYMBOL = "loom_corpus_operator_protocol"
_CMSIS_NN_DIRECT_CONVOLUTION_ID = "cmsis-nn:arm-convolve-1x1-s8-fast:e4fc696adf47aaf4"
_CMSIS_DSP_C_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_CMSIS_DSP_STATELESS_CONTROLLER_SIGNATURES = {
    "void(float,float,ptr,ptr)",
    "void(float,float,ptr,ptr,float,float)",
    "void(float,ptr,ptr)",
    "void(i32,i32,ptr,ptr)",
    "void(i32,i32,ptr,ptr,i32,i32)",
    "void(i32,ptr,ptr)",
}


def _cmsis_dsp_direct_protocol_family(
    workload: corpus_inventory.ProgramWorkload,
) -> str | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer):
        return None
    if (
        producer.selector_kind == "official"
        and producer.test_class == "BasicTestsF32"
        and producer.test_method == "test_abs_f32"
        and tuple((call.symbol, call.signature) for call in workload.protocol)
        == (("arm_abs_f32", "void(ptr,ptr,i32)"),)
    ):
        return "stateless-abs-f32"
    if (
        producer.selector_kind == "official"
        and producer.test_class == "FIRF32"
        and producer.test_method == "test_fir_f32"
        and tuple((call.symbol, call.signature) for call in workload.protocol)
        == (
            ("arm_fir_init_f32", "void(ptr,i16,ptr,ptr,i32)"),
            ("arm_fir_f32", "void(ptr,ptr,ptr,i32)"),
        )
    ):
        return "stateful-fir-f32"
    if producer.selector_kind != "benchmark-only":
        return None
    if producer.test_class not in {"ControllerF32", "ControllerQ31"}:
        return None
    if len(workload.protocol) != 1:
        return None
    if workload.protocol[0].signature not in _CMSIS_DSP_STATELESS_CONTROLLER_SIGNATURES:
        return None
    return "stateless-controller"


def supports_cmsis_dsp_harness(
    workload: corpus_inventory.ProgramWorkload,
) -> bool:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer):
        return False
    return producer.selector_kind == "official" or (
        _cmsis_dsp_direct_protocol_family(workload) is not None
    )


def _cmsis_dsp_first_parameter(suite: object) -> int:
    parameter_id = suite.data.get("PARAMID")
    matching = [values for _, name, values in suite.parameters if name == parameter_id]
    if len(matching) != 1 or len(matching[0]) != 1:
        raise WorkloadProviderError(
            "CMSIS-DSP direct protocol requires one parameter row"
        )
    parameter_names = tuple(suite.params.full)
    if len(parameter_names) != 1:
        raise WorkloadProviderError(
            "CMSIS-DSP direct protocol requires one parameter dimension"
        )
    values = matching[0][0].get("INTS")
    if not isinstance(values, list) or not values:
        raise WorkloadProviderError(
            "CMSIS-DSP direct protocol has no integer workload vector"
        )
    first = values[0]
    if not isinstance(first, int) or isinstance(first, bool) or first <= 0:
        raise WorkloadProviderError(
            "CMSIS-DSP direct protocol has an invalid workload extent"
        )
    return first


def _cmsis_dsp_literal_test_extent(
    suite: object, test_kind: int, test_class: str, vector_ordinal: int
) -> int:
    matching = [
        child
        for child in suite.children
        if child.kind == test_kind and child.data.get("class") == test_class
    ]
    if vector_ordinal >= len(matching):
        raise WorkloadProviderError(
            f"CMSIS-DSP descriptor has no vector {vector_ordinal} for {test_class}"
        )
    message = matching[vector_ordinal].data.get("message")
    if not isinstance(message, str):
        raise WorkloadProviderError("CMSIS-DSP test vector has no descriptor message")
    extent = re.search(r"\bnb=(\d+)\b", message)
    if extent is None:
        raise WorkloadProviderError(
            "CMSIS-DSP direct protocol requires a literal test extent"
        )
    value = int(extent.group(1), 10)
    if value <= 0:
        raise WorkloadProviderError("CMSIS-DSP test extent is nonpositive")
    return value


def _cmsis_dsp_pattern_bytes(path: Path) -> bytes:
    return b"".join(_cmsis_dsp_pattern_segments(path).values())


def _cmsis_dsp_pattern_segments(path: Path) -> dict[str, bytes]:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise WorkloadProviderError(
            f"cannot read generated CMSIS-DSP patterns {path}: {exc}"
        ) from exc
    match = re.search(r"const\s+char\s+patterns\[\]\s*=\s*\{(.*?)\};", text, re.S)
    if match is None:
        raise WorkloadProviderError(
            f"generated CMSIS-DSP patterns have no byte array: {path}"
        )

    segments: dict[str, bytearray] = {}
    current: bytearray | None = None
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("// "):
            name = Path(stripped[3:]).name
            if name in segments:
                raise WorkloadProviderError(
                    f"generated CMSIS-DSP patterns repeat {name}: {path}"
                )
            current = bytearray()
            segments[name] = current
            continue
        for token in stripped.split(","):
            token = token.strip()
            if not token:
                continue
            if current is None:
                raise WorkloadProviderError(
                    f"generated CMSIS-DSP pattern bytes have no owner: {path}"
                )
            try:
                value = int(token, 10)
            except ValueError as exc:
                raise WorkloadProviderError(
                    f"generated CMSIS-DSP pattern byte is not decimal: {path}"
                ) from exc
            if value < 0 or value > 255:
                raise WorkloadProviderError(
                    f"generated CMSIS-DSP pattern byte is invalid: {path}"
                )
            current.append(value)
    if not segments:
        raise WorkloadProviderError(
            f"generated CMSIS-DSP pattern segments are incomplete: {path}"
        )
    return {name: bytes(value) for name, value in segments.items()}


def _f32_literal(raw: bytes) -> str:
    value = struct.unpack("<f", raw)[0]
    if not math.isfinite(value):
        raise WorkloadProviderError(
            "CMSIS-DSP direct protocol requires finite float input"
        )
    return f"{value.hex()}f"


def _cmsis_dsp_scalar_literals(
    pattern_bytes: bytes, scalar: str, count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    byte_count = count * 2 * 4
    if len(pattern_bytes) < byte_count:
        raise WorkloadProviderError(
            "CMSIS-DSP direct protocol pattern is smaller than its input extent"
        )
    chunks = tuple(
        pattern_bytes[offset : offset + 4] for offset in range(0, byte_count, 4)
    )
    if scalar == "float32_t":
        values = tuple(_f32_literal(chunk) for chunk in chunks)
    else:
        values = tuple(
            str(int.from_bytes(chunk, byteorder="little", signed=True))
            for chunk in chunks
        )
    return values[:count], values[count:]


def _format_cpp_array(values: Sequence[str]) -> str:
    lines = [
        "  " + ", ".join(values[index : index + 4])
        for index in range(0, len(values), 4)
    ]
    return ",\n".join(lines)


def _decode_f32_pattern(raw: bytes, name: str) -> tuple[str, ...]:
    if len(raw) % 4 != 0:
        raise WorkloadProviderError(f"CMSIS-DSP {name} is not f32-aligned")
    return tuple(
        _f32_literal(raw[offset : offset + 4]) for offset in range(0, len(raw), 4)
    )


def _decode_i16_pattern(raw: bytes, name: str) -> tuple[int, ...]:
    if len(raw) % 2 != 0:
        raise WorkloadProviderError(f"CMSIS-DSP {name} is not i16-aligned")
    return tuple(
        int.from_bytes(raw[offset : offset + 2], byteorder="little", signed=True)
        for offset in range(0, len(raw), 2)
    )


def _require_pattern_segment(segments: dict[str, bytes], name: str) -> bytes:
    try:
        value = segments[name]
    except KeyError as exc:
        raise WorkloadProviderError(
            f"generated CMSIS-DSP patterns omit {name}"
        ) from exc
    if not value:
        raise WorkloadProviderError(f"generated CMSIS-DSP pattern {name} is empty")
    return value


def _render_stateful_fir_f32_protocol(patterns: Path) -> str:
    segments = _cmsis_dsp_pattern_segments(patterns)
    inputs = _decode_f32_pattern(
        _require_pattern_segment(segments, "FirInput1_f32.txt"), "FIR input"
    )
    coefficients = _decode_f32_pattern(
        _require_pattern_segment(segments, "FirCoefs1_f32.txt"),
        "FIR coefficients",
    )
    expected = _decode_f32_pattern(
        _require_pattern_segment(segments, "FirRefs1_f32.txt"), "FIR reference"
    )
    configs = _decode_i16_pattern(
        _require_pattern_segment(segments, "FirConfigs1_s16.txt"),
        "FIR configuration",
    )
    if len(configs) % 2 != 0 or not configs:
        raise WorkloadProviderError("CMSIS-DSP FIR configuration is not paired")
    pairs = tuple(zip(configs[0::2], configs[1::2], strict=True))
    if any(block <= 0 or taps <= 0 for block, taps in pairs):
        raise WorkloadProviderError("CMSIS-DSP FIR configuration is nonpositive")
    if len(inputs) < 2 * max(block for block, _ in pairs):
        raise WorkloadProviderError("CMSIS-DSP FIR input does not cover its blocks")
    coefficient_count = sum(taps for _, taps in pairs)
    if len(coefficients) < coefficient_count:
        raise WorkloadProviderError("CMSIS-DSP FIR coefficient projection is not total")
    coefficients = coefficients[:coefficient_count]
    if len(expected) != sum(2 * block for block, _ in pairs):
        raise WorkloadProviderError("CMSIS-DSP FIR reference projection is not total")
    state_count = max(block + taps - 1 for block, taps in pairs)

    config_literals = tuple(str(value) for value in configs)
    return f"""#include <cstddef>
#include <cstdint>

#include "arm_math.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::size_t kConfigCount = {len(pairs)};
constexpr std::size_t kOutputCount = {len(expected)};
constexpr std::size_t kStateCount = {state_count};
constexpr float32_t kInput[] = {{
{_format_cpp_array(inputs)}
}};
constexpr float32_t kCoefficients[] = {{
{_format_cpp_array(coefficients)}
}};
constexpr std::int16_t kConfigs[] = {{
{_format_cpp_array(config_literals)}
}};
constexpr float32_t kExpected[] = {{
{_format_cpp_array(expected)}
}};

bool oracle_matches(const float32_t *output) {{
  for (std::size_t index = 0; index < kOutputCount; ++index) {{
    const float32_t expected = kExpected[index];
    const float32_t difference = output[index] > expected
                                     ? output[index] - expected
                                     : expected - output[index];
    const float32_t magnitude = expected < 0.0f ? -expected : expected;
    if (difference > 1.0e-6f + 3.0e-5f * magnitude)
      return false;
  }}
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE void {_CORPUS_OPERATOR_PROTOCOL_SYMBOL}(
    const float32_t *input, const float32_t *coefficients,
    const std::int16_t *configs, float32_t *state, float32_t *output) {{
  std::size_t coefficient_offset = 0;
  std::size_t output_offset = 0;
  for (std::size_t config = 0; config < kConfigCount; ++config) {{
    const std::uint32_t block_size = static_cast<std::uint32_t>(configs[2 * config]);
    const std::uint16_t num_taps = static_cast<std::uint16_t>(configs[2 * config + 1]);
    arm_fir_instance_f32 instance{{}};
    arm_fir_init_f32(&instance, num_taps, coefficients + coefficient_offset,
                     state, block_size);
    arm_fir_f32(&instance, input, output + output_offset, block_size);
    arm_fir_f32(&instance, input + block_size,
                output + output_offset + block_size, block_size);
    coefficient_offset += num_taps;
    output_offset += 2 * block_size;
  }}
}}

int main() {{
  float32_t state[kStateCount]{{}};
  float32_t output[kOutputCount]{{}};
  {_CORPUS_OPERATOR_PROTOCOL_SYMBOL}(kInput, kCoefficients, kConfigs, state,
                                     output);
  return oracle_matches(output) ? 0 : 1;
}}
"""


def _render_stateless_abs_f32_protocol(patterns: Path, sample_count: int) -> str:
    segments = _cmsis_dsp_pattern_segments(patterns)
    inputs = _decode_f32_pattern(
        _require_pattern_segment(segments, "Input1_f32.txt"), "absolute input"
    )
    expected = _decode_f32_pattern(
        _require_pattern_segment(segments, "Reference10_f32.txt"),
        "absolute reference",
    )
    if sample_count <= 0 or len(inputs) < sample_count or len(expected) < sample_count:
        raise WorkloadProviderError(
            "CMSIS-DSP absolute pattern does not cover its workload extent"
        )
    inputs = inputs[:sample_count]
    expected = expected[:sample_count]

    return f"""#include <cstddef>
#include <cstdint>
#include <cstring>

#include "arm_math.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::uint32_t kSampleCount = {sample_count};
constexpr float32_t kInput[] = {{
{_format_cpp_array(inputs)}
}};
constexpr float32_t kExpected[] = {{
{_format_cpp_array(expected)}
}};

bool oracle_matches(const float32_t *output) {{
  return std::memcmp(output, kExpected, sizeof(kExpected)) == 0;
}}
}} // namespace

extern "C" LOOM_NOINLINE void {_CORPUS_OPERATOR_PROTOCOL_SYMBOL}(
    const float32_t *input, float32_t *output, std::uint32_t count) {{
  arm_abs_f32(input, output, count);
}}

int main() {{
  float32_t input[kSampleCount];
  float32_t output[kSampleCount]{{}};
  for (std::uint32_t index = 0; index < kSampleCount; ++index)
    input[index] = kInput[index];
  {_CORPUS_OPERATOR_PROTOCOL_SYMBOL}(input, output, kSampleCount);
  return oracle_matches(output) ? 0 : 1;
}}
"""


def _render_stateless_controller_protocol(
    workload: corpus_inventory.ProgramWorkload,
    pattern_bytes: bytes,
    sample_count: int,
) -> str:
    call = workload.protocol[0]
    if _CMSIS_DSP_C_IDENTIFIER.fullmatch(call.symbol) is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP protocol symbol is not a C identifier: {call.symbol}"
        )
    is_float = call.signature.startswith("void(float")
    scalar = "float32_t" if is_float else "q31_t"
    input_a, input_b = _cmsis_dsp_scalar_literals(pattern_bytes, scalar, sample_count)

    if call.signature in {"void(float,float,ptr,ptr)", "void(i32,i32,ptr,ptr)"}:
        wrapper_parameters = f"""const {scalar} *input_a,
    const {scalar} *input_b, {scalar} *output_a, {scalar} *output_b,
    std::uint32_t count"""
        call_arguments = (
            "input_a[index], input_b[index], &output_a[index], &output_b[index]"
        )
        main_arguments = "kInputA, kInputB, output_a, output_b, kSampleCount"
    elif call.signature in {
        "void(float,float,ptr,ptr,float,float)",
        "void(i32,i32,ptr,ptr,i32,i32)",
    }:
        wrapper_parameters = f"""const {scalar} *input_a,
    const {scalar} *input_b, {scalar} *output_a, {scalar} *output_b,
    std::uint32_t count, {scalar} coefficient_a,
    {scalar} coefficient_b"""
        call_arguments = (
            "input_a[index], input_b[index], &output_a[index], &output_b[index], "
            "coefficient_a, coefficient_b"
        )
        main_arguments = (
            "kInputA, kInputB, output_a, output_b, kSampleCount, kInputA[0], kInputB[0]"
        )
    elif call.signature in {"void(float,ptr,ptr)", "void(i32,ptr,ptr)"}:
        wrapper_parameters = f"""const {scalar} *input,
    {scalar} *output_a, {scalar} *output_b, std::uint32_t count"""
        call_arguments = "input[index], &output_a[index], &output_b[index]"
        main_arguments = "kInputA, output_a, output_b, kSampleCount"
    else:
        raise WorkloadProviderError(
            f"unsupported stateless controller signature: {call.signature}"
        )

    return f"""#include <cstddef>
#include <cstdint>

#include "arm_math.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::uint32_t kSampleCount = {sample_count};
constexpr {scalar} kInputA[] = {{
{_format_cpp_array(input_a)}
}};
constexpr {scalar} kInputB[] = {{
{_format_cpp_array(input_b)}
}};

std::uint32_t digest(const void *data, std::size_t size) {{
  const auto *bytes = static_cast<const unsigned char *>(data);
  std::uint32_t value = 2166136261u;
  for (std::size_t index = 0; index < size; ++index) {{
    value ^= bytes[index];
    value *= 16777619u;
  }}
  return value;
}}
}} // namespace

extern "C" LOOM_NOINLINE void {_CORPUS_OPERATOR_PROTOCOL_SYMBOL}(
    {wrapper_parameters}) {{
  for (std::uint32_t index = 0; index < count; ++index) {{
    {call.symbol}({call_arguments});
  }}
}}

int main() {{
  {scalar} output_a[kSampleCount]{{}};
  {scalar} output_b[kSampleCount]{{}};
  {_CORPUS_OPERATOR_PROTOCOL_SYMBOL}({main_arguments});
  const std::uint32_t first = digest(output_a, sizeof(output_a));
  const std::uint32_t second = digest(output_b, sizeof(output_b));
  return static_cast<int>(first ^ (second * 16777619u));
}}
"""


def _render_cmsis_dsp_harness_cmake(
    targets: Sequence[_CmsisDspCmakeTarget],
    support_sources: Sequence[Path],
) -> str:
    suite_libraries: dict[Path, tuple[str, str, str]] = {}
    for item in targets:
        if item.source_group is None:
            continue
        suite_libraries.setdefault(
            item.shared,
            (f"loom_dsp_{item.shared.name}", item.test_class, item.source_group),
        )

    suite_blocks = []
    for shared, (library, test_class, source_group) in suite_libraries.items():
        shared_include = _cmake_quote(shared / "GeneratedInclude")
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
            target_blocks.append(
                f'''add_executable({item.target}
  "{_cmake_quote(item.direct_source)}")
target_include_directories({item.target} PRIVATE
  "${{LOOM_CMSIS_DSP_SOURCE}}/Include")
target_compile_definitions({item.target} PRIVATE EMBEDDED NOTIMING)
target_compile_options({item.target} PRIVATE -fno-inline-functions)
target_link_libraries({item.target} PRIVATE CMSISDSP)
'''
            )
            continue
        assert item.source_group is not None
        library = suite_libraries[item.shared][0]
        generated_source = _cmake_quote(item.generated / "GeneratedSource")
        generated_include = _cmake_quote(item.generated / "GeneratedInclude")
        shared_include = _cmake_quote(item.shared / "GeneratedInclude")
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

    support_items = "\n".join(
        f'  "{_cmake_quote(source)}"' for source in support_sources
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
set(DISABLEFLOAT16 ON CACHE BOOL "" FORCE)
set(FASTBUILD OFF CACHE BOOL "" FORCE)
set(HELIUM OFF CACHE BOOL "" FORCE)
set(HOST ON CACHE BOOL "" FORCE)
set(MVEF OFF CACHE BOOL "" FORCE)
set(MVEI OFF CACHE BOOL "" FORCE)
set(NEON OFF CACHE BOOL "" FORCE)
set(NEONEXPERIMENTAL OFF CACHE BOOL "" FORCE)
add_subdirectory("${{LOOM_CMSIS_DSP_SOURCE}}/Source" cmsis-dsp)
target_compile_definitions(CMSISDSP PRIVATE ARM_DSP_CUSTOM_CONFIG)
target_compile_definitions(CMSISDSP PUBLIC ARM_DSP_TESTING)
target_include_directories(CMSISDSP PRIVATE
  "${{LOOM_CMSIS_DSP_SOURCE}}/Testing")

add_library(loom_cmsis_dsp_framework STATIC
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

{"".join(suite_blocks)}
{"".join(target_blocks)}
"""


def materialize_cmsis_dsp_harness(
    workloads: Sequence[corpus_inventory.ProgramWorkload],
    external_root: Path,
    destination: Path,
) -> CmsisDspHarness:
    if not workloads:
        raise WorkloadProviderError("CMSIS-DSP harness selection is empty")
    if destination.exists():
        raise WorkloadProviderError(
            f"CMSIS-DSP harness destination already exists: {destination}"
        )

    testing_root = external_root / "cmsis-dsp" / "Testing"
    parser_module, codegen_module, tree_module = _load_cmsis_dsp_codegen(testing_root)
    source_dir = destination / "source"
    generated_root = source_dir / "generated" / "targets"
    shared_root = source_dir / "generated" / "shared"
    generated_root.mkdir(parents=True)
    parsed: dict[Path, object] = {}
    shared: dict[tuple[Path, str], Path] = {}
    targets: list[str] = []
    shared_directories: list[Path] = []
    protocol_methods: list[tuple[str, str]] = []
    protocol_symbol_sets: list[tuple[str, ...]] = []
    protocol_source_owners: list[tuple[Path, Path]] = []
    expected_entry_results: list[int | None] = []
    cmake_targets: list[_CmsisDspCmakeTarget] = []

    for workload in workloads:
        if workload.suite != "cmsis-dsp" or not isinstance(
            workload.producer, corpus_inventory.CmsisDspWorkloadProducer
        ):
            raise WorkloadProviderError(
                f"workload is not owned by the CMSIS-DSP provider: {workload.identity}"
            )
        if not supports_cmsis_dsp_harness(workload):
            raise WorkloadProviderError(
                f"CMSIS-DSP workload has no exact harness provider: {workload.identity}"
            )
        target = workload.executable
        if target != corpus_inventory.operator_workload_target(workload.operator_id):
            raise WorkloadProviderError(
                f"CMSIS-DSP workload has an invalid target: {workload.identity}"
            )
        if target in targets:
            raise WorkloadProviderError(
                f"CMSIS-DSP harness repeats a target: {workload.identity}"
            )
        descriptor = Path(workload.producer.definition)
        expected_prefix = Path("externals/cmsis-dsp/Testing")
        try:
            relative = descriptor.relative_to(expected_prefix)
        except ValueError as exc:
            raise WorkloadProviderError(
                f"CMSIS-DSP descriptor escapes its owner: {descriptor}"
            ) from exc
        descriptor_path = testing_root / relative
        if not descriptor_path.is_file():
            raise WorkloadProviderError(
                f"CMSIS-DSP descriptor is unavailable: {descriptor_path}"
            )
        root = parsed.get(descriptor_path)
        if root is None:
            import pyparsing

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pyparsing.PyparsingDeprecationWarning)
                root = parser_module.Parser().parse(str(descriptor_path))
            parsed[descriptor_path] = root
        shared_key = (descriptor_path, workload.producer.test_class)
        shared_generated = shared.get(shared_key)
        if shared_generated is None:
            shared_generated = shared_root / _cmsis_dsp_shared_name(
                relative, workload.producer.test_class
            )
            suite_tree = _select_cmsis_dsp_suite_tree(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )
            _generate_cmsis_dsp_tree(
                codegen_module,
                testing_root,
                shared_generated,
                suite_tree,
            )
            shared[shared_key] = shared_generated
        selected = _select_cmsis_dsp_test_tree(
            root,
            tree_module.TreeElem.SUITE,
            tree_module.TreeElem.TEST,
            workload.producer.test_class,
            workload.producer.test_method,
            workload.producer.vector_ordinal,
        )
        generated = generated_root / target
        _generate_cmsis_dsp_tree(
            codegen_module,
            testing_root,
            generated,
            selected,
        )
        target_include = generated / "GeneratedInclude"
        shared_include = shared_generated / "GeneratedInclude"
        target_patterns = target_include / "Patterns.h"
        shared_patterns = shared_include / "Patterns.h"
        if target_patterns.read_bytes() != shared_patterns.read_bytes():
            raise WorkloadProviderError(
                "CMSIS-DSP selected test changes its suite-owned pattern layout: "
                f"{workload.identity}"
            )
        target_patterns.unlink()
        (target_include / f"{workload.producer.test_class}_decl.h").unlink()
        targets.append(target)
        shared_directories.append(shared_generated)
        protocol_methods.append(
            (workload.producer.test_class, workload.producer.test_method)
        )
        direct_family = _cmsis_dsp_direct_protocol_family(workload)
        if direct_family == "stateless-abs-f32":
            suite = _cmsis_dsp_suite_chain(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )[-1]
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                _render_stateless_abs_f32_protocol(
                    shared_patterns,
                    _cmsis_dsp_literal_test_extent(
                        suite,
                        tree_module.TreeElem.TEST,
                        workload.producer.test_method,
                        workload.producer.vector_ordinal,
                    ),
                ),
                encoding="utf-8",
            )
            protocol_symbol_sets.append((_CORPUS_OPERATOR_PROTOCOL_SYMBOL,))
            expected_entry_results.append(0)
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "basic_math_functions.h"
            )
            if not protocol_owner.is_file():
                raise WorkloadProviderError(
                    f"CMSIS-DSP protocol owner is unavailable: {protocol_owner}"
                )
            protocol_source_owners.append((direct_source, protocol_owner))
            cmake_targets.append(
                _CmsisDspCmakeTarget(
                    target=target,
                    generated=generated,
                    shared=shared_generated,
                    test_class=workload.producer.test_class,
                    direct_source=direct_source,
                )
            )
        elif direct_family == "stateless-controller":
            suite = _cmsis_dsp_suite_chain(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )[-1]
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                _render_stateless_controller_protocol(
                    workload,
                    _cmsis_dsp_pattern_bytes(shared_patterns),
                    _cmsis_dsp_first_parameter(suite),
                ),
                encoding="utf-8",
            )
            protocol_symbol_sets.append((_CORPUS_OPERATOR_PROTOCOL_SYMBOL,))
            expected_entry_results.append(None)
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "controller_functions.h"
            )
            if not protocol_owner.is_file():
                raise WorkloadProviderError(
                    f"CMSIS-DSP protocol owner is unavailable: {protocol_owner}"
                )
            protocol_source_owners.append((direct_source, protocol_owner))
            cmake_targets.append(
                _CmsisDspCmakeTarget(
                    target=target,
                    generated=generated,
                    shared=shared_generated,
                    test_class=workload.producer.test_class,
                    direct_source=direct_source,
                )
            )
        elif direct_family == "stateful-fir-f32":
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                _render_stateful_fir_f32_protocol(shared_patterns),
                encoding="utf-8",
            )
            protocol_symbol_sets.append((_CORPUS_OPERATOR_PROTOCOL_SYMBOL,))
            expected_entry_results.append(0)
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "filtering_functions.h"
            )
            if not protocol_owner.is_file():
                raise WorkloadProviderError(
                    f"CMSIS-DSP protocol owner is unavailable: {protocol_owner}"
                )
            protocol_source_owners.append((direct_source, protocol_owner))
            cmake_targets.append(
                _CmsisDspCmakeTarget(
                    target=target,
                    generated=generated,
                    shared=shared_generated,
                    test_class=workload.producer.test_class,
                    direct_source=direct_source,
                )
            )
        else:
            source_group = "Tests"
            protocol_source = (
                testing_root
                / "Source"
                / source_group
                / f"{workload.producer.test_class}.cpp"
            )
            if not protocol_source.is_file():
                raise WorkloadProviderError(
                    f"CMSIS-DSP protocol owner is unavailable: {protocol_source}"
                )
            protocol_symbol_sets.append(())
            expected_entry_results.append(None)
            protocol_source_owners.append((protocol_source, protocol_source))
            cmake_targets.append(
                _CmsisDspCmakeTarget(
                    target=target,
                    generated=generated,
                    shared=shared_generated,
                    test_class=workload.producer.test_class,
                    source_group=source_group,
                )
            )

    (source_dir / "CMakeLists.txt").write_text(
        _render_cmsis_dsp_harness_cmake(
            cmake_targets,
            tuple(sorted((testing_root / "Source" / "Tests").glob("*.c"))),
        ),
        encoding="utf-8",
    )
    (source_dir / "OperatorMain.cpp").write_text(
        """extern int testmain(const char *patterns);
extern "C" const char *patternData;

int main() { return testmain(patternData); }
""",
        encoding="utf-8",
    )
    return CmsisDspHarness(
        source_dir,
        tuple(targets),
        tuple(shared_directories),
        tuple(protocol_methods),
        tuple(protocol_symbol_sets),
        tuple(protocol_source_owners),
        tuple(expected_entry_results),
    )


def cmake_configure_command(
    harness: CmsisNnHarness | CmsisDspHarness,
    external_source: Path,
    build_dir: Path,
    toolchain: CmakeToolchain,
) -> list[str]:
    if isinstance(harness, CmsisNnHarness):
        owner_definitions = [f"-DLOOM_CMSIS_NN_SOURCE={external_source}"]
        if harness.unity_source is not None:
            owner_definitions.append(f"-DLOOM_UNITY_SOURCE={harness.unity_source}")
    elif isinstance(harness, CmsisDspHarness):
        owner_definitions = [f"-DLOOM_CMSIS_DSP_SOURCE={external_source}"]
    else:
        raise WorkloadProviderError("unknown CMake workload harness owner")
    command = [
        "cmake",
        "-S",
        str(harness.source_dir),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DCMAKE_C_COMPILER={toolchain.c_compiler}",
        f"-DCMAKE_CXX_COMPILER={toolchain.cxx_compiler}",
        f"-DCMAKE_AR={toolchain.archiver}",
        f"-DCMAKE_RANLIB={toolchain.ranlib}",
        *owner_definitions,
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
        "--",
        "-k",
        "0",
    ]
