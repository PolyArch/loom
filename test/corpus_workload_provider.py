#!/usr/bin/env python3
"""Ephemeral builders for corpus-owned linked program workloads."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import corpus_inventory
import corpus_dsp_atomic
import corpus_dsp_cmake
import corpus_dsp_convolution
import corpus_dsp_distance
import corpus_dsp_fast_math
import corpus_dsp_fft
import corpus_dsp_filter_generated
import corpus_dsp_generated
import corpus_dsp_inline
import corpus_dsp_lms
import corpus_dsp_matrix
import corpus_dsp_pid
import corpus_dsp_protocol
import corpus_dsp_stateful
import corpus_dsp_transform
import corpus_nn_workload_provider
from corpus_nn_workload_provider import CmsisNnHarness
from corpus_workload_errors import WorkloadProviderError


@dataclass(frozen=True)
class CmsisDspHarness:
    source_dir: Path
    targets: tuple[str, ...]
    shared_directories: tuple[Path, ...]
    protocol_methods: tuple[tuple[str, str], ...]
    protocol_symbol_sets: tuple[tuple[str, ...], ...]
    protocol_sources: tuple[Path, ...]
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

    def protocol_source(self, target: str) -> Path:
        return self.protocol_sources[self.targets.index(self._target(target))]

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


def materialize_cmsis_nn_harness(
    workloads: Sequence[corpus_inventory.ProgramWorkload],
    external_root: Path,
    destination: Path,
) -> CmsisNnHarness:
    return corpus_nn_workload_provider.materialize_cmsis_nn_harness(
        workloads,
        external_root,
        destination,
        _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
    )


def supports_cmsis_nn_harness(
    workload: corpus_inventory.ProgramWorkload,
) -> bool:
    return corpus_nn_workload_provider.supports_cmsis_nn_harness(workload)


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


_CORPUS_OPERATOR_PROTOCOL_SYMBOL = "loom_corpus_operator_protocol"
_CMSIS_DSP_STATELESS_CONTROLLER_SIGNATURES = {
    "void(float,float,ptr,ptr)",
    "void(float,float,ptr,ptr,float,float)",
    "void(float,ptr,ptr)",
    "void(i32,i32,ptr,ptr)",
    "void(i32,i32,ptr,ptr,i32,i32)",
    "void(i32,ptr,ptr)",
}
_CMSIS_DSP_BASIC_F32_PROTOCOLS = {
    "test_add_f32": ("arm_add_f32", "void(ptr,ptr,ptr,i32)"),
    "test_clip_f32": ("arm_clip_f32", "void(ptr,ptr,float,float,i32)"),
    "test_dot_prod_f32": ("arm_dot_prod_f32", "void(ptr,ptr,i32,ptr)"),
    "test_mult_f32": ("arm_mult_f32", "void(ptr,ptr,ptr,i32)"),
    "test_negate_f32": ("arm_negate_f32", "void(ptr,ptr,i32)"),
    "test_offset_f32": ("arm_offset_f32", "void(ptr,float,ptr,i32)"),
    "test_scale_f32": ("arm_scale_f32", "void(ptr,float,ptr,i32)"),
    "test_sub_f32": ("arm_sub_f32", "void(ptr,ptr,ptr,i32)"),
}


def _cmsis_dsp_direct_protocol_family(
    workload: corpus_inventory.ProgramWorkload,
) -> str | None:
    if corpus_dsp_inline.header_defined_protocol(workload) is not None:
        return "header-defined-inline"
    if corpus_dsp_atomic.atomic_protocol(workload) is not None:
        return "atomic-multicall"
    if corpus_dsp_convolution.partial_convolution_protocol(workload) is not None:
        return "stateless-partial-convolution"
    if corpus_dsp_distance.distance_protocol(workload) is not None:
        return "stateless-distance"
    if corpus_dsp_fast_math.fixed_point_divide_protocol(workload) is not None:
        return "stateless-fixed-point-divide"
    if corpus_dsp_transform.transform_protocol(workload) is not None:
        return "atomic-transform"
    if corpus_dsp_filter_generated.stateful_filter_protocol(workload) is not None:
        return "generated-stateful-filter"
    if corpus_dsp_filter_generated.sequence_protocol(workload) is not None:
        return "generated-sequence"
    if corpus_dsp_generated.transform_query_protocol(workload) is not None:
        return "generated-transform-query"
    if corpus_dsp_generated.lifecycle_protocol(workload) is not None:
        return "generated-lifecycle"
    if corpus_dsp_matrix.floating_matrix_protocol(workload) is not None:
        return "floating-matrix"
    if corpus_dsp_matrix.matrix_vector_protocol(workload) is not None:
        return "stateless-matrix-vector"
    if corpus_dsp_matrix.matrix_multiplication_protocol(workload) is not None:
        return "stateless-matrix-multiplication"
    if corpus_dsp_pid.pid_protocol(workload) is not None:
        return "stateful-pid"
    if corpus_dsp_lms.lms_protocol(workload) is not None:
        return "stateful-lms"
    if corpus_dsp_fft.legacy_cfft_protocol(workload) is not None:
        return "legacy-cfft"
    if corpus_dsp_fft.radix8_f16_protocol(workload) is not None:
        return "generated-radix8-f16"
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
    basic_f32 = _CMSIS_DSP_BASIC_F32_PROTOCOLS.get(producer.test_method)
    if (
        producer.selector_kind == "official"
        and producer.test_class == "BasicTestsF32"
        and basic_f32 is not None
        and tuple((call.symbol, call.signature) for call in workload.protocol)
        == (basic_f32,)
    ):
        return "stateless-basic-f32"
    if corpus_dsp_protocol.basic_integer_protocol(workload) is not None:
        return "stateless-basic-integer"
    if corpus_dsp_protocol.window_protocol(workload) is not None:
        return "stateless-window"
    if corpus_dsp_protocol.elementary_math_protocol(workload) is not None:
        return "stateless-elementary-math"
    if corpus_dsp_stateful.fir_protocol(workload) is not None:
        return "stateful-fir"
    if corpus_dsp_stateful.svm_protocol(workload) is not None:
        return "stateful-svm"
    if corpus_dsp_stateful.biquad_protocol(workload) is not None:
        return "stateful-biquad"
    if corpus_dsp_stateful.rate_conversion_protocol(workload) is not None:
        return "stateful-rate-conversion"
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
    family = _cmsis_dsp_direct_protocol_family(workload)
    if isinstance(producer, corpus_inventory.CmsisDspGeneratedWorkloadProducer):
        return family in {
            "generated-lifecycle",
            "generated-radix8-f16",
            "generated-sequence",
            "generated-stateful-filter",
            "generated-transform-query",
        }
    return isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer) and (
        producer.selector_kind == "official" or family is not None
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


def _cmsis_dsp_named_dimensions(
    suite: object, names: tuple[str, ...], vector_ordinal: int
) -> tuple[int, ...]:
    parameter_id = suite.data.get("PARAMID")
    matching = [values for _, name, values in suite.parameters if name == parameter_id]
    if len(matching) != 1 or len(matching[0]) != len(names):
        raise WorkloadProviderError(
            "CMSIS-DSP typed protocol requires one parameter row"
        )
    if tuple(suite.params.full) != names:
        raise WorkloadProviderError(
            "CMSIS-DSP typed protocol has an invalid dimension schema"
        )
    dimensions = []
    for expected_name, column in zip(suite.params.full, matching[0], strict=True):
        if column.get("NAME") != expected_name:
            raise WorkloadProviderError(
                "CMSIS-DSP typed protocol has an invalid dimension column"
            )
        values = column.get("INTS")
        if (
            not isinstance(values, list)
            or vector_ordinal >= len(values)
            or not isinstance(values[vector_ordinal], int)
            or isinstance(values[vector_ordinal], bool)
            or values[vector_ordinal] <= 0
        ):
            raise WorkloadProviderError(
                "CMSIS-DSP typed protocol has invalid dimensions"
            )
        dimensions.append(values[vector_ordinal])
    return tuple(dimensions)


def _cmsis_dsp_legacy_cfft_length(
    suite: object, test_kind: int, test_method: str
) -> int:
    if tuple(suite.params.full) != ("NB", "IFFT", "BITREV"):
        raise WorkloadProviderError(
            "CMSIS-DSP legacy CFFT protocol has an invalid parameter schema"
        )
    matching_tests = [
        child
        for child in suite.children
        if child.kind == test_kind and child.data.get("class") == test_method
    ]
    if len(matching_tests) != 1:
        raise WorkloadProviderError(
            "CMSIS-DSP legacy CFFT protocol requires one descriptor test"
        )
    parameter_id = matching_tests[0].data.get("PARAMID")
    matching_rows = [
        values for _, name, values in suite.parameters if name == parameter_id
    ]
    if len(matching_rows) != 1 or len(matching_rows[0]) != 3:
        raise WorkloadProviderError(
            "CMSIS-DSP legacy CFFT protocol requires one parameter row"
        )
    names = tuple(column.get("NAME") for column in matching_rows[0])
    if names != ("NB", "IFFT", "REV"):
        raise WorkloadProviderError(
            "CMSIS-DSP legacy CFFT protocol has invalid parameter columns"
        )
    lengths, directions, bit_reversal = (
        column.get("INTS") for column in matching_rows[0]
    )
    if (
        not isinstance(lengths, list)
        or not lengths
        or any(not isinstance(value, int) or value <= 0 for value in lengths)
        or not isinstance(directions, list)
        or 0 not in directions
        or not isinstance(bit_reversal, list)
        or 1 not in bit_reversal
    ):
        raise WorkloadProviderError(
            "CMSIS-DSP legacy CFFT protocol has invalid parameter values"
        )
    return min(lengths)


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


def _cmsis_dsp_basic_integer_extent(
    suite: object,
    test_kind: int,
    workload: corpus_inventory.ProgramWorkload,
    operation: str,
) -> int | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer):
        raise WorkloadProviderError("CMSIS-DSP integer workload owner is invalid")
    if operation in {"shift", "clip"}:
        return None
    if (
        producer.test_class == "BasicTestsQ7"
        and producer.test_method == "test_mult_q7"
        and producer.vector_ordinal == 0
    ):
        return 32
    return _cmsis_dsp_literal_test_extent(
        suite,
        test_kind,
        producer.test_method,
        producer.vector_ordinal,
    )


def materialize_cmsis_dsp_harness(
    workloads: Sequence[corpus_inventory.ProgramWorkload],
    external_root: Path,
    destination: Path,
) -> CmsisDspHarness:
    if not workloads:
        raise WorkloadProviderError("CMSIS-DSP harness selection is empty")
    target_profiles = {workload.target_profile for workload in workloads}
    if len(target_profiles) != 1:
        raise WorkloadProviderError("CMSIS-DSP harness selection mixes target profiles")
    target_profile = next(iter(target_profiles))
    if target_profile not in {
        corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
        corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE,
    }:
        raise WorkloadProviderError(
            f"CMSIS-DSP target profile provider is unavailable: {target_profile}"
        )
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
    protocol_sources: list[Path] = []
    expected_entry_results: list[int | None] = []
    cmake_targets: list[corpus_dsp_cmake.CmakeTarget] = []
    operator_compile_options: dict[Path, tuple[str, ...]] = {}
    try:
        compile_option_authority = corpus_inventory.load_workload_inventory(
            external_root.parent
        )
    except corpus_inventory.InventoryError as exc:
        raise WorkloadProviderError(
            f"cannot resolve CMSIS-DSP source compile options: {exc}"
        ) from exc
    for owner in compile_option_authority:
        if owner.suite != "cmsis-dsp" or owner.target_profile != target_profile:
            continue
        for source_name in owner.sources:
            source = Path(source_name)
            try:
                relative_source = source.relative_to("externals/cmsis-dsp")
            except ValueError as exc:
                raise WorkloadProviderError(
                    f"CMSIS-DSP operator source escapes its owner: {source}"
                ) from exc
            previous = operator_compile_options.setdefault(
                relative_source, owner.compiler_flags
            )
            if previous != owner.compiler_flags:
                raise WorkloadProviderError(
                    "CMSIS-DSP workloads assign conflicting compiler flags to "
                    f"operator source {source}"
                )

    def record_direct_protocol(
        workload: corpus_inventory.ProgramWorkload,
        target: str,
        generated: Path,
        shared_generated: Path,
        direct_source: Path,
        protocol_owner: Path,
        expected_entry_result: int | None,
        support_sources: tuple[Path, ...] = (),
        include_directories: tuple[Path, ...] = (),
    ) -> None:
        if not protocol_owner.is_file():
            raise WorkloadProviderError(
                f"CMSIS-DSP protocol owner is unavailable: {protocol_owner}"
            )
        protocol_symbol_sets.append((_CORPUS_OPERATOR_PROTOCOL_SYMBOL,))
        expected_entry_results.append(expected_entry_result)
        protocol_sources.append(direct_source)
        producer = workload.producer
        cmake_targets.append(
            corpus_dsp_cmake.CmakeTarget(
                target=target,
                generated=generated,
                shared=shared_generated,
                test_class=(
                    producer.test_class
                    if isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer)
                    else None
                ),
                direct_source=direct_source,
                direct_support_sources=support_sources,
                direct_include_directories=include_directories,
                compiler_flags=workload.compiler_flags,
            )
        )

    for workload in workloads:
        if workload.suite != "cmsis-dsp" or not isinstance(
            workload.producer,
            (
                corpus_inventory.CmsisDspWorkloadProducer,
                corpus_inventory.CmsisDspGeneratedWorkloadProducer,
            ),
        ):
            raise WorkloadProviderError(
                f"workload is not owned by the CMSIS-DSP provider: {workload.identity}"
            )
        if not supports_cmsis_dsp_harness(workload):
            raise WorkloadProviderError(
                f"CMSIS-DSP workload has no exact harness provider: {workload.identity}"
            )
        for source_name in workload.sources:
            source = Path(source_name)
            try:
                relative_source = source.relative_to("externals/cmsis-dsp")
            except ValueError as exc:
                raise WorkloadProviderError(
                    f"CMSIS-DSP operator source escapes its owner: {source}"
                ) from exc
            if not (external_root / "cmsis-dsp" / relative_source).is_file():
                raise WorkloadProviderError(
                    f"CMSIS-DSP operator source is unavailable: {source}"
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
        direct_family = _cmsis_dsp_direct_protocol_family(workload)
        if direct_family == "stateless-fixed-point-divide":
            protocol = corpus_dsp_fast_math.fixed_point_divide_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP fixed-point divide protocol is inconsistent"
                )
            generated = generated_root / target
            generated.mkdir()
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_fast_math.render_fixed_point_divide_protocol(
                    workload,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "fast_math_functions.h"
            )
            targets.append(target)
            shared_directories.append(source_dir)
            protocol_methods.append(("FixedPointDivide", protocol.symbol))
            record_direct_protocol(
                workload,
                target,
                generated,
                source_dir,
                direct_source,
                protocol_owner,
                0,
            )
            continue
        if direct_family == "stateless-distance":
            protocol = corpus_dsp_distance.distance_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP distance protocol is inconsistent"
                )
            generated = generated_root / target
            generated.mkdir()
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_distance.render_distance_protocol(
                    workload,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / "dsp" / protocol.owner_header
            )
            targets.append(target)
            shared_directories.append(source_dir)
            protocol_methods.append(("Distance", protocol.symbol))
            record_direct_protocol(
                workload,
                target,
                generated,
                source_dir,
                direct_source,
                protocol_owner,
                0,
            )
            continue
        if direct_family == "stateless-partial-convolution":
            protocol = corpus_dsp_convolution.partial_convolution_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP partial convolution protocol is inconsistent"
                )
            generated = generated_root / target
            generated.mkdir()
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_convolution.render_partial_convolution_protocol(
                    workload,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "filtering_functions.h"
            )
            targets.append(target)
            shared_directories.append(source_dir)
            protocol_methods.append(("PartialConvolution", protocol.symbol))
            record_direct_protocol(
                workload,
                target,
                generated,
                source_dir,
                direct_source,
                protocol_owner,
                0,
            )
            continue
        if direct_family == "stateless-matrix-vector":
            protocol = corpus_dsp_matrix.matrix_vector_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP matrix-vector protocol is inconsistent"
                )
            generated = generated_root / target
            generated.mkdir()
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_matrix.render_matrix_vector_protocol(
                    workload,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / "dsp" / protocol.owner_header
            )
            targets.append(target)
            shared_directories.append(source_dir)
            protocol_methods.append(("MatrixVector", protocol.symbol))
            record_direct_protocol(
                workload,
                target,
                generated,
                source_dir,
                direct_source,
                protocol_owner,
                0,
            )
            continue
        if direct_family in {
            "generated-lifecycle",
            "generated-radix8-f16",
            "generated-sequence",
            "generated-stateful-filter",
            "generated-transform-query",
        }:
            sequence = corpus_dsp_filter_generated.sequence_protocol(workload)
            stateful_filter = corpus_dsp_filter_generated.stateful_filter_protocol(
                workload
            )
            transform_query = corpus_dsp_generated.transform_query_protocol(workload)
            lifecycle = corpus_dsp_generated.lifecycle_protocol(workload)
            radix8_f16 = corpus_dsp_fft.radix8_f16_protocol(workload)
            protocols = tuple(
                protocol
                for protocol in (
                    sequence,
                    stateful_filter,
                    transform_query,
                    lifecycle,
                    radix8_f16,
                )
                if protocol is not None
            )
            if len(protocols) != 1:
                raise WorkloadProviderError("CMSIS-DSP generated protocol is ambiguous")
            protocol = protocols[0]
            generated = generated_root / target
            generated.mkdir()
            direct_source = generated / "OperatorProtocol.cpp"
            rendered = (
                corpus_dsp_filter_generated.render_sequence_protocol(
                    workload, _CORPUS_OPERATOR_PROTOCOL_SYMBOL
                )
                if sequence is not None
                else corpus_dsp_filter_generated.render_stateful_filter_protocol(
                    workload, _CORPUS_OPERATOR_PROTOCOL_SYMBOL
                )
                if stateful_filter is not None
                else corpus_dsp_generated.render_transform_query_protocol(
                    workload, _CORPUS_OPERATOR_PROTOCOL_SYMBOL
                )
                if transform_query is not None
                else corpus_dsp_generated.render_lifecycle_protocol(
                    workload, _CORPUS_OPERATOR_PROTOCOL_SYMBOL
                )
                if lifecycle is not None
                else corpus_dsp_fft.render_radix8_f16_protocol(
                    workload, _CORPUS_OPERATOR_PROTOCOL_SYMBOL
                )
            )
            direct_source.write_text(rendered, encoding="utf-8")
            protocol_owner = (
                external_root / "cmsis-dsp" / radix8_f16.owner_source
                if radix8_f16 is not None
                else external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / protocol.owner_header
            )
            targets.append(target)
            shared_directories.append(source_dir)
            protocol_methods.append(("TransformQuery", protocol.symbol))
            record_direct_protocol(
                workload,
                target,
                generated,
                source_dir,
                direct_source,
                protocol_owner,
                0,
            )
            continue
        if not isinstance(workload.producer, corpus_inventory.CmsisDspWorkloadProducer):
            raise WorkloadProviderError(
                f"CMSIS-DSP generated workload has no provider: {workload.identity}"
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
        if direct_family == "header-defined-inline":
            protocol = corpus_dsp_inline.header_defined_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP header-defined protocol is inconsistent"
                )
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_inline.render_header_defined_protocol(
                    workload,
                    shared_patterns,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / protocol.owner_header
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateless-abs-f32":
            suite = _cmsis_dsp_suite_chain(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )[-1]
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_protocol.render_stateless_abs_f32_protocol(
                    shared_patterns,
                    _cmsis_dsp_literal_test_extent(
                        suite,
                        tree_module.TreeElem.TEST,
                        workload.producer.test_method,
                        workload.producer.vector_ordinal,
                    ),
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "basic_math_functions.h"
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateless-basic-f32":
            suite = _cmsis_dsp_suite_chain(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )[-1]
            sample_count = None
            if workload.protocol[0].symbol != "arm_clip_f32":
                sample_count = _cmsis_dsp_literal_test_extent(
                    suite,
                    tree_module.TreeElem.TEST,
                    workload.producer.test_method,
                    workload.producer.vector_ordinal,
                )
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_protocol.render_basic_f32_protocol(
                    workload,
                    shared_patterns,
                    sample_count,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "basic_math_functions.h"
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateless-basic-integer":
            protocol = corpus_dsp_protocol.basic_integer_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP basic integer protocol is inconsistent"
                )
            _, operation, _, _ = protocol
            suite = _cmsis_dsp_suite_chain(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )[-1]
            sample_count = _cmsis_dsp_basic_integer_extent(
                suite,
                tree_module.TreeElem.TEST,
                workload,
                operation,
            )
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_protocol.render_basic_integer_protocol(
                    workload,
                    shared_patterns,
                    sample_count,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "basic_math_functions.h"
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateless-window":
            suite = _cmsis_dsp_suite_chain(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )[-1]
            reference_name = corpus_dsp_protocol.window_reference_name(
                suite,
                tree_module.TreeElem.TEST,
                workload,
            )
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_protocol.render_window_protocol(
                    workload,
                    shared_patterns,
                    reference_name,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / "dsp" / "window_functions.h"
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateless-elementary-math":
            protocol = corpus_dsp_protocol.elementary_math_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP elementary math protocol is inconsistent"
                )
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_protocol.render_elementary_math_protocol(
                    workload,
                    shared_patterns,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            owner_header = (
                "statistics_functions.h"
                if protocol.kind in {"reduction", "binary-reduction"}
                else "fast_math_functions.h"
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / "dsp" / owner_header
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateless-controller":
            suite = _cmsis_dsp_suite_chain(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )[-1]
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_protocol.render_stateless_controller_protocol(
                    workload,
                    corpus_dsp_protocol.pattern_bytes(shared_patterns),
                    _cmsis_dsp_first_parameter(suite),
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "controller_functions.h"
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                None,
            )
        elif direct_family == "stateful-fir":
            protocol = corpus_dsp_stateful.fir_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError("CMSIS-DSP FIR protocol is inconsistent")
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_stateful.render_fir_protocol(
                    workload,
                    shared_patterns,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / "dsp" / protocol.owner_header
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "atomic-multicall":
            protocol = corpus_dsp_atomic.atomic_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError("CMSIS-DSP atomic protocol is inconsistent")
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_atomic.render_atomic_protocol(
                    workload, _CORPUS_OPERATOR_PROTOCOL_SYMBOL
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / "dsp" / protocol.owner_header
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "atomic-transform":
            protocol = corpus_dsp_transform.transform_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP transform protocol is inconsistent"
                )
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_transform.render_transform_protocol(
                    workload,
                    shared_patterns,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / "dsp" / protocol.owner_header
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
                support_sources=(
                    testing_root / "Source" / "Tests" / protocol.data_source,
                )
                if isinstance(protocol, corpus_dsp_transform.MfccProtocol)
                else (),
                include_directories=(testing_root / "Include" / "Tests",)
                if isinstance(protocol, corpus_dsp_transform.MfccProtocol)
                else (),
            )
        elif direct_family == "floating-matrix":
            protocol = corpus_dsp_matrix.floating_matrix_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP floating matrix protocol is inconsistent"
                )
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_matrix.render_floating_matrix_protocol(
                    workload,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / "dsp" / protocol.owner_header
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateless-matrix-multiplication":
            matrix_protocol = corpus_dsp_matrix.matrix_multiplication_protocol(workload)
            if matrix_protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP matrix multiplication protocol is inconsistent"
                )
            suite = _cmsis_dsp_suite_chain(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )[-1]
            direct_source = generated / "OperatorProtocol.cpp"
            fixed_anchor = (
                len(workload.protocol) == 2 or matrix_protocol.input_a_pattern is None
            )
            direct_source.write_text(
                corpus_dsp_matrix.render_matrix_multiplication_protocol(
                    workload,
                    None if fixed_anchor else shared_patterns,
                    (2, 2, 2)
                    if fixed_anchor
                    else _cmsis_dsp_named_dimensions(
                        suite,
                        ("NBR", "NBI", "NBC"),
                        workload.producer.vector_ordinal,
                    ),
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / "arm_math.h"
                if len(workload.protocol) == 2
                else external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "matrix_functions.h"
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateful-pid":
            suite = _cmsis_dsp_suite_chain(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )[-1]
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_pid.render_pid_protocol(
                    workload,
                    shared_patterns,
                    _cmsis_dsp_first_parameter(suite),
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "controller_functions.h"
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateful-lms":
            suite = _cmsis_dsp_suite_chain(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )[-1]
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_lms.render_lms_protocol(
                    workload,
                    shared_patterns,
                    _cmsis_dsp_named_dimensions(
                        suite,
                        ("NumTaps", "NB"),
                        workload.producer.vector_ordinal,
                    ),
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "filtering_functions.h"
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "legacy-cfft":
            protocol = corpus_dsp_fft.legacy_cfft_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP legacy CFFT protocol is inconsistent"
                )
            suite = _cmsis_dsp_suite_chain(
                root,
                tree_module.TreeElem.SUITE,
                workload.producer.test_class,
            )[-1]
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_fft.render_legacy_cfft_protocol(
                    workload,
                    shared_patterns,
                    _cmsis_dsp_legacy_cfft_length(
                        suite,
                        tree_module.TreeElem.TEST,
                        workload.producer.test_method,
                    ),
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / "dsp" / protocol.owner_header
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateful-svm":
            protocol = corpus_dsp_stateful.svm_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError("CMSIS-DSP SVM protocol is inconsistent")
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_stateful.render_svm_protocol(
                    workload,
                    shared_patterns,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / protocol.value.owner_header
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateful-biquad":
            protocol = corpus_dsp_stateful.biquad_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError("CMSIS-DSP Biquad protocol is inconsistent")
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_stateful.render_biquad_protocol(
                    workload,
                    shared_patterns,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root / "cmsis-dsp" / "Include" / "dsp" / protocol.owner_header
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
            )
        elif direct_family == "stateful-rate-conversion":
            protocol = corpus_dsp_stateful.rate_conversion_protocol(workload)
            if protocol is None:
                raise WorkloadProviderError(
                    "CMSIS-DSP rate-conversion protocol is inconsistent"
                )
            direct_source = generated / "OperatorProtocol.cpp"
            direct_source.write_text(
                corpus_dsp_stateful.render_rate_conversion_protocol(
                    workload,
                    shared_patterns,
                    _CORPUS_OPERATOR_PROTOCOL_SYMBOL,
                ),
                encoding="utf-8",
            )
            protocol_owner = (
                external_root
                / "cmsis-dsp"
                / "Include"
                / "dsp"
                / "filtering_functions.h"
            )
            record_direct_protocol(
                workload,
                target,
                generated,
                shared_generated,
                direct_source,
                protocol_owner,
                0,
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
            protocol_sources.append(protocol_source)
            cmake_targets.append(
                corpus_dsp_cmake.CmakeTarget(
                    target=target,
                    generated=generated,
                    shared=shared_generated,
                    test_class=workload.producer.test_class,
                    source_group=source_group,
                )
            )

    (source_dir / "CMakeLists.txt").write_text(
        corpus_dsp_cmake.render_harness(
            cmake_targets,
            tuple(sorted((testing_root / "Source" / "Tests").glob("*.c"))),
            operator_compile_options,
            enable_float16=(
                target_profile == corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE
            ),
            enable_wrapper=any(
                (protocol := corpus_dsp_fft.legacy_cfft_protocol(workload)) is not None
                and protocol.radix == 2
                for workload in workloads
            ),
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
        tuple(protocol_sources),
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
