#!/usr/bin/env python3
"""Typed matrix protocols for CMSIS-DSP benchmark workloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import corpus_dsp_protocol
import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


@dataclass(frozen=True)
class MatrixMultiplicationProtocol:
    symbol: str
    signature: str
    benchmark_test_class: str
    official_test_class: str
    test_method: str
    shift_symbol: str
    scalar_type: str
    bit_width: int
    input_a_pattern: str
    input_b_pattern: str
    fast_q15: bool

    @property
    def owner_header(self) -> str:
        return "matrix_functions.h"


@dataclass(frozen=True)
class FloatingMatrixProtocol:
    symbol: str
    signature: str
    test_class: str
    test_method: str
    suffix: str
    scalar_type: str
    kind: str
    tolerance: str

    @property
    def owner_header(self) -> str:
        return (
            "matrix_functions_f16.h"
            if self.suffix == "f16"
            else "matrix_functions.h"
        )


_PROTOCOLS = (
    MatrixMultiplicationProtocol(
        "arm_mat_mult_fast_q15",
        "i32(ptr,ptr,ptr,ptr)",
        "BinaryQ15",
        "BinaryTestsQ15",
        "test_mat_mult_fast_q15",
        "arm_shift_q15",
        "q15_t",
        16,
        "InputA1_q15.txt",
        "InputB1_q15.txt",
        True,
    ),
    MatrixMultiplicationProtocol(
        "arm_mat_mult_q7",
        "i32(ptr,ptr,ptr,ptr)",
        "BinaryQ7",
        "BinaryTestsQ7",
        "test_mat_mult_q7",
        "arm_shift_q7",
        "q7_t",
        8,
        "InputA1_q7.txt",
        "InputB1_q7.txt",
        False,
    ),
)

_FLOATING_PROTOCOLS = (
    FloatingMatrixProtocol(
        "arm_mat_cmplx_mult_f16",
        "i32(ptr,ptr,ptr)",
        "BinaryTestsF16",
        "test_mat_cmplx_mult_f16",
        "f16",
        "float16_t",
        "complex-multiply",
        "6.0e-2",
    ),
    FloatingMatrixProtocol(
        "arm_mat_mult_f16",
        "i32(ptr,ptr,ptr)",
        "BinaryTestsF16",
        "test_mat_mult_f16",
        "f16",
        "float16_t",
        "multiply",
        "2.0e-2",
    ),
    FloatingMatrixProtocol(
        "arm_mat_mult_f32",
        "i32(ptr,ptr,ptr)",
        "BinaryTestsF32",
        "test_mat_mult_f32",
        "f32",
        "float32_t",
        "multiply",
        "2.0e-5",
    ),
    FloatingMatrixProtocol(
        "arm_mat_mult_f64",
        "i32(ptr,ptr,ptr)",
        "BinaryTestsF64",
        "test_mat_mult_f64",
        "f64",
        "float64_t",
        "multiply",
        "2.0e-12",
    ),
    *(
        FloatingMatrixProtocol(
            f"arm_mat_inverse_{suffix}",
            "i32(ptr,ptr)",
            f"UnaryTests{suffix.upper()}",
            f"test_mat_inverse_{suffix}",
            suffix,
            scalar_type,
            "inverse",
            tolerance,
        )
        for suffix, scalar_type, tolerance in (
            ("f16", "float16_t", "4.0e-2"),
            ("f32", "float32_t", "2.0e-4"),
            ("f64", "float64_t", "2.0e-11"),
        )
    ),
    *(
        FloatingMatrixProtocol(
            f"arm_mat_ldlt_{suffix}",
            "i32(ptr,ptr,ptr,ptr)",
            f"UnaryTests{suffix.upper()}",
            f"test_mat_ldl_{suffix}",
            suffix,
            scalar_type,
            "ldlt",
            tolerance,
        )
        for suffix, scalar_type, tolerance in (
            ("f32", "float32_t", "2.0e-5"),
            ("f64", "float64_t", "2.0e-12"),
        )
    ),
    *(
        FloatingMatrixProtocol(
            f"arm_mat_qr_{suffix}",
            signature,
            f"UnaryTests{suffix.upper()}",
            f"test_mat_qr_{suffix}",
            suffix,
            scalar_type,
            "qr",
            tolerance,
        )
        for suffix, scalar_type, signature, tolerance in (
            ("f16", "float16_t", "i32(ptr,half,ptr,ptr,ptr,ptr,ptr)", "8.0e-2"),
            ("f32", "float32_t", "i32(ptr,float,ptr,ptr,ptr,ptr,ptr)", "2.0e-4"),
            ("f64", "float64_t", "i32(ptr,double,ptr,ptr,ptr,ptr,ptr)", "2.0e-11"),
        )
    ),
)


def floating_matrix_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> FloatingMatrixProtocol | None:
    producer = workload.producer
    if (
        workload.suite != "cmsis-dsp"
        or not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer)
        or producer.selector_kind != "official"
        or producer.vector_ordinal != 0
    ):
        return None
    calls = tuple((call.symbol, call.signature) for call in workload.protocol)
    for protocol in _FLOATING_PROTOCOLS:
        expected_profile = (
            corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE
            if protocol.suffix == "f16"
            else corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        )
        if (
            workload.target_profile == expected_profile
            and producer.test_class == protocol.test_class
            and producer.test_method == protocol.test_method
            and calls == ((protocol.symbol, protocol.signature),)
        ):
            return protocol
    return None


def _cpp_values(values: tuple[float, ...]) -> str:
    return corpus_dsp_protocol.format_cpp_array(
        tuple(format(value, ".17g") for value in values)
    )


def _complex_product(
    input_a: tuple[complex, ...], input_b: tuple[complex, ...]
) -> tuple[float, ...]:
    output: list[float] = []
    for row in range(2):
        for column in range(2):
            value = sum(
                input_a[row * 2 + inner] * input_b[inner * 2 + column]
                for inner in range(2)
            )
            output.extend((value.real, value.imag))
    return tuple(output)


def _floating_matrix_prelude(protocol: FloatingMatrixProtocol) -> str:
    return f"""#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dsp/{protocol.owner_header}"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
using Scalar = {protocol.scalar_type};
using Matrix = arm_matrix_instance_{protocol.suffix};
constexpr double kTolerance = {protocol.tolerance};

bool close_enough(Scalar actual, double expected) {{
  const double difference = std::fabs(static_cast<double>(actual) - expected);
  return difference <= kTolerance * (1.0 + std::fabs(expected));
}}
}} // namespace
"""


def _render_floating_multiply(
    protocol: FloatingMatrixProtocol, protocol_symbol: str
) -> str:
    if protocol.kind == "complex-multiply":
        complex_a = (1 + 2j, 3 - 1j, 0.5 + 0.25j, -2 + 1j)
        complex_b = (2 - 1j, 1 + 0j, 0.5 + 2j, -1 + 3j)
        input_a = tuple(part for value in complex_a for part in (value.real, value.imag))
        input_b = tuple(part for value in complex_b for part in (value.real, value.imag))
        expected = _complex_product(complex_a, complex_b)
        count = 8
    else:
        input_a = (1.25, -2.0, 3.0, 0.5)
        input_b = (2.0, 1.0, -1.0, 4.0)
        expected = (4.5, -6.75, 5.5, 5.0)
        count = 4
    return _floating_matrix_prelude(protocol) + f"""
namespace {{
constexpr Scalar kInputA[] = {{
{_cpp_values(input_a)}
}};
constexpr Scalar kInputB[] = {{
{_cpp_values(input_b)}
}};
constexpr double kExpected[] = {{
{_cpp_values(expected)}
}};

bool output_matches_expected(const Scalar *output) {{
  for (std::size_t index = 0; index < {count}; ++index)
    if (!close_enough(output[index], kExpected[index]))
      return false;
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    const Scalar *input_a, const Scalar *input_b, Scalar *output) {{
  Matrix matrix_a{{2, 2, const_cast<Scalar *>(input_a)}};
  Matrix matrix_b{{2, 2, const_cast<Scalar *>(input_b)}};
  Matrix matrix_output{{2, 2, output}};
  return {protocol.symbol}(&matrix_a, &matrix_b, &matrix_output);
}}

int main() {{
  Scalar output[{count}]{{}};
  const arm_status status = {protocol_symbol}(kInputA, kInputB, output);
  return status == ARM_MATH_SUCCESS && output_matches_expected(output) ? 0 : 1;
}}
"""


def _render_floating_inverse(
    protocol: FloatingMatrixProtocol, protocol_symbol: str
) -> str:
    return _floating_matrix_prelude(protocol) + f"""
namespace {{
constexpr Scalar kInput[] = {{
{_cpp_values((4.0, 7.0, 2.0, 6.0))}
}};
constexpr double kExpected[] = {{
{_cpp_values((0.6, -0.7, -0.2, 0.4))}
}};

bool output_matches_expected(const Scalar *output) {{
  for (std::size_t index = 0; index < 4; ++index)
    if (!close_enough(output[index], kExpected[index]))
      return false;
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    Scalar *input, Scalar *output) {{
  Matrix source{{2, 2, input}};
  Matrix destination{{2, 2, output}};
  return {protocol.symbol}(&source, &destination);
}}

int main() {{
  Scalar input[4]{{kInput[0], kInput[1], kInput[2], kInput[3]}};
  Scalar output[4]{{}};
  const arm_status status = {protocol_symbol}(input, output);
  return status == ARM_MATH_SUCCESS && output_matches_expected(output) ? 0 : 1;
}}
"""


def _render_floating_ldlt(
    protocol: FloatingMatrixProtocol, protocol_symbol: str
) -> str:
    return _floating_matrix_prelude(protocol) + f"""
namespace {{
constexpr Scalar kInput[] = {{
{_cpp_values((9.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0))}
}};
constexpr double kExpectedL[] = {{
{_cpp_values((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))}
}};
constexpr double kExpectedD[] = {{
{_cpp_values((9.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0))}
}};

bool output_matches_expected(const Scalar *lower, const Scalar *diagonal,
                             const std::uint16_t *permutation) {{
  for (std::size_t index = 0; index < 9; ++index)
    if (!close_enough(lower[index], kExpectedL[index]) ||
        !close_enough(diagonal[index], kExpectedD[index]))
      return false;
  for (std::uint16_t index = 0; index < 3; ++index)
    if (permutation[index] != index)
      return false;
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    const Scalar *input, Scalar *lower, Scalar *diagonal,
    std::uint16_t *permutation) {{
  Matrix source{{3, 3, const_cast<Scalar *>(input)}};
  Matrix lower_matrix{{3, 3, lower}};
  Matrix diagonal_matrix{{3, 3, diagonal}};
  return {protocol.symbol}(
      &source, &lower_matrix, &diagonal_matrix, permutation);
}}

int main() {{
  Scalar lower[9]{{}};
  Scalar diagonal[9]{{}};
  std::uint16_t permutation[3]{{}};
  const arm_status status =
      {protocol_symbol}(kInput, lower, diagonal, permutation);
  return status == ARM_MATH_SUCCESS &&
                 output_matches_expected(lower, diagonal, permutation)
             ? 0
             : 1;
}}
"""


def _render_floating_qr(
    protocol: FloatingMatrixProtocol, protocol_symbol: str
) -> str:
    return _floating_matrix_prelude(protocol) + f"""
namespace {{
constexpr std::uint16_t kRows = 2;
constexpr std::uint16_t kColumns = 2;
constexpr Scalar kInput[] = {{
{_cpp_values((12.0, -51.0, 6.0, 167.0))}
}};

bool output_matches_expected(const Scalar *upper, const Scalar *orthogonal) {{
  for (std::size_t row = 0; row < kRows; ++row) {{
    for (std::size_t column = 0; column < kColumns; ++column) {{
      double reconstructed = 0.0;
      for (std::size_t inner = 0; inner < kColumns; ++inner) {{
        const double r = inner <= column
                             ? static_cast<double>(upper[inner * kColumns + column])
                             : 0.0;
        reconstructed +=
            static_cast<double>(orthogonal[row * kRows + inner]) * r;
      }}
      if (std::fabs(reconstructed -
                    static_cast<double>(kInput[row * kColumns + column])) >
          kTolerance *
              (1.0 + std::fabs(static_cast<double>(
                         kInput[row * kColumns + column]))))
        return false;
    }}
  }}
  for (std::size_t left = 0; left < kRows; ++left) {{
    for (std::size_t right = 0; right < kRows; ++right) {{
      double dot = 0.0;
      for (std::size_t row = 0; row < kRows; ++row)
        dot += static_cast<double>(orthogonal[row * kRows + left]) *
               static_cast<double>(orthogonal[row * kRows + right]);
      const double expected = left == right ? 1.0 : 0.0;
      if (std::fabs(dot - expected) > 4.0 * kTolerance)
        return false;
    }}
  }}
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    Scalar *input, Scalar *upper, Scalar *orthogonal, Scalar *tau,
    Scalar *temporary_a, Scalar *temporary_b) {{
  Matrix source{{kRows, kColumns, input}};
  Matrix upper_matrix{{kRows, kColumns, upper}};
  Matrix orthogonal_matrix{{kRows, kRows, orthogonal}};
  return {protocol.symbol}(
      &source, static_cast<Scalar>(1.0e-4), &upper_matrix,
      &orthogonal_matrix, tau, temporary_a, temporary_b);
}}

int main() {{
  Scalar input[4]{{kInput[0], kInput[1], kInput[2], kInput[3]}};
  Scalar upper[4]{{}};
  Scalar orthogonal[4]{{}};
  Scalar tau[2]{{}};
  Scalar temporary_a[2]{{}};
  Scalar temporary_b[2]{{}};
  const arm_status status = {protocol_symbol}(
      input, upper, orthogonal, tau, temporary_a, temporary_b);
  return status == ARM_MATH_SUCCESS &&
                 output_matches_expected(upper, orthogonal)
             ? 0
             : 1;
}}
"""


def render_floating_matrix_protocol(
    workload: corpus_inventory.ProgramWorkload, protocol_symbol: str
) -> str:
    protocol = floating_matrix_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no floating matrix provider: {workload.identity}"
        )
    if protocol.kind in {"multiply", "complex-multiply"}:
        return _render_floating_multiply(protocol, protocol_symbol)
    if protocol.kind == "inverse":
        return _render_floating_inverse(protocol, protocol_symbol)
    if protocol.kind == "ldlt":
        return _render_floating_ldlt(protocol, protocol_symbol)
    if protocol.kind == "qr":
        return _render_floating_qr(protocol, protocol_symbol)
    raise WorkloadProviderError(
        f"CMSIS-DSP matrix protocol has an unknown kind: {protocol.kind}"
    )


def matrix_multiplication_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> MatrixMultiplicationProtocol | None:
    producer = workload.producer
    if (
        workload.suite != "cmsis-dsp"
        or workload.target_profile != corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        or not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer)
    ):
        return None
    calls = tuple((call.symbol, call.signature) for call in workload.protocol)
    for protocol in _PROTOCOLS:
        benchmark_calls = ((protocol.symbol, protocol.signature),)
        official_calls = (
            (protocol.shift_symbol, "void(ptr,i8,ptr,i32)"),
            (protocol.symbol, protocol.signature),
        )
        expected_class = (
            protocol.benchmark_test_class
            if producer.selector_kind == "benchmark-only" and calls == benchmark_calls
            else protocol.official_test_class
            if producer.selector_kind == "official" and calls == official_calls
            else None
        )
        if (
            expected_class is not None
            and producer.test_class == expected_class
            and producer.test_method == protocol.test_method
            and producer.vector_ordinal == 0
        ):
            return protocol
    return None


def _decode_signed(raw: bytes, bit_width: int, name: str) -> tuple[int, ...]:
    byte_width = bit_width // 8
    if bit_width not in {8, 16} or len(raw) % byte_width != 0:
        raise WorkloadProviderError(
            f"CMSIS-DSP matrix pattern {name} has an invalid scalar width"
        )
    return tuple(
        int.from_bytes(
            raw[offset : offset + byte_width], byteorder="little", signed=True
        )
        for offset in range(0, len(raw), byte_width)
    )


def _matrix_product(
    input_a: tuple[int, ...],
    input_b: tuple[int, ...],
    rows: int,
    inner: int,
    columns: int,
    protocol: MatrixMultiplicationProtocol,
) -> tuple[int, ...]:
    output = []
    for row in range(rows):
        for column in range(columns):
            if protocol.fast_q15:
                accumulator = 0
                for index in range(inner):
                    product = (
                        input_a[row * inner + index] * input_b[index * columns + column]
                    )
                    accumulator = (accumulator + product) & 0xFFFFFFFF
                signed = (
                    accumulator
                    if accumulator < 0x80000000
                    else accumulator - 0x100000000
                )
                narrowed = (signed >> 15) & 0xFFFF
                output.append(narrowed if narrowed < 0x8000 else narrowed - 0x10000)
                continue

            accumulator = sum(
                input_a[row * inner + index] * input_b[index * columns + column]
                for index in range(inner)
            )
            shifted = accumulator >> 7
            output.append(max(-128, min(127, shifted)))
    return tuple(output)


def render_matrix_multiplication_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path | None,
    dimensions: tuple[int, int, int],
    protocol_symbol: str,
) -> str:
    protocol = matrix_multiplication_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no matrix provider: {workload.identity}"
        )
    rows, inner, columns = dimensions
    if min(dimensions) <= 0:
        raise WorkloadProviderError("CMSIS-DSP matrix dimensions must be positive")

    if patterns is None:
        input_a = (
            (16384, -8192, 4096, 12288)
            if protocol.bit_width == 16
            else (64, -32, 16, 48)
        )
        input_b = (
            (8192, 4096, -2048, 16384)
            if protocol.bit_width == 16
            else (32, 16, -8, 64)
        )
    else:
        segments = corpus_dsp_protocol.pattern_segments(patterns)
        input_a = _decode_signed(
            corpus_dsp_protocol.require_pattern_segment(
                segments, protocol.input_a_pattern
            ),
            protocol.bit_width,
            "input A",
        )
        input_b = _decode_signed(
            corpus_dsp_protocol.require_pattern_segment(
                segments, protocol.input_b_pattern
            ),
            protocol.bit_width,
            "input B",
        )
        input_a = input_a[: rows * inner]
        input_b = input_b[: inner * columns]
    if len(input_a) != rows * inner or len(input_b) != inner * columns:
        raise WorkloadProviderError(
            "CMSIS-DSP matrix pattern is smaller than its selected dimensions"
        )
    expected = _matrix_product(input_a, input_b, rows, inner, columns, protocol)
    use_shift = len(workload.protocol) == 2
    shifted_declaration = (
        f"  {protocol.scalar_type} shifted_a[kRows * kInner]{{}};\n"
        if use_shift
        else ""
    )
    shift_call = (
        f"  {protocol.shift_symbol}(input_a, 0, shifted_a, kRows * kInner);\n"
        if use_shift
        else ""
    )
    matrix_input = "shifted_a" if use_shift else "input_a"

    format_array = corpus_dsp_protocol.format_cpp_array
    return f"""#include <cstddef>
#include <cstdint>

#include "dsp/basic_math_functions.h"
#include "dsp/{protocol.owner_header}"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::uint16_t kRows = {rows};
constexpr std::uint16_t kInner = {inner};
constexpr std::uint16_t kColumns = {columns};
constexpr std::size_t kOutputCount = kRows * kColumns;
constexpr {protocol.scalar_type} kInputA[] = {{
{format_array(tuple(str(value) for value in input_a))}
}};
constexpr {protocol.scalar_type} kInputB[] = {{
{format_array(tuple(str(value) for value in input_b))}
}};
constexpr {protocol.scalar_type} kExpected[] = {{
{format_array(tuple(str(value) for value in expected))}
}};

bool output_matches_expected(const {protocol.scalar_type} *output) {{
  for (std::size_t index = 0; index < kOutputCount; ++index) {{
    if (output[index] != kExpected[index]) {{
      return false;
    }}
  }}
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    const {protocol.scalar_type} *input_a,
    const {protocol.scalar_type} *input_b,
    {protocol.scalar_type} *output,
    {protocol.scalar_type} *scratch) {{
{shifted_declaration}{shift_call}  const {protocol.scalar_type} *matrix_input = {matrix_input};
  arm_matrix_instance_{"q15" if protocol.bit_width == 16 else "q7"} matrix_a{{
      kRows, kInner, const_cast<{protocol.scalar_type} *>(matrix_input)}};
  arm_matrix_instance_{"q15" if protocol.bit_width == 16 else "q7"} matrix_b{{
      kInner, kColumns, const_cast<{protocol.scalar_type} *>(input_b)}};
  arm_matrix_instance_{"q15" if protocol.bit_width == 16 else "q7"} matrix_output{{
      kRows, kColumns, output}};
  return {protocol.symbol}(&matrix_a, &matrix_b, &matrix_output, scratch);
}}

int main() {{
  {protocol.scalar_type} output[kOutputCount]{{}};
  {protocol.scalar_type} scratch[kInner * kColumns]{{}};
  const arm_status status = {protocol_symbol}(
      kInputA, kInputB, output, scratch);
  return status == ARM_MATH_SUCCESS && output_matches_expected(output) ? 0 : 1;
}}
"""
