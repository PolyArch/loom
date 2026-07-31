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
    test_class: str
    test_method: str
    scalar_type: str
    bit_width: int
    input_a_pattern: str
    input_b_pattern: str
    fast_q15: bool

    @property
    def owner_header(self) -> str:
        return "matrix_functions.h"


_PROTOCOLS = (
    MatrixMultiplicationProtocol(
        "arm_mat_mult_fast_q15",
        "i32(ptr,ptr,ptr,ptr)",
        "BinaryQ15",
        "test_mat_mult_fast_q15",
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
        "test_mat_mult_q7",
        "q7_t",
        8,
        "InputA1_q7.txt",
        "InputB1_q7.txt",
        False,
    ),
)
_PROTOCOL_BY_CALL = {
    (protocol.symbol, protocol.signature): protocol for protocol in _PROTOCOLS
}


def matrix_multiplication_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> MatrixMultiplicationProtocol | None:
    producer = workload.producer
    if (
        workload.suite != "cmsis-dsp"
        or workload.target_profile != corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        or not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer)
        or producer.selector_kind != "benchmark-only"
        or len(workload.protocol) != 1
    ):
        return None
    call = workload.protocol[0]
    protocol = _PROTOCOL_BY_CALL.get((call.symbol, call.signature))
    if protocol is None:
        return None
    if (
        producer.test_class != protocol.test_class
        or producer.test_method != protocol.test_method
        or producer.vector_ordinal != 0
    ):
        return None
    return protocol


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
    patterns: Path,
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

    segments = corpus_dsp_protocol.pattern_segments(patterns)
    input_a = _decode_signed(
        corpus_dsp_protocol.require_pattern_segment(segments, protocol.input_a_pattern),
        protocol.bit_width,
        "input A",
    )
    input_b = _decode_signed(
        corpus_dsp_protocol.require_pattern_segment(segments, protocol.input_b_pattern),
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

    format_array = corpus_dsp_protocol.format_cpp_array
    return f"""#include <cstddef>
#include <cstdint>

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
  arm_matrix_instance_{"q15" if protocol.bit_width == 16 else "q7"} matrix_a{{
      kRows, kInner, const_cast<{protocol.scalar_type} *>(input_a)}};
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
