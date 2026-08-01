#!/usr/bin/env python3
"""Typed convolution protocols for CMSIS-DSP corpus workloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import corpus_inventory
import corpus_dsp_protocol
from corpus_workload_errors import WorkloadProviderError


class ConvolutionScalarKind(Enum):
    F32 = "f32"
    Q7 = "q7"
    Q15 = "q15"
    Q31 = "q31"


@dataclass(frozen=True)
class PartialConvolutionProtocol:
    symbol: str
    test_class: str
    test_method: str
    signature: str
    scalar_kind: ConvolutionScalarKind
    uses_scratch: bool = False

    @property
    def scalar_type(self) -> str:
        return {
            ConvolutionScalarKind.F32: "float32_t",
            ConvolutionScalarKind.Q7: "q7_t",
            ConvolutionScalarKind.Q15: "q15_t",
            ConvolutionScalarKind.Q31: "q31_t",
        }[self.scalar_kind]


_PARTIAL_PROTOCOLS = (
    PartialConvolutionProtocol(
        "arm_conv_partial_f32",
        "MISCF32",
        "test_conv_partial_f32",
        "i32(ptr,i32,ptr,i32,ptr,i32,i32)",
        ConvolutionScalarKind.F32,
    ),
    PartialConvolutionProtocol(
        "arm_conv_partial_fast_opt_q15",
        "MISCQ15",
        "test_conv_partial_fast_opt_q15",
        "i32(ptr,i32,ptr,i32,ptr,i32,i32,ptr,ptr)",
        ConvolutionScalarKind.Q15,
        uses_scratch=True,
    ),
    PartialConvolutionProtocol(
        "arm_conv_partial_fast_q15",
        "MISCQ15",
        "test_conv_partial_fast_q15",
        "i32(ptr,i32,ptr,i32,ptr,i32,i32)",
        ConvolutionScalarKind.Q15,
    ),
    PartialConvolutionProtocol(
        "arm_conv_partial_fast_q31",
        "MISCQ31",
        "test_conv_partial_fast_q31",
        "i32(ptr,i32,ptr,i32,ptr,i32,i32)",
        ConvolutionScalarKind.Q31,
    ),
    PartialConvolutionProtocol(
        "arm_conv_partial_opt_q15",
        "MISCQ15",
        "test_conv_partial_opt_q15",
        "i32(ptr,i32,ptr,i32,ptr,i32,i32,ptr,ptr)",
        ConvolutionScalarKind.Q15,
        uses_scratch=True,
    ),
    PartialConvolutionProtocol(
        "arm_conv_partial_opt_q7",
        "MISCQ7",
        "test_conv_partial_opt_q7",
        "i32(ptr,i32,ptr,i32,ptr,i32,i32,ptr,ptr)",
        ConvolutionScalarKind.Q7,
        uses_scratch=True,
    ),
    PartialConvolutionProtocol(
        "arm_conv_partial_q15",
        "MISCQ15",
        "test_conv_partial_q15",
        "i32(ptr,i32,ptr,i32,ptr,i32,i32)",
        ConvolutionScalarKind.Q15,
    ),
    PartialConvolutionProtocol(
        "arm_conv_partial_q31",
        "MISCQ31",
        "test_conv_partial_q31",
        "i32(ptr,i32,ptr,i32,ptr,i32,i32)",
        ConvolutionScalarKind.Q31,
    ),
    PartialConvolutionProtocol(
        "arm_conv_partial_q7",
        "MISCQ7",
        "test_conv_partial_q7",
        "i32(ptr,i32,ptr,i32,ptr,i32,i32)",
        ConvolutionScalarKind.Q7,
    ),
)


def partial_convolution_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> PartialConvolutionProtocol | None:
    producer = workload.producer
    if (
        workload.suite != "cmsis-dsp"
        or workload.target_profile != corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        or not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer)
        or producer.selector_kind != "official"
        or producer.vector_ordinal != 0
        or len(workload.protocol) != 1
    ):
        return None
    call = workload.protocol[0]
    for protocol in _PARTIAL_PROTOCOLS:
        if (
            producer.test_class == protocol.test_class
            and producer.test_method == protocol.test_method
            and call.symbol == protocol.symbol
            and call.signature == protocol.signature
        ):
            return protocol
    return None


def _values(
    scalar_kind: ConvolutionScalarKind,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if scalar_kind is ConvolutionScalarKind.F32:
        return (
            ("0.25f", "-0.5f", "0.125f", "0.375f"),
            ("0.5f", "0.0f", "0.0f"),
            ("-0.25f", "0.0625f", "0.1875f"),
        )
    if scalar_kind is ConvolutionScalarKind.Q7:
        return (
            ("32", "-64", "16", "48"),
            ("64", "0", "0"),
            ("-32", "8", "24"),
        )
    if scalar_kind is ConvolutionScalarKind.Q15:
        return (
            ("8192", "-16384", "4096", "12288"),
            ("16384", "0", "0"),
            ("-8192", "2048", "6144"),
        )
    if scalar_kind is ConvolutionScalarKind.Q31:
        return (
            ("536870912", "-1073741824", "268435456", "805306368"),
            ("1073741824", "0", "0"),
            ("-536870912", "134217728", "402653184"),
        )
    raise WorkloadProviderError("unknown partial convolution scalar kind")


def render_partial_convolution_protocol(
    workload: corpus_inventory.ProgramWorkload, protocol_symbol: str
) -> str:
    protocol = partial_convolution_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no partial convolution provider: "
            f"{workload.identity}"
        )
    input_a, input_b, expected = _values(protocol.scalar_kind)
    input_a_values = corpus_dsp_protocol.format_cpp_array(input_a)
    input_b_values = corpus_dsp_protocol.format_cpp_array(input_b)
    expected_values = corpus_dsp_protocol.format_cpp_array(expected)
    scratch_arguments = ", scratch1, scratch2" if protocol.uses_scratch else ""
    return f"""#include <cstddef>

#include "dsp/filtering_functions.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
using Scalar = {protocol.scalar_type};
constexpr std::size_t kInputACount = 4;
constexpr std::size_t kInputBCount = 3;
constexpr std::size_t kOutputCount = kInputACount + kInputBCount - 1;
constexpr std::size_t kFirstOutput = 1;
constexpr std::size_t kPartialCount = 3;
constexpr Scalar kInputA[kInputACount] = {{
{input_a_values}
}};
constexpr Scalar kInputB[kInputBCount] = {{
{input_b_values}
}};
constexpr Scalar kExpected[kPartialCount] = {{
{expected_values}
}};

bool output_matches_expected(const Scalar *output) {{
  for (std::size_t index = 0; index < kPartialCount; ++index)
    if (output[kFirstOutput + index] != kExpected[index])
      return false;
  return true;
}}
}} // namespace

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    const Scalar *input_a, const Scalar *input_b, Scalar *output,
    q15_t *scratch1, q15_t *scratch2) {{
  return {protocol.symbol}(input_a, kInputACount, input_b, kInputBCount,
                           output, kFirstOutput, kPartialCount{scratch_arguments});
}}

int main() {{
  Scalar output[kOutputCount]{{}};
  q15_t scratch1[kInputACount + 2 * kInputBCount - 2]{{}};
  q15_t scratch2[kInputBCount]{{}};
  const arm_status status =
      {protocol_symbol}(kInputA, kInputB, output, scratch1, scratch2);
  return status == ARM_MATH_SUCCESS && output_matches_expected(output) ? 0 : 1;
}}
"""
