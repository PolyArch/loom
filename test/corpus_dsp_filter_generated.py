#!/usr/bin/env python3
"""Typed generated protocols for CMSIS-DSP filtering variants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


class SequenceOperation(Enum):
    CONVOLUTION = "convolution"
    CORRELATION = "correlation"


@dataclass(frozen=True)
class SequenceProtocol:
    operation: SequenceOperation
    symbol: str
    signature: str
    baseline_symbol: str
    value_type: str
    input_a: tuple[int, ...]
    input_b: tuple[int, ...]
    scratch_count: int

    @property
    def owner_header(self) -> str:
        return "filtering_functions.h"


_Q15_A = (16384, -8192, 4096, 12288, -4096)
_Q15_B = (8192, 4096, -2048)
_Q31_A = (268435456, -134217728, 67108864, 201326592, -67108864)
_Q31_B = (134217728, 67108864, -33554432)
_Q7_A = (64, -32, 16, 48, -16)
_Q7_B = (32, 16, -8)


def _sequence_protocols() -> tuple[SequenceProtocol, ...]:
    return (
        SequenceProtocol(
            SequenceOperation.CONVOLUTION,
            "arm_conv_fast_opt_q15",
            "void(ptr,i32,ptr,i32,ptr,ptr,ptr)",
            "arm_conv_q15",
            "q15_t",
            _Q15_A,
            _Q15_B,
            2,
        ),
        SequenceProtocol(
            SequenceOperation.CONVOLUTION,
            "arm_conv_fast_q15",
            "void(ptr,i32,ptr,i32,ptr)",
            "arm_conv_q15",
            "q15_t",
            _Q15_A,
            _Q15_B,
            0,
        ),
        SequenceProtocol(
            SequenceOperation.CONVOLUTION,
            "arm_conv_fast_q31",
            "void(ptr,i32,ptr,i32,ptr)",
            "arm_conv_q31",
            "q31_t",
            _Q31_A,
            _Q31_B,
            0,
        ),
        SequenceProtocol(
            SequenceOperation.CONVOLUTION,
            "arm_conv_opt_q15",
            "void(ptr,i32,ptr,i32,ptr,ptr,ptr)",
            "arm_conv_q15",
            "q15_t",
            _Q15_A,
            _Q15_B,
            2,
        ),
        SequenceProtocol(
            SequenceOperation.CONVOLUTION,
            "arm_conv_opt_q7",
            "void(ptr,i32,ptr,i32,ptr,ptr,ptr)",
            "arm_conv_q7",
            "q7_t",
            _Q7_A,
            _Q7_B,
            2,
        ),
        SequenceProtocol(
            SequenceOperation.CORRELATION,
            "arm_correlate_fast_opt_q15",
            "void(ptr,i32,ptr,i32,ptr,ptr)",
            "arm_correlate_q15",
            "q15_t",
            _Q15_A,
            _Q15_B,
            1,
        ),
        SequenceProtocol(
            SequenceOperation.CORRELATION,
            "arm_correlate_fast_q15",
            "void(ptr,i32,ptr,i32,ptr)",
            "arm_correlate_q15",
            "q15_t",
            _Q15_A,
            _Q15_B,
            0,
        ),
        SequenceProtocol(
            SequenceOperation.CORRELATION,
            "arm_correlate_fast_q31",
            "void(ptr,i32,ptr,i32,ptr)",
            "arm_correlate_q31",
            "q31_t",
            _Q31_A,
            _Q31_B,
            0,
        ),
        SequenceProtocol(
            SequenceOperation.CORRELATION,
            "arm_correlate_opt_q15",
            "void(ptr,i32,ptr,i32,ptr,ptr)",
            "arm_correlate_q15",
            "q15_t",
            _Q15_A,
            _Q15_B,
            1,
        ),
        SequenceProtocol(
            SequenceOperation.CORRELATION,
            "arm_correlate_opt_q7",
            "void(ptr,i32,ptr,i32,ptr,ptr,ptr)",
            "arm_correlate_q7",
            "q7_t",
            _Q7_A,
            _Q7_B,
            2,
        ),
    )


_SEQUENCE_BY_CALL = {
    (protocol.symbol, protocol.signature): protocol
    for protocol in _sequence_protocols()
}


def sequence_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> SequenceProtocol | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspGeneratedWorkloadProducer):
        return None
    if (
        workload.suite != "cmsis-dsp"
        or workload.target_profile != corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        or producer.selector_kind != "filter-completion"
        or len(workload.protocol) != 1
    ):
        return None
    call = workload.protocol[0]
    protocol = _SEQUENCE_BY_CALL.get((call.symbol, call.signature))
    if protocol is None:
        return None
    if workload.vector_identity != f"filter-completion:{protocol.symbol}:0":
        return None
    return protocol


def _array(values: tuple[int, ...]) -> str:
    return ", ".join(str(value) for value in values)


def render_sequence_protocol(
    workload: corpus_inventory.ProgramWorkload,
    protocol_symbol: str,
) -> str:
    protocol = sequence_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no sequence provider: {workload.identity}"
        )
    output_count = (
        len(protocol.input_a) + len(protocol.input_b) - 1
        if protocol.operation is SequenceOperation.CONVOLUTION
        else 2 * max(len(protocol.input_a), len(protocol.input_b)) - 1
    )
    scratch_declarations = ""
    scratch_arguments = ""
    if protocol.scratch_count >= 1:
        scratch_declarations += "  q15_t scratch1[9];\n"
        scratch_arguments += ", scratch1"
    if protocol.scratch_count == 2:
        scratch_declarations += "  q15_t scratch2[3];\n"
        scratch_arguments += ", scratch2"

    return f"""#include <cstddef>
#include <cstdint>

#include "dsp/filtering_functions.h"

#if defined(__GNUC__) || defined(__clang__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const {protocol.value_type} *input_a, std::uint32_t input_a_count,
    const {protocol.value_type} *input_b, std::uint32_t input_b_count,
    {protocol.value_type} *output) {{
{scratch_declarations}  {protocol.symbol}(input_a, input_a_count, input_b, input_b_count,
                    output{scratch_arguments});
}}

int main() {{
  const {protocol.value_type} input_a[] = {{{_array(protocol.input_a)}}};
  const {protocol.value_type} input_b[] = {{{_array(protocol.input_b)}}};
  {protocol.value_type} output[{output_count}]{{}};
  {protocol.value_type} reference[{output_count}]{{}};
  {protocol.baseline_symbol}(input_a, {len(protocol.input_a)}, input_b,
                             {len(protocol.input_b)}, reference);
  {protocol_symbol}(input_a, {len(protocol.input_a)}, input_b,
                    {len(protocol.input_b)}, output);
  const auto output_matches_reference = [&]() {{
    for (std::size_t index = 0; index < {output_count}; ++index) {{
      if (output[index] != reference[index]) {{
        return false;
      }}
    }}
    return true;
  }};
  return output_matches_reference() ? 0 : 1;
}}
"""
