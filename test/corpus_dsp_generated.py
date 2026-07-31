#!/usr/bin/env python3
"""Typed generated protocols for CMSIS-DSP census workloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


class TransformQueryKind(Enum):
    CFFT_OUTPUT = "cfft-output"
    CFFT_TEMPORARY = "cfft-temporary"
    CIFFT_OUTPUT = "cifft-output"
    MFCC_TEMPORARY = "mfcc-temporary"
    RFFT_OUTPUT = "rfft-output"
    RFFT_TEMPORARY = "rfft-temporary"
    RIFFT_INPUT = "rifft-input"


@dataclass(frozen=True)
class TransformQueryProtocol:
    kind: TransformQueryKind
    symbol: str
    signature: str

    @property
    def owner_header(self) -> str:
        return "transform_functions.h"


_TRANSFORM_QUERY_PROTOCOLS = (
    TransformQueryProtocol(
        TransformQueryKind.CFFT_OUTPUT,
        "arm_cfft_output_buffer_size",
        "i32(i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.CFFT_TEMPORARY,
        "arm_cfft_tmp_buffer_size",
        "i32(i32,i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.CIFFT_OUTPUT,
        "arm_cifft_output_buffer_size",
        "i32(i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.MFCC_TEMPORARY,
        "arm_mfcc_tmp_buffer_size",
        "i32(i32,i32,i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.RFFT_OUTPUT,
        "arm_rfft_output_buffer_size",
        "i32(i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.RFFT_TEMPORARY,
        "arm_rfft_tmp_buffer_size",
        "i32(i32,i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.RIFFT_INPUT,
        "arm_rifft_input_buffer_size",
        "i32(i32,i32,i32)",
    ),
)
_TRANSFORM_QUERY_BY_CALL = {
    (protocol.symbol, protocol.signature): protocol
    for protocol in _TRANSFORM_QUERY_PROTOCOLS
}


def transform_query_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> TransformQueryProtocol | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspGeneratedWorkloadProducer):
        return None
    if (
        workload.suite != "cmsis-dsp"
        or workload.target_profile != corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        or producer.selector_kind != "transform-query"
        or len(workload.protocol) != 1
    ):
        return None
    call = workload.protocol[0]
    protocol = _TRANSFORM_QUERY_BY_CALL.get((call.symbol, call.signature))
    if protocol is None:
        return None
    if workload.vector_identity != f"transform-query:{protocol.symbol}:0":
        return None
    return protocol


def _query_body(kind: TransformQueryKind) -> str:
    if kind in {TransformQueryKind.CFFT_OUTPUT, TransformQueryKind.CIFFT_OUTPUT}:
        symbol = (
            "arm_cfft_output_buffer_size"
            if kind is TransformQueryKind.CFFT_OUTPUT
            else "arm_cifft_output_buffer_size"
        )
        return f"""  int failures = 0;
  failures += {symbol}(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                       sample_count) != 2 * sample_count;
  failures += {symbol}(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                       sample_count + 1) != 2 * (sample_count + 1);
  return failures != 0;
"""
    if kind is TransformQueryKind.CFFT_TEMPORARY:
        return """  int failures = 0;
  failures += arm_cfft_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                       sample_count, 1) != 0;
  failures += arm_cfft_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                                       sample_count, 2) != 0;
  return failures != 0;
"""
    if kind is TransformQueryKind.RFFT_OUTPUT:
        return """  int failures = 0;
  failures += arm_rfft_output_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                          sample_count) != sample_count;
  failures += arm_rfft_output_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                                          sample_count) != 2 * sample_count;
  return failures != 0;
"""
    if kind is TransformQueryKind.RFFT_TEMPORARY:
        return """  int failures = 0;
  failures += arm_rfft_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                       sample_count, 1) != 0;
  failures += arm_rfft_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                                       sample_count, 2) != 0;
  return failures != 0;
"""
    if kind is TransformQueryKind.RIFFT_INPUT:
        return """  int failures = 0;
  failures += arm_rifft_input_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                          sample_count) != sample_count;
  failures += arm_rifft_input_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                                          sample_count) != sample_count + 2;
  return failures != 0;
"""
    if kind is TransformQueryKind.MFCC_TEMPORARY:
        return """  const std::uint32_t buf_id = 1;
  const std::uint32_t use_cfft = 1;
  int failures = 0;
  failures += arm_mfcc_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                       sample_count, buf_id,
                                       use_cfft) != 2 * sample_count;
  failures += arm_mfcc_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                       sample_count, buf_id, 0) != sample_count;
  failures += arm_mfcc_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                                       sample_count, buf_id,
                                       0) != 2 * sample_count;
  failures += arm_mfcc_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                       sample_count, 2, 0) != 0;
  return failures != 0;
"""
    raise WorkloadProviderError(f"unknown transform query kind: {kind.value}")


def render_transform_query_protocol(
    workload: corpus_inventory.ProgramWorkload,
    protocol_symbol: str,
) -> str:
    protocol = transform_query_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no transform query provider: {workload.identity}"
        )
    body = _query_body(protocol.kind)
    return f"""#include <cstdint>

#include "dsp/transform_functions.h"

#if defined(__GNUC__) || defined(__clang__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

extern "C" {{
std::uint32_t loom_corpus_sample_count = 64U;
}}

extern "C" LOOM_NOINLINE int {protocol_symbol}(
    std::uint32_t sample_count) {{
{body}}}

int main() {{
  return {protocol_symbol}(loom_corpus_sample_count);
}}
"""
