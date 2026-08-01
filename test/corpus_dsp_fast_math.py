#!/usr/bin/env python3
"""Typed fast-math protocols for CMSIS-DSP corpus workloads."""

from __future__ import annotations

from dataclasses import dataclass

import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


@dataclass(frozen=True)
class FixedPointDivideProtocol:
    symbol: str
    test_class: str
    test_method: str
    signature: str
    scalar_type: str
    numerator: str
    denominator: str
    expected_quotient: str


_DIVIDE_PROTOCOLS = (
    FixedPointDivideProtocol(
        "arm_divide_q15",
        "FastMathQ15",
        "test_division_q15",
        "i32(i16,i16,ptr,ptr)",
        "q15_t",
        "-24576",
        "8192",
        "-24576",
    ),
    FixedPointDivideProtocol(
        "arm_divide_q31",
        "FastMathQ31",
        "test_division_q31",
        "i32(i32,i32,ptr,ptr)",
        "q31_t",
        "-1610612736",
        "536870912",
        "-1610612736",
    ),
)


def fixed_point_divide_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> FixedPointDivideProtocol | None:
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
    for protocol in _DIVIDE_PROTOCOLS:
        if (
            producer.test_class == protocol.test_class
            and producer.test_method == protocol.test_method
            and call.symbol == protocol.symbol
            and call.signature == protocol.signature
        ):
            return protocol
    return None


def render_fixed_point_divide_protocol(
    workload: corpus_inventory.ProgramWorkload, protocol_symbol: str
) -> str:
    protocol = fixed_point_divide_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no fixed-point divide provider: "
            f"{workload.identity}"
        )
    return f"""#include <cstdint>

#include "dsp/fast_math_functions.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
using Scalar = {protocol.scalar_type};
constexpr Scalar kNumerator = {protocol.numerator};
constexpr Scalar kDenominator = {protocol.denominator};
constexpr Scalar kExpectedQuotient = {protocol.expected_quotient};
constexpr std::int16_t kExpectedShift = 2;
}} // namespace

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    Scalar *quotient, std::int16_t *shift) {{
  return {protocol.symbol}(kNumerator, kDenominator, quotient, shift);
}}

int main() {{
  Scalar quotient = 0;
  std::int16_t shift = 0;
  const arm_status status = {protocol_symbol}(&quotient, &shift);
  return status == ARM_MATH_SUCCESS && quotient == kExpectedQuotient &&
                 shift == kExpectedShift
             ? 0
             : 1;
}}
"""
