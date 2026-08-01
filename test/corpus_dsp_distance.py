#!/usr/bin/env python3
"""Typed distance protocols for CMSIS-DSP corpus workloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import corpus_inventory
import corpus_dsp_protocol
from corpus_workload_errors import WorkloadProviderError


class DistanceKind(Enum):
    CORRELATION = "correlation"
    COSINE = "cosine"


class DistanceScalarKind(Enum):
    F16 = "f16"
    F32 = "f32"
    F64 = "f64"


@dataclass(frozen=True)
class DistanceProtocol:
    symbol: str
    test_class: str
    test_method: str
    signature: str
    kind: DistanceKind
    scalar_kind: DistanceScalarKind

    @property
    def scalar_type(self) -> str:
        return {
            DistanceScalarKind.F16: "float16_t",
            DistanceScalarKind.F32: "float32_t",
            DistanceScalarKind.F64: "float64_t",
        }[self.scalar_kind]

    @property
    def target_profile(self) -> str:
        if self.scalar_kind is DistanceScalarKind.F16:
            return corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE
        return corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE

    @property
    def owner_header(self) -> str:
        if self.scalar_kind is DistanceScalarKind.F16:
            return "distance_functions_f16.h"
        return "distance_functions.h"


_DISTANCE_PROTOCOLS = (
    DistanceProtocol(
        "arm_correlation_distance_f16",
        "DistanceTestsF16",
        "test_correlation_distance_f16",
        "half(ptr,ptr,i32)",
        DistanceKind.CORRELATION,
        DistanceScalarKind.F16,
    ),
    DistanceProtocol(
        "arm_correlation_distance_f32",
        "DistanceTestsF32",
        "test_correlation_distance_f32",
        "float(ptr,ptr,i32)",
        DistanceKind.CORRELATION,
        DistanceScalarKind.F32,
    ),
    DistanceProtocol(
        "arm_cosine_distance_f16",
        "DistanceTestsF16",
        "test_cosine_distance_f16",
        "half(ptr,ptr,i32)",
        DistanceKind.COSINE,
        DistanceScalarKind.F16,
    ),
    DistanceProtocol(
        "arm_cosine_distance_f32",
        "DistanceTestsF32",
        "test_cosine_distance_f32",
        "float(ptr,ptr,i32)",
        DistanceKind.COSINE,
        DistanceScalarKind.F32,
    ),
    DistanceProtocol(
        "arm_cosine_distance_f64",
        "DistanceTestsF64",
        "test_cosine_distance_f64",
        "double(ptr,ptr,i32)",
        DistanceKind.COSINE,
        DistanceScalarKind.F64,
    ),
)


def distance_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> DistanceProtocol | None:
    producer = workload.producer
    if (
        workload.suite != "cmsis-dsp"
        or not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer)
        or producer.selector_kind != "official"
        or producer.vector_ordinal != 0
        or len(workload.protocol) != 1
    ):
        return None
    call = workload.protocol[0]
    for protocol in _DISTANCE_PROTOCOLS:
        if (
            workload.target_profile == protocol.target_profile
            and producer.test_class == protocol.test_class
            and producer.test_method == protocol.test_method
            and call.symbol == protocol.symbol
            and call.signature == protocol.signature
        ):
            return protocol
    return None


def _values(
    protocol: DistanceProtocol,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    suffix = "" if protocol.scalar_kind is DistanceScalarKind.F64 else "f"
    if protocol.kind is DistanceKind.CORRELATION:
        return (
            (f"1.0{suffix}", f"-1.0{suffix}"),
            (f"1.0{suffix}", f"-1.0{suffix}"),
            f"0.0{suffix}",
        )
    if protocol.kind is DistanceKind.COSINE:
        return (
            (f"1.0{suffix}", f"0.0{suffix}"),
            (f"0.0{suffix}", f"1.0{suffix}"),
            f"1.0{suffix}",
        )
    raise WorkloadProviderError("unknown distance protocol kind")


def render_distance_protocol(
    workload: corpus_inventory.ProgramWorkload, protocol_symbol: str
) -> str:
    protocol = distance_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no distance provider: {workload.identity}"
        )
    input_a, input_b, expected = _values(protocol)
    input_a_values = corpus_dsp_protocol.format_cpp_array(input_a)
    input_b_values = corpus_dsp_protocol.format_cpp_array(input_b)
    return f"""#include <cstddef>

#include "dsp/{protocol.owner_header}"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
using Scalar = {protocol.scalar_type};
constexpr std::size_t kCount = 2;
constexpr Scalar kInputA[kCount] = {{
{input_a_values}
}};
constexpr Scalar kInputB[kCount] = {{
{input_b_values}
}};
constexpr Scalar kExpected = {expected};

bool output_matches_expected(Scalar output) {{ return output == kExpected; }}
}} // namespace

extern "C" LOOM_NOINLINE Scalar {protocol_symbol}(
    Scalar *input_a, Scalar *input_b) {{
  return {protocol.symbol}(input_a, input_b, kCount);
}}

int main() {{
  Scalar input_a[kCount];
  Scalar input_b[kCount];
  for (std::size_t index = 0; index < kCount; ++index) {{
    input_a[index] = kInputA[index];
    input_b[index] = kInputB[index];
  }}
  const Scalar output = {protocol_symbol}(input_a, input_b);
  return output_matches_expected(output) ? 0 : 1;
}}
"""
