#!/usr/bin/env python3
"""Typed PID protocols for CMSIS-DSP benchmark workloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import corpus_dsp_protocol
import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


@dataclass(frozen=True)
class PidProtocol:
    init_symbol: str
    process_symbol: str
    init_signature: str
    process_signature: str
    test_class: str
    test_method: str
    scalar_type: str
    pattern_name: str
    kind: str

    @property
    def owner_header(self) -> str:
        return "controller_functions.h"


_PROTOCOLS = (
    PidProtocol(
        "arm_pid_init_f32",
        "arm_pid_f32",
        "void(ptr,i32)",
        "float(ptr,float)",
        "ControllerF32",
        "test_pid_f32",
        "float32_t",
        "Samples1_f32.txt",
        "f32",
    ),
    PidProtocol(
        "arm_pid_init_q15",
        "arm_pid_q15",
        "void(ptr,i32)",
        "i16(ptr,i16)",
        "ControllerQ15",
        "test_pid_q15",
        "q15_t",
        "Samples1_q15.txt",
        "q15",
    ),
    PidProtocol(
        "arm_pid_init_q31",
        "arm_pid_q31",
        "void(ptr,i32)",
        "i32(ptr,i32)",
        "ControllerQ31",
        "test_pid_q31",
        "q31_t",
        "Samples1_q31.txt",
        "q31",
    ),
)
_PROTOCOL_BY_CALLS = {
    (
        (protocol.init_symbol, protocol.init_signature),
        (protocol.process_symbol, protocol.process_signature),
    ): protocol
    for protocol in _PROTOCOLS
}


def pid_protocol(workload: corpus_inventory.ProgramWorkload) -> PidProtocol | None:
    producer = workload.producer
    if (
        workload.suite != "cmsis-dsp"
        or workload.target_profile != corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        or not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer)
        or producer.selector_kind != "benchmark-only"
        or len(workload.protocol) != 2
    ):
        return None
    calls = tuple((call.symbol, call.signature) for call in workload.protocol)
    protocol = _PROTOCOL_BY_CALLS.get(calls)
    if protocol is None:
        return None
    if (
        producer.test_class != protocol.test_class
        or producer.test_method != protocol.test_method
        or producer.vector_ordinal != 0
    ):
        return None
    return protocol


def _reference_body(kind: str) -> str:
    if kind == "f32":
        return """  float32_t state0 = 0.0f;
  float32_t state1 = 0.0f;
  float32_t state2 = 0.0f;
  for (std::size_t index = 0; index < kSampleCount; ++index) {
    const float32_t expected =
        (0.4375f * kInput[index]) + (-0.375f * state0) +
        (0.0625f * state1) + state2;
    const float32_t error = std::fabs(output[index] - expected);
    if (error > 1.0e-6f * (1.0f + std::fabs(expected))) {
      return false;
    }
    state1 = state0;
    state0 = kInput[index];
    state2 = expected;
  }
  return true;
"""
    if kind == "q15":
        return """  std::int32_t state0 = 0;
  std::int32_t state1 = 0;
  std::int32_t state2 = 0;
  for (std::size_t index = 0; index < kSampleCount; ++index) {
    const std::int64_t accumulator =
        14336LL * kInput[index] - 12288LL * state0 +
        2048LL * state1 + static_cast<std::int64_t>(state2) * 32768LL;
    const std::int64_t shifted = arithmetic_shift(accumulator, 15);
    const q15_t expected = static_cast<q15_t>(
        shifted < -32768 ? -32768 : shifted > 32767 ? 32767 : shifted);
    if (output[index] != expected) {
      return false;
    }
    state1 = state0;
    state0 = kInput[index];
    state2 = expected;
  }
  return true;
"""
    if kind == "q31":
        return """  std::int32_t state0 = 0;
  std::int32_t state1 = 0;
  std::int32_t state2 = 0;
  for (std::size_t index = 0; index < kSampleCount; ++index) {
    const std::int64_t accumulator =
        939524096LL * kInput[index] - 805306368LL * state0 +
        134217728LL * state1;
    const std::int64_t shifted = arithmetic_shift(accumulator, 31);
    const std::uint32_t bits = static_cast<std::uint32_t>(shifted) +
                               static_cast<std::uint32_t>(state2);
    const q31_t expected = signed_from_bits(bits);
    if (output[index] != expected) {
      return false;
    }
    state1 = state0;
    state0 = kInput[index];
    state2 = expected;
  }
  return true;
"""
    raise WorkloadProviderError(f"unknown PID protocol kind: {kind}")


def _coefficients(kind: str) -> tuple[str, str, str]:
    if kind == "f32":
        return "0.25f", "0.125f", "0.0625f"
    if kind == "q15":
        return "8192", "4096", "2048"
    if kind == "q31":
        return "536870912", "268435456", "134217728"
    raise WorkloadProviderError(f"unknown PID protocol kind: {kind}")


def render_pid_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path,
    sample_count: int,
    protocol_symbol: str,
) -> str:
    protocol = pid_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no PID provider: {workload.identity}"
        )
    if sample_count <= 0:
        raise WorkloadProviderError("CMSIS-DSP PID sample count must be positive")

    segments = corpus_dsp_protocol.pattern_segments(patterns)
    raw_input = corpus_dsp_protocol.require_pattern_segment(
        segments, protocol.pattern_name
    )
    if protocol.kind == "f32":
        values = corpus_dsp_protocol.decode_f32_pattern(raw_input, "PID input")
    elif protocol.kind == "q15":
        values = corpus_dsp_protocol.decode_integer_pattern(
            raw_input, 16, True, "PID input"
        )
    else:
        values = corpus_dsp_protocol.decode_integer_pattern(
            raw_input, 32, True, "PID input"
        )
    values = values[:sample_count]
    if len(values) != sample_count:
        raise WorkloadProviderError(
            "CMSIS-DSP PID pattern is smaller than its selected sample count"
        )

    kp, ki, kd = _coefficients(protocol.kind)
    instance_type = f"arm_pid_instance_{protocol.kind}"
    support = (
        ""
        if protocol.kind == "f32"
        else """
std::int64_t arithmetic_shift(std::int64_t value, unsigned amount) {
  if (value >= 0) {
    return value >> amount;
  }
  const std::int64_t scale = std::int64_t{1} << amount;
  return -((-value + scale - 1) / scale);
}
"""
    )
    if protocol.kind == "q31":
        support += """
q31_t signed_from_bits(std::uint32_t bits) {
  if (bits <= 0x7fffffffU) {
    return static_cast<q31_t>(bits);
  }
  return static_cast<q31_t>(static_cast<std::int64_t>(bits) - 0x100000000LL);
}
"""

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
constexpr std::size_t kSampleCount = {sample_count};
constexpr {protocol.scalar_type} kInput[] = {{
{corpus_dsp_protocol.format_cpp_array(values)}
}};
{support}
bool output_matches_expected(const {protocol.scalar_type} *output) {{
{_reference_body(protocol.kind)}}}
}} // namespace

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const {protocol.scalar_type} *input,
    {protocol.scalar_type} *output,
    std::size_t count) {{
  {instance_type} instance{{}};
  instance.Kp = {kp};
  instance.Ki = {ki};
  instance.Kd = {kd};
  {protocol.init_symbol}(&instance, 1);
  for (std::size_t index = 0; index < count; ++index) {{
    output[index] = {protocol.process_symbol}(&instance, input[index]);
  }}
}}

int main() {{
  {protocol.scalar_type} output[kSampleCount]{{}};
  {protocol_symbol}(kInput, output, kSampleCount);
  return output_matches_expected(output) ? 0 : 1;
}}
"""
