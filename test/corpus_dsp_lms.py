#!/usr/bin/env python3
"""Typed LMS protocols for CMSIS-DSP benchmark workloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import corpus_dsp_protocol
import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


@dataclass(frozen=True)
class LmsProtocol:
    test_class: str
    suffix: str
    value_type: str
    bits: int | None
    normalized: bool
    init_signature: str

    @property
    def stem(self) -> str:
        return "lms_norm" if self.normalized else "lms"

    @property
    def test_method(self) -> str:
        return f"test_{self.stem}_{self.suffix}"

    @property
    def calls(self) -> tuple[tuple[str, str], ...]:
        init_signature = self.init_signature
        if self.normalized and self.suffix in {"q15", "q31"}:
            init_signature = init_signature[:-4] + "i8)"
        return (
            (f"arm_{self.stem}_init_{self.suffix}", init_signature),
            (f"arm_{self.stem}_{self.suffix}", "void(ptr,ptr,ptr,ptr,ptr,i32)"),
        )

    @property
    def owner_header(self) -> str:
        return "filtering_functions.h"


_PROTOCOLS = tuple(
    LmsProtocol(test_class, suffix, value_type, bits, normalized, init_signature)
    for test_class, suffix, value_type, bits, init_signature in (
        ("FIRF32", "f32", "float32_t", None, "void(ptr,i16,ptr,ptr,float,i32)"),
        ("FIRQ15", "q15", "q15_t", 16, "void(ptr,i16,ptr,ptr,i16,i32,i32)"),
        ("FIRQ31", "q31", "q31_t", 32, "void(ptr,i16,ptr,ptr,i32,i32,i32)"),
    )
    for normalized in (False, True)
)


def lms_protocol(workload: corpus_inventory.ProgramWorkload) -> LmsProtocol | None:
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
    for protocol in _PROTOCOLS:
        if calls != protocol.calls:
            continue
        if (
            producer.test_class == protocol.test_class
            and producer.test_method == protocol.test_method
            and producer.vector_ordinal == 0
        ):
            return protocol
    return None


def _decode_values(protocol: LmsProtocol, raw: bytes, name: str) -> tuple[str, ...]:
    if protocol.suffix == "f32":
        return corpus_dsp_protocol.decode_f32_pattern(raw, name)
    if protocol.bits is None:
        raise WorkloadProviderError("CMSIS-DSP LMS scalar type is incomplete")
    return corpus_dsp_protocol.decode_integer_pattern(raw, protocol.bits, True, name)


def _oracle_support(protocol: LmsProtocol) -> str:
    if protocol.suffix == "f32":
        return """bool error_matches_reference(
    const float32_t *reference, const float32_t *output,
    const float32_t *error) {
  for (std::size_t index = 0; index < kSampleCount; ++index) {
    const float32_t expected = reference[index] - output[index];
    if (!std::isfinite(expected) || !std::isfinite(error[index])) {
      return false;
    }
    const float32_t difference = std::fabs(error[index] - expected);
    if (difference > 1.0e-6f * (1.0f + std::fabs(expected))) {
      return false;
    }
  }
  return true;
}
"""
    bit_width = protocol.bits
    assert bit_width is not None
    unsigned_type = f"std::uint{bit_width}_t"
    signed_type = f"std::int{bit_width}_t"
    sign_bit = 1 << (bit_width - 1)
    modulus = 1 << bit_width
    return f"""{signed_type} signed_from_bits({unsigned_type} bits) {{
  if (bits < {sign_bit}U) {{
    return static_cast<{signed_type}>(bits);
  }}
  return static_cast<{signed_type}>(
      static_cast<std::int64_t>(bits) - {modulus}LL);
}}

bool error_matches_reference(
    const {protocol.value_type} *reference,
    const {protocol.value_type} *output,
    const {protocol.value_type} *error) {{
  for (std::size_t index = 0; index < kSampleCount; ++index) {{
    const {unsigned_type} bits =
        static_cast<{unsigned_type}>(reference[index]) -
        static_cast<{unsigned_type}>(output[index]);
    if (error[index] != signed_from_bits(bits)) {{
      return false;
    }}
  }}
  return true;
}}
"""


def _init_tail(protocol: LmsProtocol) -> str:
    if protocol.suffix == "f32":
        return "0.1f, kSampleCount"
    return "100, kSampleCount, 1"


def render_lms_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path,
    dimensions: tuple[int, int],
    protocol_symbol: str,
) -> str:
    protocol = lms_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no LMS provider: {workload.identity}"
        )
    num_taps, sample_count = dimensions
    if min(dimensions) <= 0:
        raise WorkloadProviderError("CMSIS-DSP LMS dimensions must be positive")

    segments = corpus_dsp_protocol.pattern_segments(patterns)
    suffix = protocol.suffix
    inputs = _decode_values(
        protocol,
        corpus_dsp_protocol.require_pattern_segment(segments, f"Samples1_{suffix}.txt"),
        "LMS input",
    )[:sample_count]
    reference = _decode_values(
        protocol,
        corpus_dsp_protocol.require_pattern_segment(segments, f"Refs1_{suffix}.txt"),
        "LMS reference",
    )[:sample_count]
    coefficients = _decode_values(
        protocol,
        corpus_dsp_protocol.require_pattern_segment(segments, f"Coefs1_{suffix}.txt"),
        "LMS coefficients",
    )[:num_taps]
    if (
        len(inputs) != sample_count
        or len(reference) != sample_count
        or len(coefficients) != num_taps
    ):
        raise WorkloadProviderError(
            "CMSIS-DSP LMS patterns are smaller than their selected dimensions"
        )

    init_symbol, process_symbol = (call[0] for call in protocol.calls)
    instance_type = f"arm_{protocol.stem}_instance_{suffix}"
    format_array = corpus_dsp_protocol.format_cpp_array
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
constexpr std::size_t kNumTaps = {num_taps};
constexpr std::size_t kSampleCount = {sample_count};
constexpr {protocol.value_type} kInput[] = {{
{format_array(inputs)}
}};
constexpr {protocol.value_type} kReference[] = {{
{format_array(reference)}
}};
constexpr {protocol.value_type} kInitialCoefficients[] = {{
{format_array(coefficients)}
}};

{_oracle_support(protocol)}
bool coefficients_changed(const {protocol.value_type} *coefficients) {{
  for (std::size_t index = 0; index < kNumTaps; ++index) {{
    if (coefficients[index] != kInitialCoefficients[index]) {{
      return true;
    }}
  }}
  return false;
}}

bool output_is_active(const {protocol.value_type} *output,
                      const {protocol.value_type} *error) {{
  for (std::size_t index = 0; index < kSampleCount; ++index) {{
    if (output[index] != 0 || error[index] != 0) {{
      return true;
    }}
  }}
  return false;
}}
}} // namespace

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const {protocol.value_type} *input,
    const {protocol.value_type} *reference,
    {protocol.value_type} *coefficients,
    {protocol.value_type} *state,
    {protocol.value_type} *output,
    {protocol.value_type} *error) {{
  {instance_type} instance;
  {init_symbol}(&instance, kNumTaps, coefficients, state,
                {_init_tail(protocol)});
  {process_symbol}(&instance, input,
                   const_cast<{protocol.value_type} *>(reference),
                   output, error, kSampleCount);
}}

int main() {{
  {protocol.value_type} coefficients[kNumTaps];
  {protocol.value_type} state[kNumTaps + kSampleCount - 1];
  {protocol.value_type} output[kSampleCount]{{}};
  {protocol.value_type} error[kSampleCount]{{}};
  for (std::size_t index = 0; index < kNumTaps; ++index) {{
    coefficients[index] = kInitialCoefficients[index];
  }}
  {protocol_symbol}(kInput, kReference, coefficients, state, output, error);
  return error_matches_reference(kReference, output, error) &&
                 coefficients_changed(coefficients) &&
                 output_is_active(output, error)
             ? 0
             : 1;
}}
"""
