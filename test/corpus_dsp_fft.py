#!/usr/bin/env python3
"""Typed legacy complex FFT protocols for CMSIS-DSP workloads."""

from __future__ import annotations

import cmath
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import corpus_dsp_protocol
import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


@dataclass(frozen=True)
class LegacyCfftProtocol:
    radix: int
    suffix: str
    value_type: str
    bits: int | None
    test_class: str

    @property
    def test_method(self) -> str:
        return f"test_cfft_radix{self.radix}_{self.suffix}"

    @property
    def calls(self) -> tuple[tuple[str, str], ...]:
        return (
            (
                f"arm_cfft_radix{self.radix}_init_{self.suffix}",
                "i32(ptr,i16,i8,i8)",
            ),
            (f"arm_cfft_radix{self.radix}_{self.suffix}", "void(ptr,ptr)"),
        )

    @property
    def owner_header(self) -> str:
        return (
            "transform_functions_f16.h"
            if self.suffix == "f16"
            else "transform_functions.h"
        )

    @property
    def instance_type(self) -> str:
        return f"arm_cfft_radix{self.radix}_instance_{self.suffix}"

    @property
    def pattern_name(self) -> str:
        return f"ComplexInputSamples_Noisy_512_6_{self.suffix}.txt"


_SCALARS = (
    ("f16", "float16_t", None, "TransformF16"),
    ("f32", "float32_t", None, "TransformF32"),
    ("q15", "q15_t", 16, "TransformQ15"),
    ("q31", "q31_t", 32, "TransformQ31"),
)
_PROTOCOLS = tuple(
    LegacyCfftProtocol(radix, suffix, value_type, bits, test_class)
    for radix in (2, 4)
    for suffix, value_type, bits, test_class in _SCALARS
)


def legacy_cfft_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> LegacyCfftProtocol | None:
    producer = workload.producer
    if (
        workload.suite != "cmsis-dsp"
        or not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer)
        or producer.selector_kind != "benchmark-only"
        or producer.vector_ordinal != 0
        or len(workload.protocol) != 2
    ):
        return None
    calls = tuple((call.symbol, call.signature) for call in workload.protocol)
    for protocol in _PROTOCOLS:
        expected_profile = (
            corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE
            if protocol.suffix == "f16"
            else corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        )
        if workload.target_profile != expected_profile or calls != protocol.calls:
            continue
        if (
            producer.test_class == protocol.test_class
            and producer.test_method == protocol.test_method
        ):
            return protocol
    return None


def _decode_input(
    protocol: LegacyCfftProtocol, raw: bytes, scalar_count: int
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if protocol.suffix == "f16":
        byte_width, scalar_format = 2, "<e"
    elif protocol.suffix == "f32":
        byte_width, scalar_format = 4, "<f"
    else:
        if protocol.bits is None:
            raise WorkloadProviderError("CMSIS-DSP CFFT scalar type is incomplete")
        byte_width, scalar_format = protocol.bits // 8, None
    required = scalar_count * byte_width
    if len(raw) < required:
        raise WorkloadProviderError(
            "CMSIS-DSP CFFT input pattern is smaller than the transform extent"
        )

    literals: list[str] = []
    values: list[float] = []
    for offset in range(0, required, byte_width):
        chunk = raw[offset : offset + byte_width]
        if scalar_format is not None:
            value = struct.unpack(scalar_format, chunk)[0]
            if not math.isfinite(value):
                raise WorkloadProviderError("CMSIS-DSP CFFT input must be finite")
            literal = value.hex() + "f"
            if protocol.suffix == "f16":
                literal = f"static_cast<float16_t>({literal})"
        else:
            integer = int.from_bytes(chunk, byteorder="little", signed=True)
            assert protocol.bits is not None
            value = integer / float(1 << (protocol.bits - 1))
            literal = str(integer)
        literals.append(literal)
        values.append(value)
    return tuple(literals), tuple(values)


def _independent_dft(
    protocol: LegacyCfftProtocol, values: tuple[float, ...], fft_length: int
) -> tuple[str, ...]:
    samples = tuple(
        complex(values[2 * index], values[2 * index + 1]) for index in range(fft_length)
    )
    fixed_scale = 1.0 / fft_length if protocol.bits is not None else 1.0
    expected: list[str] = []
    for frequency in range(fft_length):
        value = sum(
            sample * cmath.exp(-2j * math.pi * frequency * index / fft_length)
            for index, sample in enumerate(samples)
        )
        expected.extend(
            ((value.real * fixed_scale).hex(), (value.imag * fixed_scale).hex())
        )
    return tuple(expected)


def _comparison_constants(protocol: LegacyCfftProtocol) -> tuple[str, str, str]:
    if protocol.suffix == "f16":
        return "1.0", "2.5e-2", "2.5e-2"
    if protocol.suffix == "f32":
        return "1.0", "2.0e-5", "2.0e-5"
    if protocol.suffix == "q15":
        return "32768.0", "3.0e-3", "1.0e-3"
    if protocol.suffix == "q31":
        return "2147483648.0", "2.0e-7", "2.0e-7"
    raise WorkloadProviderError(f"unknown CFFT scalar kind: {protocol.suffix}")


def render_legacy_cfft_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path,
    fft_length: int,
    protocol_symbol: str,
) -> str:
    protocol = legacy_cfft_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no legacy CFFT provider: {workload.identity}"
        )
    if fft_length <= 0 or fft_length & (fft_length - 1):
        raise WorkloadProviderError("CMSIS-DSP CFFT length must be a power of two")

    segments = corpus_dsp_protocol.pattern_segments(patterns)
    inputs, numeric_inputs = _decode_input(
        protocol,
        corpus_dsp_protocol.require_pattern_segment(segments, protocol.pattern_name),
        2 * fft_length,
    )
    expected = _independent_dft(protocol, numeric_inputs, fft_length)
    output_scale, absolute_error, relative_error = _comparison_constants(protocol)
    format_array = corpus_dsp_protocol.format_cpp_array
    init_symbol, transform_symbol = (call[0] for call in protocol.calls)
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
constexpr std::uint16_t kFftLength = {fft_length};
constexpr std::size_t kScalarCount = 2 * kFftLength;
constexpr {protocol.value_type} kInput[kScalarCount] = {{
{format_array(inputs)}
}};
constexpr double kExpected[kScalarCount] = {{
{format_array(expected)}
}};
}} // namespace

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    {protocol.value_type} *data) {{
  {protocol.instance_type} instance;
  const arm_status status = {init_symbol}(&instance, kFftLength, 0, 1);
  if (status == ARM_MATH_SUCCESS) {{
    {transform_symbol}(&instance, data);
  }}
  return status;
}}

int main() {{
  {protocol.value_type} output[kScalarCount];
  for (std::size_t index = 0; index < kScalarCount; ++index) {{
    output[index] = kInput[index];
  }}
  const arm_status status = {protocol_symbol}(output);
  const auto output_matches_independent_dft = [&]() {{
    for (std::size_t index = 0; index < kScalarCount; ++index) {{
      const double actual = static_cast<double>(output[index]) / {output_scale};
      const double expected = kExpected[index];
      const double tolerance = {absolute_error} + {relative_error} * std::fabs(expected);
      if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {{
        return false;
      }}
    }}
    return true;
  }};
  return status == ARM_MATH_SUCCESS && output_matches_independent_dft() ? 0 : 1;
}}
"""
