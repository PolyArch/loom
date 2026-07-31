#!/usr/bin/env python3
"""Typed atomic FFT protocols for CMSIS-DSP workloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import corpus_dsp_protocol
import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


class TransformKind(Enum):
    CFFT = "cfft"
    FAST_RFFT = "fast-rfft"
    FIXED_RFFT = "fixed-rfft"


@dataclass(frozen=True)
class TransformProtocol:
    kind: TransformKind
    suffix: str
    value_type: str
    bit_width: int
    test_class: str
    test_method: str
    calls: tuple[tuple[str, str], ...]
    target_profile: str
    absolute_tolerance: float | int
    relative_tolerance: float

    @property
    def owner_header(self) -> str:
        return (
            "transform_functions_f16.h"
            if self.suffix == "f16"
            else "transform_functions.h"
        )

    @property
    def floating_point(self) -> bool:
        return self.suffix in {"f16", "f32", "f64"}


@dataclass(frozen=True)
class MfccProtocol:
    suffix: str
    value_type: str
    bit_width: int
    test_class: str
    calls: tuple[tuple[str, str], ...]
    target_profile: str
    absolute_tolerance: float | int
    relative_tolerance: float

    @property
    def owner_header(self) -> str:
        return (
            "transform_functions_f16.h"
            if self.suffix == "f16"
            else "transform_functions.h"
        )

    @property
    def data_header(self) -> str:
        return "mfccdata_f16.h" if self.suffix == "f16" else "mfccdata.h"

    @property
    def data_source(self) -> str:
        return "mfccdata_f16.c" if self.suffix == "f16" else "mfccdata.c"

    @property
    def floating_point(self) -> bool:
        return self.suffix in {"f16", "f32"}


def _cfft_protocols() -> tuple[TransformProtocol, ...]:
    values = (
        (
            "f16",
            "float16_t",
            16,
            "TransformCF16",
            corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE,
            1.0e-1,
            1.0e-3,
        ),
        (
            "f32",
            "float32_t",
            32,
            "TransformCF32",
            corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
            8.0e-5,
            2.0e-5,
        ),
        (
            "f64",
            "float64_t",
            64,
            "TransformCF64",
            corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
            2.0e-13,
            1.0e-13,
        ),
        (
            "q15",
            "q15_t",
            16,
            "TransformCQ15",
            corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
            15,
            0.0,
        ),
        (
            "q31",
            "q31_t",
            32,
            "TransformCQ31",
            corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
            53,
            0.0,
        ),
    )
    return tuple(
        TransformProtocol(
            TransformKind.CFFT,
            suffix,
            value_type,
            bit_width,
            test_class,
            f"test_cfft_{suffix}",
            (
                (f"arm_cfft_init_{suffix}", "i32(ptr,i16)"),
                (f"arm_cfft_{suffix}", "void(ptr,ptr,i8,i8)"),
            ),
            profile,
            absolute_tolerance,
            relative_tolerance,
        )
        for (
            suffix,
            value_type,
            bit_width,
            test_class,
            profile,
            absolute_tolerance,
            relative_tolerance,
        ) in values
    )


def _rfft_protocols() -> tuple[TransformProtocol, ...]:
    floating = (
        (
            "f16",
            "float16_t",
            16,
            "TransformRF16",
            corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE,
            4.0e-2,
            5.0e-3,
        ),
        (
            "f32",
            "float32_t",
            32,
            "TransformRF32",
            corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
            5.0e-5,
            1.0e-5,
        ),
        (
            "f64",
            "float64_t",
            64,
            "TransformRF64",
            corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
            2.0e-13,
            3.0e-15,
        ),
    )
    fixed = (
        ("q15", "q15_t", 16, "TransformRQ15", 14),
        ("q31", "q31_t", 32, "TransformRQ31", 33),
    )
    return (
        *(
            TransformProtocol(
                TransformKind.FAST_RFFT,
                suffix,
                value_type,
                bit_width,
                test_class,
                f"test_rfft_{suffix}",
                (
                    (f"arm_rfft_fast_init_{suffix}", "i32(ptr,i16)"),
                    (f"arm_rfft_fast_{suffix}", "void(ptr,ptr,ptr,i8)"),
                ),
                profile,
                absolute_tolerance,
                relative_tolerance,
            )
            for (
                suffix,
                value_type,
                bit_width,
                test_class,
                profile,
                absolute_tolerance,
                relative_tolerance,
            ) in floating
        ),
        *(
            TransformProtocol(
                TransformKind.FIXED_RFFT,
                suffix,
                value_type,
                bit_width,
                test_class,
                f"test_rfft_{suffix}",
                (
                    (f"arm_rfft_init_{suffix}", "i32(ptr,i32,i32,i32)"),
                    (f"arm_rfft_{suffix}", "void(ptr,ptr,ptr)"),
                ),
                corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
                absolute_tolerance,
                0.0,
            )
            for suffix, value_type, bit_width, test_class, absolute_tolerance in fixed
        ),
    )


def _mfcc_protocols() -> tuple[MfccProtocol, ...]:
    values = (
        (
            "f16",
            "float16_t",
            16,
            "MFCCF16",
            corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE,
            2.0e-2,
            2.0e-2,
            "void(ptr,ptr,ptr,ptr)",
        ),
        (
            "f32",
            "float32_t",
            32,
            "MFCCF32",
            corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
            1.0e-5,
            1.2e-3,
            "void(ptr,ptr,ptr,ptr)",
        ),
        (
            "q15",
            "q15_t",
            16,
            "MFCCQ15",
            corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
            30,
            0.0,
            "i32(ptr,ptr,ptr,ptr)",
        ),
        (
            "q31",
            "q31_t",
            32,
            "MFCCQ31",
            corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
            49000,
            0.0,
            "i32(ptr,ptr,ptr,ptr)",
        ),
    )
    return tuple(
        MfccProtocol(
            suffix,
            value_type,
            bit_width,
            test_class,
            (
                (
                    f"arm_mfcc_init_{suffix}",
                    "i32(ptr,i32,i32,i32,ptr,ptr,ptr,ptr,ptr)",
                ),
                (f"arm_mfcc_{suffix}", execute_signature),
            ),
            profile,
            absolute_tolerance,
            relative_tolerance,
        )
        for (
            suffix,
            value_type,
            bit_width,
            test_class,
            profile,
            absolute_tolerance,
            relative_tolerance,
            execute_signature,
        ) in values
    )


_PROTOCOLS = (*_cfft_protocols(), *_rfft_protocols(), *_mfcc_protocols())
TransformProtocolRecord = TransformProtocol | MfccProtocol


def transform_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> TransformProtocolRecord | None:
    producer = workload.producer
    if (
        workload.suite != "cmsis-dsp"
        or not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer)
        or producer.selector_kind != "official"
        or producer.vector_ordinal != 0
    ):
        return None
    calls = tuple((call.symbol, call.signature) for call in workload.protocol)
    for protocol in _PROTOCOLS:
        if (
            workload.target_profile == protocol.target_profile
            and producer.test_class == protocol.test_class
            and producer.test_method
            == (
                protocol.test_method
                if isinstance(protocol, TransformProtocol)
                else f"test_mfcc_{protocol.suffix}"
            )
            and calls == protocol.calls
        ):
            return protocol
    return None


def _decode_values(
    protocol: TransformProtocolRecord, raw: bytes, name: str
) -> tuple[str, ...]:
    if protocol.suffix == "f16":
        return corpus_dsp_protocol.decode_f16_pattern(raw, name)
    if protocol.suffix == "f32":
        return corpus_dsp_protocol.decode_f32_pattern(raw, name)
    if protocol.suffix == "f64":
        return corpus_dsp_protocol.decode_f64_pattern(raw, name)
    return corpus_dsp_protocol.decode_integer_pattern(
        raw, protocol.bit_width, True, name
    )


def _comparison_body(protocol: TransformProtocolRecord, output_count: int) -> str:
    if protocol.floating_point:
        return f"""  for (std::size_t index = 0; index < {output_count}; ++index) {{
    const double actual = static_cast<double>(output[index]);
    const double expected = static_cast<double>(kExpected[index]);
    const double tolerance = {protocol.absolute_tolerance} +
                             {protocol.relative_tolerance} * std::fabs(expected);
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) return 1;
  }}
"""
    return f"""  for (std::size_t index = 0; index < {output_count}; ++index) {{
    std::int64_t delta = static_cast<std::int64_t>(output[index]) -
                         static_cast<std::int64_t>(kExpected[index]);
    if (delta < 0) delta = -delta;
    if (delta > {protocol.absolute_tolerance}) return 1;
  }}
"""


def _render_cfft(
    protocol: TransformProtocol,
    segments: dict[str, bytes],
    protocol_symbol: str,
) -> str:
    fft_length = 16
    input_name = f"ComplexInputSamples_Noisy_{fft_length}_1_{protocol.suffix}.txt"
    expected_name = f"ComplexFFTSamples_Noisy_{fft_length}_1_{protocol.suffix}.txt"
    inputs = _decode_values(
        protocol,
        corpus_dsp_protocol.require_pattern_segment(segments, input_name),
        "CFFT input",
    )
    expected = _decode_values(
        protocol,
        corpus_dsp_protocol.require_pattern_segment(segments, expected_name),
        "CFFT reference",
    )
    if len(inputs) != 2 * fft_length or len(expected) != len(inputs):
        raise WorkloadProviderError("CMSIS-DSP CFFT pattern has an invalid extent")
    arrays = corpus_dsp_protocol.format_cpp_array
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
constexpr std::size_t kScalarCount = {len(inputs)};
constexpr {protocol.value_type} kInput[kScalarCount] = {{
{arrays(inputs)}
}};
constexpr {protocol.value_type} kExpected[kScalarCount] = {{
{arrays(expected)}
}};
}}

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    {protocol.value_type} *data) {{
  arm_cfft_instance_{protocol.suffix} instance;
  const arm_status status = arm_cfft_init_{protocol.suffix}(&instance, kFftLength);
  if (status == ARM_MATH_SUCCESS)
    arm_cfft_{protocol.suffix}(&instance, data, 0, 1);
  return status;
}}

int main() {{
  {protocol.value_type} output[kScalarCount];
  for (std::size_t index = 0; index < kScalarCount; ++index) output[index] = kInput[index];
  if ({protocol_symbol}(output) != ARM_MATH_SUCCESS) return 1;
{_comparison_body(protocol, len(expected))}  return 0;
}}
"""


def _render_rfft(
    protocol: TransformProtocol,
    segments: dict[str, bytes],
    protocol_symbol: str,
) -> str:
    fft_length = 32
    input_name = f"RealInputSamples_Noisy_{fft_length}_2_{protocol.suffix}.txt"
    expected_name = f"RealFFTSamples_Noisy_{fft_length}_2_{protocol.suffix}.txt"
    inputs = _decode_values(
        protocol,
        corpus_dsp_protocol.require_pattern_segment(segments, input_name),
        "RFFT input",
    )
    expected = _decode_values(
        protocol,
        corpus_dsp_protocol.require_pattern_segment(segments, expected_name),
        "RFFT reference",
    )
    if len(inputs) != fft_length or len(expected) < fft_length:
        raise WorkloadProviderError("CMSIS-DSP RFFT pattern has an invalid extent")
    arrays = corpus_dsp_protocol.format_cpp_array
    output_capacity = (
        2 * len(expected)
        if protocol.kind is TransformKind.FIXED_RFFT
        else len(expected)
    )
    input_assignment = (
        "mutable_input[index] = static_cast<float16_t>(\n"
        "        static_cast<float>(input[index]) / 6000.0f);"
        if protocol.suffix == "f16"
        else "mutable_input[index] = input[index];"
    )
    output_rescale = (
        "  for (std::size_t index = 0; index < kExpectedCount; ++index)\n"
        "    output[index] = static_cast<float16_t>(\n"
        "        static_cast<float>(output[index]) * 6000.0f);\n"
        if protocol.suffix == "f16"
        else ""
    )
    if protocol.kind is TransformKind.FIXED_RFFT:
        init_call = f"arm_rfft_init_{protocol.suffix}(&instance, kFftLength, 0, 1)"
        execute_call = f"arm_rfft_{protocol.suffix}(&instance, mutable_input, output)"
        instance_type = f"arm_rfft_instance_{protocol.suffix}"
    else:
        init_call = f"arm_rfft_fast_init_{protocol.suffix}(&instance, kFftLength)"
        execute_call = (
            f"arm_rfft_fast_{protocol.suffix}(&instance, mutable_input, output, 0)"
        )
        instance_type = f"arm_rfft_fast_instance_{protocol.suffix}"
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
constexpr std::uint32_t kFftLength = {fft_length};
constexpr std::size_t kExpectedCount = {len(expected)};
constexpr {protocol.value_type} kInput[kFftLength] = {{
{arrays(inputs)}
}};
constexpr {protocol.value_type} kExpected[kExpectedCount] = {{
{arrays(expected)}
}};
}}

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    const {protocol.value_type} *input, {protocol.value_type} *output) {{
  {protocol.value_type} mutable_input[kFftLength];
  for (std::size_t index = 0; index < kFftLength; ++index)
    {input_assignment}
  {instance_type} instance;
  const arm_status status = {init_call};
  if (status == ARM_MATH_SUCCESS)
    {execute_call};
{output_rescale}  return status;
}}

int main() {{
  {protocol.value_type} output[{output_capacity}]{{}};
  if ({protocol_symbol}(kInput, output) != ARM_MATH_SUCCESS) return 1;
{_comparison_body(protocol, len(expected))}  return 0;
}}
"""


def _render_mfcc(
    protocol: MfccProtocol,
    segments: dict[str, bytes],
    protocol_symbol: str,
) -> str:
    fft_length = 256
    output_count = 13
    input_name = f"MFCCNoiseInput_{fft_length}_1_{protocol.suffix}.txt"
    expected_name = f"MFCCNoiseRef_{fft_length}_1_{protocol.suffix}.txt"
    inputs = _decode_values(
        protocol,
        corpus_dsp_protocol.require_pattern_segment(segments, input_name),
        "MFCC input",
    )
    expected = _decode_values(
        protocol,
        corpus_dsp_protocol.require_pattern_segment(segments, expected_name),
        "MFCC reference",
    )
    if len(inputs) != fft_length or len(expected) != output_count:
        raise WorkloadProviderError("CMSIS-DSP MFCC pattern has an invalid extent")
    arrays = corpus_dsp_protocol.format_cpp_array
    scratch_type = protocol.value_type if protocol.floating_point else "q31_t"
    execute = f"arm_mfcc_{protocol.suffix}(&instance, mutable_input, output, scratch)"
    execute_body = (
        f"  {execute};\n  return ARM_MATH_SUCCESS;"
        if protocol.floating_point
        else f"  return {execute};"
    )
    return f"""#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dsp/{protocol.owner_header}"
#include "{protocol.data_header}"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::uint32_t kFftLength = {fft_length};
constexpr std::uint32_t kFilterCount = 20;
constexpr std::uint32_t kOutputCount = {output_count};
constexpr {protocol.value_type} kInput[kFftLength] = {{
{arrays(inputs)}
}};
constexpr {protocol.value_type} kExpected[kOutputCount] = {{
{arrays(expected)}
}};
}}

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    const {protocol.value_type} *input, {protocol.value_type} *output) {{
  {protocol.value_type} mutable_input[kFftLength];
  {scratch_type} scratch[2 * kFftLength]{{}};
  for (std::size_t index = 0; index < kFftLength; ++index)
    mutable_input[index] = input[index];
  arm_mfcc_instance_{protocol.suffix} instance;
  const arm_status status = arm_mfcc_init_{protocol.suffix}(
      &instance, kFftLength, kFilterCount, kOutputCount,
      mfcc_dct_coefs_config1_{protocol.suffix},
      mfcc_filter_pos_config3_{protocol.suffix},
      mfcc_filter_len_config3_{protocol.suffix},
      mfcc_filter_coefs_config3_{protocol.suffix},
      mfcc_window_coefs_config3_{protocol.suffix});
  if (status != ARM_MATH_SUCCESS) return status;
{execute_body}
}}

int main() {{
  {protocol.value_type} output[kOutputCount]{{}};
  if ({protocol_symbol}(kInput, output) != ARM_MATH_SUCCESS) return 1;
{_comparison_body(protocol, output_count)}  return 0;
}}
"""


def render_transform_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path,
    protocol_symbol: str,
) -> str:
    protocol = transform_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no transform protocol: {workload.identity}"
        )
    segments = corpus_dsp_protocol.pattern_segments(patterns)
    if isinstance(protocol, MfccProtocol):
        return _render_mfcc(protocol, segments, protocol_symbol)
    if protocol.kind is TransformKind.CFFT:
        return _render_cfft(protocol, segments, protocol_symbol)
    return _render_rfft(protocol, segments, protocol_symbol)
