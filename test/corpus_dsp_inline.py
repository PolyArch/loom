#!/usr/bin/env python3
"""Typed protocols for header-defined CMSIS-DSP operators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import corpus_dsp_protocol
import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


class HeaderProtocolKind(Enum):
    DIVIDE_S64 = "divide-s64"
    NORMALIZE_U64 = "normalize-u64"
    RECIPROCAL_Q15 = "reciprocal-q15"
    RECIPROCAL_Q31 = "reciprocal-q31"
    SQRT_F16 = "sqrt-f16"
    READ_Q15X2 = "read-q15x2"
    READ_Q15X2_IA = "read-q15x2-ia"
    READ_Q15X2_DA = "read-q15x2-da"
    READ_Q7X4_IA = "read-q7x4-ia"
    READ_Q7X4_DA = "read-q7x4-da"
    WRITE_Q15X2 = "write-q15x2"
    WRITE_Q15X2_IA = "write-q15x2-ia"
    WRITE_Q7X4_IA = "write-q7x4-ia"


@dataclass(frozen=True)
class HeaderDefinedProtocol:
    kind: HeaderProtocolKind
    target_profile: str
    test_class: str
    test_method: str
    call: tuple[str, str]
    owner_header: str


_PORTABLE = corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
_FLOAT16 = corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE
_PROTOCOLS = (
    HeaderDefinedProtocol(
        HeaderProtocolKind.DIVIDE_S64,
        _PORTABLE,
        "FastMathQ63",
        "test_div_int64_to_int32",
        ("arm_div_int64_to_int32", "i32(i64,i32)"),
        "dsp/utils.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.NORMALIZE_U64,
        _PORTABLE,
        "FastMathQ63",
        "test_norm_64_to_32u",
        ("arm_norm_64_to_32u", "void(i64,ptr,ptr)"),
        "dsp/utils.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.RECIPROCAL_Q15,
        _PORTABLE,
        "FastMathQ15",
        "test_recip_q15",
        ("arm_recip_q15", "i32(i16,ptr,ptr)"),
        "dsp/utils.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.RECIPROCAL_Q31,
        _PORTABLE,
        "FastMathQ31",
        "test_recip_q31",
        ("arm_recip_q31", "i32(i32,ptr,ptr)"),
        "dsp/utils.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.SQRT_F16,
        _FLOAT16,
        "FastMathF16",
        "test_sqrt_f16",
        ("arm_sqrt_f16", "i32(half,ptr)"),
        "dsp/fast_math_functions_f16.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.READ_Q15X2,
        _PORTABLE,
        "SupportTestsQ15",
        "test_read_q15x2",
        ("read_q15x2", "i32(ptr)"),
        "arm_math_memory.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.READ_Q15X2_IA,
        _PORTABLE,
        "SupportTestsQ15",
        "test_read_q15x2_ia",
        ("read_q15x2_ia", "i32(ptr)"),
        "arm_math_memory.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.READ_Q15X2_DA,
        _PORTABLE,
        "SupportTestsQ15",
        "test_read_q15x2_da",
        ("read_q15x2_da", "i32(ptr)"),
        "arm_math_memory.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.READ_Q7X4_IA,
        _PORTABLE,
        "SupportTestsQ7",
        "test_read_q7x4_ia",
        ("read_q7x4_ia", "i32(ptr)"),
        "arm_math_memory.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.READ_Q7X4_DA,
        _PORTABLE,
        "SupportTestsQ7",
        "test_read_q7x4_da",
        ("read_q7x4_da", "i32(ptr)"),
        "arm_math_memory.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.WRITE_Q15X2,
        _PORTABLE,
        "SupportTestsQ15",
        "test_write_q15x2",
        ("write_q15x2", "void(ptr,i32)"),
        "arm_math_memory.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.WRITE_Q15X2_IA,
        _PORTABLE,
        "SupportTestsQ15",
        "test_write_q15x2_ia",
        ("write_q15x2_ia", "void(ptr,i32)"),
        "arm_math_memory.h",
    ),
    HeaderDefinedProtocol(
        HeaderProtocolKind.WRITE_Q7X4_IA,
        _PORTABLE,
        "SupportTestsQ7",
        "test_write_q7x4_ia",
        ("write_q7x4_ia", "void(ptr,i32)"),
        "arm_math_memory.h",
    ),
)


def header_defined_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> HeaderDefinedProtocol | None:
    producer = workload.producer
    if (
        workload.suite != "cmsis-dsp"
        or not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer)
        or producer.selector_kind != "official"
        or producer.vector_ordinal != 0
        or len(workload.protocol) != 1
    ):
        return None
    call = (workload.protocol[0].symbol, workload.protocol[0].signature)
    matches = tuple(
        protocol
        for protocol in _PROTOCOLS
        if protocol.target_profile == workload.target_profile
        and protocol.test_class == producer.test_class
        and protocol.test_method == producer.test_method
        and protocol.call == call
    )
    if len(matches) > 1:
        raise WorkloadProviderError(
            f"CMSIS-DSP header protocol is ambiguous: {workload.identity}"
        )
    return matches[0] if matches else None


def _preamble(*headers: str) -> str:
    includes = "\n".join(f'#include "{header}"' for header in headers)
    return f"""#include <cmath>
#include <cstddef>
#include <cstdint>

{includes}

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif
"""


def _decode_ints(raw: bytes, bits: int, *, signed: bool) -> tuple[int, ...]:
    width = bits // 8
    if bits not in {8, 16, 32, 64} or len(raw) % width:
        raise WorkloadProviderError("CMSIS-DSP integer pattern has invalid width")
    return tuple(
        int.from_bytes(raw[offset : offset + width], "little", signed=signed)
        for offset in range(0, len(raw), width)
    )


def _repeat(values: Sequence[int | str], minimum: int = 32) -> tuple[int | str, ...]:
    if not values:
        raise WorkloadProviderError("CMSIS-DSP header protocol pattern is empty")
    count = max(minimum, len(values))
    return tuple(values[index % len(values)] for index in range(count))


def _integer_literal(value: int, bits: int, *, signed: bool) -> str:
    if not signed:
        return f"UINT{bits}_C({value})"
    minimum = -(1 << (bits - 1))
    if value == minimum:
        return f"INT{bits}_MIN"
    if value < 0:
        return f"-INT{bits}_C({-value})"
    return f"INT{bits}_C({value})"


def _format_integers(values: Sequence[int], bits: int, *, signed: bool) -> str:
    return corpus_dsp_protocol.format_cpp_array(
        tuple(_integer_literal(value, bits, signed=signed) for value in values)
    )


def _segment_ints(
    segments: dict[str, bytes], name: str, bits: int, *, signed: bool
) -> tuple[int, ...]:
    return _decode_ints(
        corpus_dsp_protocol.require_pattern_segment(segments, name),
        bits,
        signed=signed,
    )


def _render_divide(patterns: Path, symbol: str) -> str:
    segments = corpus_dsp_protocol.pattern_segments(patterns)
    numerators = _repeat(
        _segment_ints(segments, "DivDenInput1_s64.txt", 64, signed=True)
    )
    denominators = _repeat(
        _segment_ints(segments, "DivNumInput1_s32.txt", 32, signed=True)
    )
    expected = _repeat(_segment_ints(segments, "DivRef1_s32.txt", 32, signed=True))
    if not (len(numerators) == len(denominators) == len(expected)):
        raise WorkloadProviderError("CMSIS-DSP divide patterns have unequal extents")
    return f"""{_preamble("arm_math.h", "dsp/utils.h")}
namespace {{
constexpr std::size_t kCount = {len(expected)};
constexpr std::int64_t kNumerators[] = {{
{_format_integers(numerators, 64, signed=True)}
}};
constexpr std::int32_t kDenominators[] = {{
{_format_integers(denominators, 32, signed=True)}
}};
constexpr std::int32_t kExpected[] = {{
{_format_integers(expected, 32, signed=True)}
}};
}}

extern "C" LOOM_NOINLINE void {symbol}(
    const std::int64_t *numerators, const std::int32_t *denominators,
    std::int32_t *output, std::uint32_t count) {{
  for (std::uint32_t index = 0; index < count; ++index)
    output[index] = arm_div_int64_to_int32(numerators[index], denominators[index]);
}}

int main() {{
  std::int32_t output[kCount]{{}};
  {symbol}(kNumerators, kDenominators, output, kCount);
  for (std::size_t index = 0; index < kCount; ++index)
    if (output[index] != kExpected[index]) return 1;
  return 0;
}}
"""


def _render_normalize(patterns: Path, symbol: str) -> str:
    segments = corpus_dsp_protocol.pattern_segments(patterns)
    inputs = _repeat(
        _segment_ints(segments, "Norm64To32_Input1_u64.txt", 64, signed=False)
    )
    expected_values = _repeat(
        _segment_ints(segments, "RefNorm64To32_Vals1_s32.txt", 32, signed=True)
    )
    expected_norms = _repeat(
        _segment_ints(segments, "RefNorm64To32_Norms1_s16.txt", 16, signed=True)
    )
    if not (len(inputs) == len(expected_values) == len(expected_norms)):
        raise WorkloadProviderError("CMSIS-DSP normalize patterns have unequal extents")
    return f"""{_preamble("arm_math.h", "dsp/utils.h")}
namespace {{
constexpr std::size_t kCount = {len(inputs)};
constexpr std::uint64_t kInputs[] = {{
{_format_integers(inputs, 64, signed=False)}
}};
constexpr std::int32_t kExpectedValues[] = {{
{_format_integers(expected_values, 32, signed=True)}
}};
constexpr std::int16_t kExpectedNorms[] = {{
{_format_integers(expected_norms, 16, signed=True)}
}};
}}

extern "C" LOOM_NOINLINE void {symbol}(
    const std::uint64_t *input, std::int32_t *values, std::int32_t *norms,
    std::uint32_t count) {{
  for (std::uint32_t index = 0; index < count; ++index)
    arm_norm_64_to_32u(input[index], &values[index], &norms[index]);
}}

int main() {{
  std::int32_t values[kCount]{{}};
  std::int32_t norms[kCount]{{}};
  {symbol}(kInputs, values, norms, kCount);
  for (std::size_t index = 0; index < kCount; ++index)
    if (values[index] != kExpectedValues[index] ||
        norms[index] != kExpectedNorms[index]) return 1;
  return 0;
}}
"""


def _render_reciprocal(
    patterns: Path, symbol: str, *, q_format: int, storage_bits: int
) -> str:
    suffix = f"q{q_format}"
    segments = corpus_dsp_protocol.pattern_segments(patterns)
    inputs = _repeat(
        _segment_ints(segments, f"RecipInput1_{suffix}.txt", storage_bits, signed=True)
    )
    expected = _repeat(
        _segment_ints(segments, f"RecipRef1_{suffix}.txt", storage_bits, signed=True)
    )
    shifts = _repeat(_segment_ints(segments, "RecipShift1_s16.txt", 16, signed=True))
    if not (len(inputs) == len(expected) == len(shifts)):
        raise WorkloadProviderError(
            "CMSIS-DSP reciprocal patterns have unequal extents"
        )
    value_type = f"q{q_format}_t"
    table = f"armRecipTableQ{q_format}"
    tolerance = 2 if q_format == 15 else 10
    return f"""{_preamble("arm_math.h", "arm_common_tables.h", "dsp/utils.h")}
namespace {{
constexpr std::size_t kCount = {len(inputs)};
constexpr {value_type} kInputs[] = {{
{_format_integers(inputs, storage_bits, signed=True)}
}};
constexpr {value_type} kExpected[] = {{
{_format_integers(expected, storage_bits, signed=True)}
}};
constexpr std::int16_t kExpectedShifts[] = {{
{_format_integers(shifts, 16, signed=True)}
}};
}}

extern "C" LOOM_NOINLINE void {symbol}(
    const {value_type} *input, {value_type} *output, std::int16_t *shifts,
    std::uint32_t count) {{
  for (std::uint32_t index = 0; index < count; ++index)
    shifts[index] = static_cast<std::int16_t>(
        arm_recip_q{q_format}(input[index], &output[index], {table}));
}}

int main() {{
  {value_type} output[kCount]{{}};
  std::int16_t shifts[kCount]{{}};
  {symbol}(kInputs, output, shifts, kCount);
  for (std::size_t index = 0; index < kCount; ++index)
    if (std::abs(static_cast<std::int64_t>(output[index]) -
                 static_cast<std::int64_t>(kExpected[index])) > {tolerance} ||
        shifts[index] != kExpectedShifts[index]) return 1;
  return 0;
}}
"""


def _render_sqrt_f16(patterns: Path, symbol: str) -> str:
    segments = corpus_dsp_protocol.pattern_segments(patterns)
    inputs = corpus_dsp_protocol.decode_f16_pattern(
        corpus_dsp_protocol.require_pattern_segment(segments, "SqrtInput1_f16.txt"),
        "sqrt f16 input",
    )[:32]
    expected = corpus_dsp_protocol.decode_f16_pattern(
        corpus_dsp_protocol.require_pattern_segment(segments, "Sqrt1_f16.txt"),
        "sqrt f16 reference",
    )[:32]
    if not inputs or len(inputs) != len(expected):
        raise WorkloadProviderError("CMSIS-DSP sqrt f16 patterns have unequal extents")
    return f"""{_preamble("dsp/fast_math_functions_f16.h")}
namespace {{
constexpr std::size_t kCount = {len(inputs)};
const float16_t kInputs[] = {{
{corpus_dsp_protocol.format_cpp_array(inputs)}
}};
const float16_t kExpected[] = {{
{corpus_dsp_protocol.format_cpp_array(expected)}
}};
}}

extern "C" LOOM_NOINLINE void {symbol}(
    const float16_t *input, float16_t *output, std::int32_t *status,
    std::uint32_t count) {{
  for (std::uint32_t index = 0; index < count; ++index)
    status[index] = static_cast<std::int32_t>(arm_sqrt_f16(input[index], &output[index]));
}}

int main() {{
  float16_t output[kCount]{{}};
  std::int32_t status[kCount]{{}};
  {symbol}(kInputs, output, status, kCount);
  for (std::size_t index = 0; index < kCount; ++index) {{
    const std::int32_t expected_status =
        static_cast<float>(kInputs[index]) < 0.0f
            ? ARM_MATH_ARGUMENT_ERROR : ARM_MATH_SUCCESS;
    const float actual = static_cast<float>(output[index]);
    const float expected = static_cast<float>(kExpected[index]);
    if (status[index] != expected_status ||
        std::fabs(actual - expected) > 2.0e-3f + 2.0e-3f * std::fabs(expected))
      return 1;
  }}
  return 0;
}}
"""


def _packed_words(bits: int, count: int = 32) -> tuple[int, ...]:
    lanes = 32 // bits
    mask = (1 << bits) - 1
    words = []
    for word_index in range(count):
        word = 0
        for lane in range(lanes):
            value = (word_index * 37 + lane * 53 + 11) & mask
            word |= value << (lane * bits)
        words.append(word)
    return tuple(words)


def _word_literal(word: int) -> str:
    return f"static_cast<q31_t>(UINT32_C(0x{word:08x}))"


def _unpack_words(words: Sequence[int], bits: int) -> tuple[int, ...]:
    lanes = 32 // bits
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    values = []
    for word in words:
        for lane in range(lanes):
            value = (word >> (lane * bits)) & mask
            values.append(value - (1 << bits) if value & sign else value)
    return tuple(values)


def _render_read_memory(
    protocol: HeaderDefinedProtocol, symbol: str, *, bits: int
) -> str:
    words = _packed_words(bits)
    lanes = 32 // bits
    q_format = bits - 1
    value_type = f"q{q_format}_t"
    call = protocol.call[0]
    direct_read = f"read_q{q_format}x{lanes}"
    advancing = protocol.kind in {
        HeaderProtocolKind.READ_Q15X2_IA,
        HeaderProtocolKind.READ_Q15X2_DA,
        HeaderProtocolKind.READ_Q7X4_IA,
        HeaderProtocolKind.READ_Q7X4_DA,
    }
    descending = protocol.kind in {
        HeaderProtocolKind.READ_Q15X2_DA,
        HeaderProtocolKind.READ_Q7X4_DA,
    }
    if not advancing:
        invocation = f"output[index] = {call}(input + index * kLanes);"
        output_count = len(words)
        expected_words = words
    else:
        pair_count = len(words) // 2
        start_offset = (
            "kLanes + index * 2 * kLanes" if descending else "index * 2 * kLanes"
        )
        invocation = f"""{value_type} *cursor = const_cast<{value_type} *>(input + {start_offset});
    output[2 * index] = {call}(&cursor);
    output[2 * index + 1] = {direct_read}(cursor);"""
        output_count = len(words)
        expected_words = tuple(
            value
            for index in range(pair_count)
            for value in (
                (words[2 * index], words[2 * index - 1] if index else 0)
                if descending
                else (words[2 * index], words[2 * index + 1])
            )
        )
        if descending:
            words = (0, *words)
    input_values = _unpack_words(words, bits)
    return f"""{_preamble("arm_math_memory.h")}
namespace {{
constexpr std::uint32_t kLanes = {lanes};
constexpr std::uint32_t kCount = {output_count if not advancing else output_count // 2};
constexpr {value_type} kInput[] = {{
{_format_integers(input_values, bits, signed=True)}
}};
constexpr std::uint32_t kExpected[] = {{
{corpus_dsp_protocol.format_cpp_array(tuple(f"UINT32_C(0x{word:08x})" for word in expected_words))}
}};
}}

extern "C" LOOM_NOINLINE void {symbol}(
    const {value_type} *input, q31_t *output, std::uint32_t count) {{
  for (std::uint32_t index = 0; index < count; ++index) {{
    {invocation}
  }}
}}

int main() {{
  q31_t output[{output_count}]{{}};
  {symbol}(kInput, output, kCount);
  for (std::size_t index = 0; index < {output_count}; ++index)
    if (static_cast<std::uint32_t>(output[index]) != kExpected[index]) return 1;
  return 0;
}}
"""


def _render_write_memory(
    protocol: HeaderDefinedProtocol, symbol: str, *, bits: int
) -> str:
    words = _packed_words(bits)
    lanes = 32 // bits
    value_type = f"q{bits - 1}_t"
    call = protocol.call[0]
    advancing = protocol.kind in {
        HeaderProtocolKind.WRITE_Q15X2_IA,
        HeaderProtocolKind.WRITE_Q7X4_IA,
    }
    if advancing:
        pair_count = len(words) // 2
        invocation = f"""{value_type} *cursor = output + index * 2 * kLanes;
    {call}(&cursor, input[2 * index]);
    const std::uint32_t second = static_cast<std::uint32_t>(input[2 * index + 1]);
    for (std::uint32_t lane = 0; lane < kLanes; ++lane)
      cursor[lane] = static_cast<{value_type}>(second >> (lane * {bits}));"""
        count = pair_count
    else:
        invocation = f"{call}(output + index * kLanes, input[index]);"
        count = len(words)
    expected = _unpack_words(words, bits)
    return f"""{_preamble("arm_math_memory.h")}
namespace {{
constexpr std::uint32_t kLanes = {lanes};
constexpr std::uint32_t kCount = {count};
constexpr q31_t kInput[] = {{
{corpus_dsp_protocol.format_cpp_array(tuple(_word_literal(word) for word in words))}
}};
constexpr {value_type} kExpected[] = {{
{_format_integers(expected, bits, signed=True)}
}};
}}

extern "C" LOOM_NOINLINE void {symbol}(
    const q31_t *input, {value_type} *output, std::uint32_t count) {{
  for (std::uint32_t index = 0; index < count; ++index) {{
    {invocation}
  }}
}}

int main() {{
  {value_type} output[{len(expected)}]{{}};
  {symbol}(kInput, output, kCount);
  for (std::size_t index = 0; index < {len(expected)}; ++index)
    if (output[index] != kExpected[index]) return 1;
  return 0;
}}
"""


def render_header_defined_protocol(
    workload: corpus_inventory.ProgramWorkload,
    patterns: Path,
    protocol_symbol: str,
) -> str:
    protocol = header_defined_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload is not a header-defined protocol: {workload.identity}"
        )
    if protocol.kind == HeaderProtocolKind.DIVIDE_S64:
        return _render_divide(patterns, protocol_symbol)
    if protocol.kind == HeaderProtocolKind.NORMALIZE_U64:
        return _render_normalize(patterns, protocol_symbol)
    if protocol.kind == HeaderProtocolKind.RECIPROCAL_Q15:
        return _render_reciprocal(
            patterns, protocol_symbol, q_format=15, storage_bits=16
        )
    if protocol.kind == HeaderProtocolKind.RECIPROCAL_Q31:
        return _render_reciprocal(
            patterns, protocol_symbol, q_format=31, storage_bits=32
        )
    if protocol.kind == HeaderProtocolKind.SQRT_F16:
        return _render_sqrt_f16(patterns, protocol_symbol)
    if protocol.kind in {
        HeaderProtocolKind.READ_Q15X2,
        HeaderProtocolKind.READ_Q15X2_IA,
        HeaderProtocolKind.READ_Q15X2_DA,
    }:
        return _render_read_memory(protocol, protocol_symbol, bits=16)
    if protocol.kind in {
        HeaderProtocolKind.READ_Q7X4_IA,
        HeaderProtocolKind.READ_Q7X4_DA,
    }:
        return _render_read_memory(protocol, protocol_symbol, bits=8)
    if protocol.kind in {
        HeaderProtocolKind.WRITE_Q15X2,
        HeaderProtocolKind.WRITE_Q15X2_IA,
    }:
        return _render_write_memory(protocol, protocol_symbol, bits=16)
    if protocol.kind == HeaderProtocolKind.WRITE_Q7X4_IA:
        return _render_write_memory(protocol, protocol_symbol, bits=8)
    raise AssertionError(f"unhandled header protocol kind: {protocol.kind}")
