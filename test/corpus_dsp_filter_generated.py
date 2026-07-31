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


class StatefulFilterFamily(Enum):
    FAST_BIQUAD = "fast-biquad"
    FAST_DECIMATE = "fast-decimate"
    FAST_FIR = "fast-fir"
    FIR_LATTICE = "fir-lattice"
    FIR_SPARSE = "fir-sparse"
    IIR_LATTICE = "iir-lattice"


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


@dataclass(frozen=True)
class StatefulFilterProtocol:
    family: StatefulFilterFamily
    suffix: str
    value_type: str
    calls: tuple[tuple[str, str], ...]

    @property
    def symbol(self) -> str:
        return self.calls[-1][0]

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


def _stateful_filter_protocols() -> tuple[StatefulFilterProtocol, ...]:
    return (
        StatefulFilterProtocol(
            StatefulFilterFamily.FAST_BIQUAD,
            "q15",
            "q15_t",
            (
                ("arm_biquad_cascade_df1_init_q15", "void(ptr,i8,ptr,ptr,i8)"),
                ("arm_biquad_cascade_df1_fast_q15", "void(ptr,ptr,ptr,i32)"),
            ),
        ),
        StatefulFilterProtocol(
            StatefulFilterFamily.FAST_BIQUAD,
            "q31",
            "q31_t",
            (
                ("arm_biquad_cascade_df1_init_q31", "void(ptr,i8,ptr,ptr,i8)"),
                ("arm_biquad_cascade_df1_fast_q31", "void(ptr,ptr,ptr,i32)"),
            ),
        ),
        StatefulFilterProtocol(
            StatefulFilterFamily.FAST_DECIMATE,
            "q15",
            "q15_t",
            (
                ("arm_fir_decimate_init_q15", "i32(ptr,i16,i8,ptr,ptr,i32)"),
                ("arm_fir_decimate_fast_q15", "void(ptr,ptr,ptr,i32)"),
            ),
        ),
        StatefulFilterProtocol(
            StatefulFilterFamily.FAST_DECIMATE,
            "q31",
            "q31_t",
            (
                ("arm_fir_decimate_init_q31", "i32(ptr,i16,i8,ptr,ptr,i32)"),
                ("arm_fir_decimate_fast_q31", "void(ptr,ptr,ptr,i32)"),
            ),
        ),
        StatefulFilterProtocol(
            StatefulFilterFamily.FAST_FIR,
            "q15",
            "q15_t",
            (
                ("arm_fir_init_q15", "i32(ptr,i16,ptr,ptr,i32)"),
                ("arm_fir_fast_q15", "void(ptr,ptr,ptr,i32)"),
            ),
        ),
        StatefulFilterProtocol(
            StatefulFilterFamily.FAST_FIR,
            "q31",
            "q31_t",
            (
                ("arm_fir_init_q31", "void(ptr,i16,ptr,ptr,i32)"),
                ("arm_fir_fast_q31", "void(ptr,ptr,ptr,i32)"),
            ),
        ),
        *(
            StatefulFilterProtocol(
                StatefulFilterFamily.FIR_LATTICE,
                suffix,
                value_type,
                (
                    (f"arm_fir_lattice_init_{suffix}", "void(ptr,i16,ptr,ptr)"),
                    (f"arm_fir_lattice_{suffix}", "void(ptr,ptr,ptr,i32)"),
                ),
            )
            for suffix, value_type in (
                ("f32", "float32_t"),
                ("q15", "q15_t"),
                ("q31", "q31_t"),
            )
        ),
        *(
            StatefulFilterProtocol(
                StatefulFilterFamily.FIR_SPARSE,
                suffix,
                value_type,
                (
                    (
                        f"arm_fir_sparse_init_{suffix}",
                        "void(ptr,i16,ptr,ptr,ptr,i16,i32)",
                    ),
                    (
                        f"arm_fir_sparse_{suffix}",
                        "void(ptr,ptr,ptr,ptr,i32)"
                        if suffix in {"f32", "q31"}
                        else "void(ptr,ptr,ptr,ptr,ptr,i32)",
                    ),
                ),
            )
            for suffix, value_type in (
                ("f32", "float32_t"),
                ("q15", "q15_t"),
                ("q31", "q31_t"),
                ("q7", "q7_t"),
            )
        ),
        *(
            StatefulFilterProtocol(
                StatefulFilterFamily.IIR_LATTICE,
                suffix,
                value_type,
                (
                    (
                        f"arm_iir_lattice_init_{suffix}",
                        "void(ptr,i16,ptr,ptr,ptr,i32)",
                    ),
                    (f"arm_iir_lattice_{suffix}", "void(ptr,ptr,ptr,i32)"),
                ),
            )
            for suffix, value_type in (
                ("f32", "float32_t"),
                ("q15", "q15_t"),
                ("q31", "q31_t"),
            )
        ),
    )


_STATEFUL_BY_CALLS = {
    protocol.calls: protocol for protocol in _stateful_filter_protocols()
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


def stateful_filter_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> StatefulFilterProtocol | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspGeneratedWorkloadProducer):
        return None
    if (
        workload.suite != "cmsis-dsp"
        or workload.target_profile != corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        or producer.selector_kind != "filter-completion"
    ):
        return None
    calls = tuple((call.symbol, call.signature) for call in workload.protocol)
    protocol = _STATEFUL_BY_CALLS.get(calls)
    if protocol is None:
        return None
    identity = "+".join(symbol for symbol, _ in protocol.calls)
    if workload.vector_identity != f"filter-completion:{identity}:0":
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


_Q15_FILTER_INPUT = (2048, -1024, 3072, 512, -2048, 1536, -512, 2560)
_Q31_FILTER_INPUT = (
    134217728,
    -67108864,
    201326592,
    33554432,
    -134217728,
    100663296,
    -33554432,
    167772160,
)
_F32_FILTER_INPUT = ("0.25f", "-0.125f", "0.375f", "0.0625f", "-0.25f", "0.1875f")


def _filter_input(protocol: StatefulFilterProtocol) -> tuple[int | str, ...]:
    if protocol.suffix == "f32":
        return _F32_FILTER_INPUT
    if protocol.suffix == "q15":
        return _Q15_FILTER_INPUT
    if protocol.suffix == "q31":
        return _Q31_FILTER_INPUT
    if protocol.suffix == "q7":
        return (32, -16, 48, 8, -32, 24, -8, 40)
    raise AssertionError(f"unsupported filter suffix: {protocol.suffix}")


def _cpp_values(values: tuple[int | str, ...]) -> str:
    return ", ".join(str(value) for value in values)


def _render_fast_fir(protocol: StatefulFilterProtocol, protocol_symbol: str) -> str:
    init_symbol, process_symbol = (call[0] for call in protocol.calls)
    baseline_symbol = process_symbol.replace("_fast_", "_")
    inputs = _filter_input(protocol)
    coefficients = (
        (8192, 4096, -2048, 1024)
        if protocol.suffix == "q15"
        else (536870912, 268435456, -134217728, 67108864)
    )
    tolerance = 3 if protocol.suffix == "q15" else 4096
    init_call = f"(void){init_symbol}" if protocol.suffix == "q15" else init_symbol
    instance_type = f"arm_fir_instance_{protocol.suffix}"
    return f"""#include <cstddef>
#include <cstdint>

#include "dsp/filtering_functions.h"

#if defined(__GNUC__) || defined(__clang__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::size_t kBlockSize = {len(inputs)};
constexpr std::size_t kTapCount = {len(coefficients)};
constexpr {protocol.value_type} kInput[] = {{{_cpp_values(inputs)}}};
constexpr {protocol.value_type} kCoefficients[] = {{{_cpp_values(coefficients)}}};
}}

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const {protocol.value_type} *input, const {protocol.value_type} *coefficients,
    {protocol.value_type} *state, {protocol.value_type} *output) {{
  {instance_type} instance{{}};
  {init_call}(&instance, kTapCount, coefficients, state, kBlockSize);
  {process_symbol}(&instance, input, output, kBlockSize);
}}

int main() {{
  {protocol.value_type} state[kBlockSize + kTapCount - 1]{{}};
  {protocol.value_type} reference_state[kBlockSize + kTapCount - 1]{{}};
  {protocol.value_type} output[kBlockSize]{{}};
  {protocol.value_type} reference[kBlockSize]{{}};
  {instance_type} reference_instance{{kTapCount, reference_state, kCoefficients}};
  {baseline_symbol}(&reference_instance, kInput, reference, kBlockSize);
  {protocol_symbol}(kInput, kCoefficients, state, output);
  const auto output_matches_reference = [&]() {{
    for (std::size_t index = 0; index < kBlockSize; ++index) {{
      const std::int64_t difference =
          static_cast<std::int64_t>(output[index]) - reference[index];
      if (difference < -{tolerance} || difference > {tolerance}) return false;
    }}
    return true;
  }};
  return output_matches_reference() ? 0 : 1;
}}
"""


def _render_fast_decimate(
    protocol: StatefulFilterProtocol, protocol_symbol: str
) -> str:
    init_symbol, process_symbol = (call[0] for call in protocol.calls)
    baseline_symbol = process_symbol.replace("_fast_", "_")
    inputs = _filter_input(protocol)
    coefficients = (
        (8192, 4096, -2048, 1024)
        if protocol.suffix == "q15"
        else (536870912, 268435456, -134217728, 67108864)
    )
    tolerance = 3 if protocol.suffix == "q15" else 4096
    instance_type = f"arm_fir_decimate_instance_{protocol.suffix}"
    return f"""#include <cstddef>
#include <cstdint>

#include "dsp/filtering_functions.h"

#if defined(__GNUC__) || defined(__clang__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::size_t kBlockSize = {len(inputs)};
constexpr std::size_t kOutputCount = kBlockSize / 2;
constexpr std::size_t kTapCount = {len(coefficients)};
constexpr {protocol.value_type} kInput[] = {{{_cpp_values(inputs)}}};
constexpr {protocol.value_type} kCoefficients[] = {{{_cpp_values(coefficients)}}};
}}

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const {protocol.value_type} *input, const {protocol.value_type} *coefficients,
    {protocol.value_type} *state, {protocol.value_type} *output) {{
  {instance_type} instance{{}};
  (void){init_symbol}(&instance, kTapCount, 2, coefficients, state, kBlockSize);
  {process_symbol}(&instance, input, output, kBlockSize);
}}

int main() {{
  {protocol.value_type} state[kBlockSize + kTapCount - 1]{{}};
  {protocol.value_type} reference_state[kBlockSize + kTapCount - 1]{{}};
  {protocol.value_type} output[kOutputCount]{{}};
  {protocol.value_type} reference[kOutputCount]{{}};
  {instance_type} reference_instance{{2, kTapCount, kCoefficients, reference_state}};
  {baseline_symbol}(&reference_instance, kInput, reference, kBlockSize);
  {protocol_symbol}(kInput, kCoefficients, state, output);
  const auto output_matches_reference = [&]() {{
    for (std::size_t index = 0; index < kOutputCount; ++index) {{
      const std::int64_t difference =
          static_cast<std::int64_t>(output[index]) - reference[index];
      if (difference < -{tolerance} || difference > {tolerance}) return false;
    }}
    return true;
  }};
  return output_matches_reference() ? 0 : 1;
}}
"""


def _render_fast_biquad(protocol: StatefulFilterProtocol, protocol_symbol: str) -> str:
    init_symbol, process_symbol = (call[0] for call in protocol.calls)
    baseline_symbol = process_symbol.replace("_fast_", "_")
    inputs = _filter_input(protocol)
    if protocol.suffix == "q15":
        coefficients = (8192, 0, 4096, -2048, 2048, -1024)
        tolerance = 4
    else:
        coefficients = (536870912, 268435456, -134217728, 134217728, -67108864)
        tolerance = 8192
    instance_type = f"arm_biquad_casd_df1_inst_{protocol.suffix}"
    return f"""#include <cstddef>
#include <cstdint>

#include "dsp/filtering_functions.h"

#if defined(__GNUC__) || defined(__clang__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::size_t kBlockSize = {len(inputs)};
constexpr {protocol.value_type} kInput[] = {{{_cpp_values(inputs)}}};
constexpr {protocol.value_type} kCoefficients[] = {{{_cpp_values(coefficients)}}};
}}

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const {protocol.value_type} *input, const {protocol.value_type} *coefficients,
    {protocol.value_type} *state, {protocol.value_type} *output) {{
  {instance_type} instance{{}};
  {init_symbol}(&instance, 1, coefficients, state, 0);
  {process_symbol}(&instance, input, output, kBlockSize);
}}

int main() {{
  {protocol.value_type} state[4]{{}};
  {protocol.value_type} reference_state[4]{{}};
  {protocol.value_type} output[kBlockSize]{{}};
  {protocol.value_type} reference[kBlockSize]{{}};
  {instance_type} reference_instance{{1, reference_state, kCoefficients, 0}};
  {baseline_symbol}(&reference_instance, kInput, reference, kBlockSize);
  {protocol_symbol}(kInput, kCoefficients, state, output);
  const auto output_matches_reference = [&]() {{
    for (std::size_t index = 0; index < kBlockSize; ++index) {{
      const std::int64_t difference =
          static_cast<std::int64_t>(output[index]) - reference[index];
      if (difference < -{tolerance} || difference > {tolerance}) return false;
    }}
    return true;
  }};
  return output_matches_reference() ? 0 : 1;
}}
"""


def _half_coefficient(protocol: StatefulFilterProtocol) -> str:
    return {"f32": "0.5f", "q15": "16384", "q31": "1073741824", "q7": "64"}[
        protocol.suffix
    ]


def _quarter_coefficient(protocol: StatefulFilterProtocol) -> str:
    return {"f32": "0.25f", "q15": "8192", "q31": "536870912", "q7": "32"}[
        protocol.suffix
    ]


def _half_expression(protocol: StatefulFilterProtocol, value: str) -> str:
    if protocol.suffix == "f32":
        return f"{value} * 0.5f"
    shift = {"q15": 15, "q31": 31, "q7": 7}[protocol.suffix]
    return f"static_cast<{protocol.value_type}>((static_cast<std::int64_t>({value}) * {_half_coefficient(protocol)}) >> {shift})"


def _two_tap_expression(protocol: StatefulFilterProtocol) -> str:
    coefficient = _quarter_coefficient(protocol)
    if protocol.suffix == "f32":
        return f"kInput[index] * {coefficient} + previous * {coefficient}"
    shift = {"q15": 15, "q31": 31, "q7": 7}[protocol.suffix]
    return (
        f"static_cast<{protocol.value_type}>(("
        f"static_cast<std::int64_t>(kInput[index]) * {coefficient} + "
        f"static_cast<std::int64_t>(previous) * {coefficient}) >> {shift})"
    )


def _render_fir_lattice(protocol: StatefulFilterProtocol, protocol_symbol: str) -> str:
    init_symbol, process_symbol = (call[0] for call in protocol.calls)
    inputs = _filter_input(protocol)
    expected = _half_expression(protocol, "previous")
    return f"""#include <cstddef>
#include <cstdint>

#include "dsp/filtering_functions.h"

#if defined(__GNUC__) || defined(__clang__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::size_t kBlockSize = {len(inputs)};
constexpr {protocol.value_type} kInput[] = {{{_cpp_values(inputs)}}};
constexpr {protocol.value_type} kCoefficients[] = {{{_half_coefficient(protocol)}}};
}}

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const {protocol.value_type} *input, const {protocol.value_type} *coefficients,
    {protocol.value_type} *state, {protocol.value_type} *output) {{
  arm_fir_lattice_instance_{protocol.suffix} instance{{}};
  {init_symbol}(&instance, 1, coefficients, state);
  {process_symbol}(&instance, input, output, kBlockSize);
}}

int main() {{
  {protocol.value_type} state[1]{{}};
  {protocol.value_type} output[kBlockSize]{{}};
  {protocol.value_type} reference[kBlockSize]{{}};
  {protocol.value_type} previous = 0;
  for (std::size_t index = 0; index < kBlockSize; ++index) {{
    reference[index] = kInput[index] + {expected};
    previous = kInput[index];
  }}
  {protocol_symbol}(kInput, kCoefficients, state, output);
  const auto output_matches_reference = [&]() {{
    for (std::size_t index = 0; index < kBlockSize; ++index) {{
      if (output[index] != reference[index]) return false;
    }}
    return true;
  }};
  return output_matches_reference() ? 0 : 1;
}}
"""


def _render_fir_sparse(protocol: StatefulFilterProtocol, protocol_symbol: str) -> str:
    init_symbol, process_symbol = (call[0] for call in protocol.calls)
    inputs = _filter_input(protocol)
    scratch_type = "q31_t" if protocol.suffix in {"q15", "q7"} else None
    scratch_declaration = ""
    scratch_argument = ""
    if scratch_type is not None:
        scratch_declaration = f"  {scratch_type} scratch_output[kBlockSize]{{}};\n"
        scratch_argument = ", scratch_output"
    return f"""#include <cstddef>
#include <cstdint>

#include "dsp/filtering_functions.h"

#if defined(__GNUC__) || defined(__clang__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::size_t kBlockSize = {len(inputs)};
constexpr {protocol.value_type} kInput[] = {{{_cpp_values(inputs)}}};
constexpr {protocol.value_type} kCoefficients[] = {{
    {_quarter_coefficient(protocol)}, {_quarter_coefficient(protocol)}}};
}}

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const {protocol.value_type} *input, const {protocol.value_type} *coefficients,
    {protocol.value_type} *state, std::int32_t *tap_delay,
    {protocol.value_type} *scratch_input, {protocol.value_type} *output) {{
  arm_fir_sparse_instance_{protocol.suffix} instance{{}};
  {init_symbol}(&instance, 2, coefficients, state, tap_delay, 1, kBlockSize);
{scratch_declaration}  {process_symbol}(&instance, input, output, scratch_input{scratch_argument},
                       kBlockSize);
}}

int main() {{
  {protocol.value_type} state[kBlockSize + 1]{{}};
  {protocol.value_type} scratch_input[kBlockSize]{{}};
  {protocol.value_type} output[kBlockSize]{{}};
  {protocol.value_type} reference[kBlockSize]{{}};
  std::int32_t tap_delay[] = {{0, 1}};
  {protocol.value_type} previous = 0;
  for (std::size_t index = 0; index < kBlockSize; ++index) {{
    reference[index] = {_two_tap_expression(protocol)};
    previous = kInput[index];
  }}
  {protocol_symbol}(kInput, kCoefficients, state, tap_delay, scratch_input, output);
  const auto output_matches_reference = [&]() {{
    for (std::size_t index = 0; index < kBlockSize; ++index) {{
      if (output[index] != reference[index]) return false;
    }}
    return true;
  }};
  return output_matches_reference() ? 0 : 1;
}}
"""


def _render_iir_lattice(protocol: StatefulFilterProtocol, protocol_symbol: str) -> str:
    init_symbol, process_symbol = (call[0] for call in protocol.calls)
    inputs = _filter_input(protocol)
    return f"""#include <cstddef>
#include <cstdint>

#include "dsp/filtering_functions.h"

#if defined(__GNUC__) || defined(__clang__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

namespace {{
constexpr std::size_t kBlockSize = {len(inputs)};
constexpr {protocol.value_type} kInput[] = {{{_cpp_values(inputs)}}};
}}

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    const {protocol.value_type} *input, {protocol.value_type} *reflection,
    {protocol.value_type} *ladder, {protocol.value_type} *state,
    {protocol.value_type} *output) {{
  arm_iir_lattice_instance_{protocol.suffix} instance{{}};
  {init_symbol}(&instance, 1, reflection, ladder, state, kBlockSize);
  {process_symbol}(&instance, input, output, kBlockSize);
}}

int main() {{
  {protocol.value_type} reflection[] = {{0}};
  {protocol.value_type} ladder[] = {{0, {_half_coefficient(protocol)}}};
  {protocol.value_type} state[kBlockSize + 1]{{}};
  {protocol.value_type} output[kBlockSize]{{}};
  {protocol.value_type} reference[kBlockSize]{{}};
  for (std::size_t index = 0; index < kBlockSize; ++index) {{
    reference[index] = {_half_expression(protocol, "kInput[index]")};
  }}
  {protocol_symbol}(kInput, reflection, ladder, state, output);
  const auto output_matches_reference = [&]() {{
    for (std::size_t index = 0; index < kBlockSize; ++index) {{
      if (output[index] != reference[index]) return false;
    }}
    return true;
  }};
  return output_matches_reference() ? 0 : 1;
}}
"""


def render_stateful_filter_protocol(
    workload: corpus_inventory.ProgramWorkload,
    protocol_symbol: str,
) -> str:
    protocol = stateful_filter_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no stateful filter provider: {workload.identity}"
        )
    renderers = {
        StatefulFilterFamily.FAST_BIQUAD: _render_fast_biquad,
        StatefulFilterFamily.FAST_DECIMATE: _render_fast_decimate,
        StatefulFilterFamily.FAST_FIR: _render_fast_fir,
        StatefulFilterFamily.FIR_LATTICE: _render_fir_lattice,
        StatefulFilterFamily.FIR_SPARSE: _render_fir_sparse,
        StatefulFilterFamily.IIR_LATTICE: _render_iir_lattice,
    }
    return renderers[protocol.family](protocol, protocol_symbol)
