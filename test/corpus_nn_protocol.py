#!/usr/bin/env python3
"""Typed generated-public CMSIS-NN workload protocols."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


@dataclass(frozen=True)
class GeneratedCmsisNnProtocol:
    source: str
    protocol_symbol: str
    compiled_owner: Path | None
    authoritative_owner: Path


def _render_in_place_activation(
    symbol: str,
    element_type: str,
    values: str,
    expected: str,
) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

enum {{ kElementCount = 17 }};

static const {element_type} kInput[kElementCount] = {{
    {values},
}};

int main(void)
{{
    {element_type} data[kElementCount];
    for (size_t index = 0; index < kElementCount; ++index)
    {{
        data[index] = kInput[index];
    }}

    {symbol}(data, kElementCount);

    for (size_t index = 0; index < kElementCount; ++index)
    {{
        const {element_type} expected = {expected};
        if (data[index] != expected)
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""


def _render_relu_q7(_wrapper_symbol: str) -> str:
    return _render_in_place_activation(
        "arm_relu_q7",
        "int8_t",
        "-128, -31, -7, -1, 0, 1, 2, 6, 7, 15, 31, 63, 95, 126, 127, -64, -2",
        "kInput[index] < 0 ? 0 : kInput[index]",
    )


def _render_relu_q15(_wrapper_symbol: str) -> str:
    return _render_in_place_activation(
        "arm_relu_q15",
        "int16_t",
        "-32768, -16384, -1025, -1, 0, 1, 2, 6, 7, "
        "255, 1024, 8191, 16384, 32766, 32767, -4096, -2",
        "kInput[index] < 0 ? 0 : kInput[index]",
    )


def _render_relu6_s8(_wrapper_symbol: str) -> str:
    return _render_in_place_activation(
        "arm_relu6_s8",
        "int8_t",
        "-128, -31, -7, -1, 0, 1, 2, 5, 6, 7, 15, 31, 63, 126, 127, -64, -2",
        "kInput[index] < 0 ? 0 : (kInput[index] > 6 ? 6 : kInput[index])",
    )


def _render_reshape_s8(_wrapper_symbol: str) -> str:
    return """#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

enum { kElementCount = 33 };

static const int8_t kInput[kElementCount] = {
    -128, -97, -64, -33, -16, -8, -4, -2, -1, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 15, 16, 23, 31, 32, 47, 63,
    64, 79, 95, 111, 126, 127, -55, 42,
};

int main(void)
{
    int8_t output[kElementCount] = {0};
    arm_reshape_s8(kInput, output, kElementCount);
    for (size_t index = 0; index < kElementCount; ++index)
    {
        if (output[index] != kInput[index])
        {
            return 1;
        }
    }
    return 0;
}
"""


def _render_q7_to_q15_with_offset(_wrapper_symbol: str) -> str:
    return """#include <stddef.h>
#include <stdint.h>

#include "arm_nnsupportfunctions.h"

enum { kElementCount = 33 };
static const int16_t kOffset = 257;

static const int8_t kInput[kElementCount] = {
    -128, -97, -64, -33, -16, -8, -4, -2, -1, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 15, 16, 23, 31, 32, 47, 63,
    64, 79, 95, 111, 126, 127, -55, 42,
};

int main(void)
{
    int16_t output[kElementCount] = {0};
    arm_q7_to_q15_with_offset(
        kInput, output, kElementCount, kOffset);
    for (size_t index = 0; index < kElementCount; ++index)
    {
        const int16_t expected = (int16_t)kInput[index] + kOffset;
        if (output[index] != expected)
        {
            return 1;
        }
    }
    return 0;
}
"""


def _render_header_memory(
    wrapper_symbol: str,
    operation: str,
    element_type: str,
    input_values: str,
) -> str:
    is_set = operation == "arm_memset_s8"
    wrapper_parameters = (
        f"{element_type} *output, const int8_t value, uint32_t byte_count"
        if is_set
        else f"{element_type} *output, const {element_type} *input, uint32_t byte_count"
    )
    operation_arguments = (
        "output, value, byte_count" if is_set else "output, input, byte_count"
    )
    main_arguments = (
        "output, kFillValue, kByteCount" if is_set else "output, kInput, kByteCount"
    )
    expected = "kFillValue" if is_set else "kInput[index]"
    input_declaration = ""
    if not is_set:
        input_declaration = f"""static const {element_type} kInput[kElementCount] = {{
    {input_values},
}};

"""
    return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnsupportfunctions.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

enum {{ kElementCount = 33 }};
enum {{ kByteCount = kElementCount * sizeof({element_type}) }};
static const int8_t kFillValue = -37;

{input_declaration}LOOM_NOINLINE void {wrapper_symbol}({wrapper_parameters})
{{
    {operation}({operation_arguments});
}}

int main(void)
{{
    {element_type} output[kElementCount] = {{0}};
    {wrapper_symbol}({main_arguments});
    for (size_t index = 0; index < kElementCount; ++index)
    {{
        if (output[index] != {expected})
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""


def _render_memcpy_s8(wrapper_symbol: str) -> str:
    return _render_header_memory(
        wrapper_symbol,
        "arm_memcpy_s8",
        "int8_t",
        "-128, -97, -64, -33, -16, -8, -4, -2, -1, 0, 1, "
        "2, 3, 4, 5, 6, 7, 8, 15, 16, 23, 31, 32, 47, 63, "
        "64, 79, 95, 111, 126, 127, -55, 42",
    )


def _render_memcpy_q15(wrapper_symbol: str) -> str:
    return _render_header_memory(
        wrapper_symbol,
        "arm_memcpy_q15",
        "int16_t",
        "-32768, -30001, -16384, -8193, -1024, -255, -16, -2, -1, 0, 1, "
        "2, 3, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 4096, "
        "8191, 16384, 24575, 30000, 32766, 32767, -4096, 777",
    )


def _render_memset_s8(wrapper_symbol: str) -> str:
    return _render_header_memory(
        wrapper_symbol,
        "arm_memset_s8",
        "int8_t",
        "",
    )


def _render_concatenation(_wrapper_symbol: str, axis: str) -> str:
    output_dimension = {
        "x": "kOutputX",
        "y": "kOutputY",
        "z": "kOutputZ",
    }.get(axis)
    extra_argument = "" if axis == "w" else f",\n        {output_dimension}"
    destination = {
        "x": "(((w * kInputZ + z) * kInputY + y) * kOutputX + (x + kOffset))",
        "y": "(((w * kInputZ + z) * kOutputY + (y + kOffset)) * kInputX + x)",
        "z": "(((w * kOutputZ + (z + kOffset)) * kInputY + y) * kInputX + x)",
        "w": "((((w + kOffset) * kInputZ + z) * kInputY + y) * kInputX + x)",
    }[axis]
    output_count = {
        "x": "kOutputX * kInputY * kInputZ * kInputW",
        "y": "kInputX * kOutputY * kInputZ * kInputW",
        "z": "kInputX * kInputY * kOutputZ * kInputW",
        "w": "kInputX * kInputY * kInputZ * kOutputW",
    }[axis]
    return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

enum {{
    kInputX = 3,
    kInputY = 2,
    kInputZ = 2,
    kInputW = 2,
    kOutputX = 5,
    kOutputY = 4,
    kOutputZ = 4,
    kOutputW = 4,
    kOffset = 1,
    kInputCount = kInputX * kInputY * kInputZ * kInputW,
    kOutputCount = {output_count},
}};

int main(void)
{{
    int8_t input[kInputCount];
    int8_t output[kOutputCount];
    int8_t expected[kOutputCount];
    for (size_t source = 0; source < kInputCount; ++source)
    {{
        input[source] = (int8_t)((source * 17 + 11) % 127 - 63);
    }}
    for (size_t index = 0; index < kOutputCount; ++index)
    {{
        output[index] = -101;
        expected[index] = -101;
    }}

    arm_concatenation_s8_{axis}(
        input, kInputX, kInputY, kInputZ, kInputW, output{extra_argument},
        kOffset);

    for (size_t w = 0; w < kInputW; ++w)
    {{
        for (size_t z = 0; z < kInputZ; ++z)
        {{
            for (size_t y = 0; y < kInputY; ++y)
            {{
                for (size_t x = 0; x < kInputX; ++x)
                {{
                    const size_t source =
                        (((w * kInputZ + z) * kInputY + y) * kInputX + x);
                    const size_t destination = {destination};
                    expected[destination] = input[source];
                }}
            }}
        }}
    }}
    for (size_t index = 0; index < kOutputCount; ++index)
    {{
        if (output[index] != expected[index])
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""


def _render_concatenation_w(wrapper_symbol: str) -> str:
    return _render_concatenation(wrapper_symbol, "w")


def _render_concatenation_x(wrapper_symbol: str) -> str:
    return _render_concatenation(wrapper_symbol, "x")


def _render_concatenation_y(wrapper_symbol: str) -> str:
    return _render_concatenation(wrapper_symbol, "y")


def _render_concatenation_z(wrapper_symbol: str) -> str:
    return _render_concatenation(wrapper_symbol, "z")


def _query_renderer(
    symbol: str,
    arguments: str,
    expected: str,
) -> Callable[[str], str]:
    def render(_wrapper_symbol: str) -> str:
        return f"""#include <stdint.h>

#include "arm_nnfunctions.h"

enum {{ kOutputWidth = 7, kChannels = 13 }};
static const cmsis_nn_dims kDimensions = {{
    .n = 7,
    .h = 5,
    .w = 3,
    .c = 11,
}};

int main(void)
{{
    const int32_t result = {symbol}({arguments});
    const int32_t expected = {expected};
    return result == expected ? 0 : 1;
}}
"""

    return render


def _render_elementwise_mul_batch(
    symbol: str,
    output_type: str,
    activation_min: int,
    activation_max: int,
) -> Callable[[str], str]:
    def render(_wrapper_symbol: str) -> str:
        return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

enum {{
    kBlockSize = 17,
    kBatchSize = 3,
    kBatchOffset = 2,
    kInputCount = kBlockSize * kBatchSize,
    kOutputCount = kBlockSize * (1 + (kBatchSize - 1) * kBatchOffset),
    kOutputOffset = 7,
    kMultiplier = 1 << 30,
    kShift = 1,
}};

int main(void)
{{
    int16_t input_1[kInputCount];
    int16_t input_2[kInputCount];
    {output_type} output[kOutputCount];
    {output_type} expected[kOutputCount];
    for (size_t index = 0; index < kInputCount; ++index)
    {{
        input_1[index] = (int16_t)((index * 37 % 257) - 128);
        input_2[index] = (int16_t)((index * 19 % 127) - 63);
    }}
    for (size_t index = 0; index < kOutputCount; ++index)
    {{
        output[index] = ({output_type})-101;
        expected[index] = ({output_type})-101;
    }}

    const arm_cmsis_nn_status status = {symbol}(
        input_1, input_2, output, kOutputOffset, kMultiplier, kShift,
        kBlockSize, kBatchSize, kBatchOffset);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {{
        return 1;
    }}

    for (size_t batch = 0; batch < kBatchSize; ++batch)
    {{
        for (size_t lane = 0; lane < kBlockSize; ++lane)
        {{
            const size_t index = batch * kBlockSize + lane;
            const size_t destination = batch * kBatchOffset * kBlockSize + lane;
            int32_t reference = input_1[index] * input_2[index] + kOutputOffset;
            if (reference < {activation_min})
            {{
                reference = {activation_min};
            }}
            if (reference > {activation_max})
            {{
                reference = {activation_max};
            }}
            expected[destination] = ({output_type})reference;
        }}
    }}
    for (size_t index = 0; index < kOutputCount; ++index)
    {{
        if (output[index] != expected[index])
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""

    return render


def _render_elementwise_mul_acc_s16(_wrapper_symbol: str) -> str:
    return """#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

enum {
    kElementCount = 33,
    kMultiplier = 1 << 30,
    kShift = 1,
    kActivationMin = -12000,
    kActivationMax = 12000,
};

int main(void)
{
    int16_t input_1[kElementCount];
    int16_t input_2[kElementCount];
    int16_t initial_output[kElementCount];
    int16_t output[kElementCount];
    for (size_t index = 0; index < kElementCount; ++index)
    {
        input_1[index] = (int16_t)((index * 37 % 257) - 128);
        input_2[index] = (int16_t)((index * 19 % 127) - 63);
        initial_output[index] = (int16_t)((index * 23 % 401) - 200);
        output[index] = initial_output[index];
    }

    const arm_cmsis_nn_status status = arm_elementwise_mul_acc_s16(
        input_1, input_2, 0, 0, output, 0, kMultiplier, kShift,
        kActivationMin, kActivationMax, kElementCount);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return 1;
    }

    for (size_t index = 0; index < kElementCount; ++index)
    {
        int32_t reference = initial_output[index] + input_1[index] * input_2[index];
        if (reference < kActivationMin)
        {
            reference = kActivationMin;
        }
        if (reference > kActivationMax)
        {
            reference = kActivationMax;
        }
        if (output[index] != (int16_t)reference)
        {
            return 1;
        }
    }
    return 0;
}
"""


def _header_scalar_renderer(
    symbol: str,
    parameters: str,
    fields: str,
    wrapper_arguments: str,
    vector_arguments: str,
    vectors: str,
) -> Callable[[str], str]:
    def render(wrapper_symbol: str) -> str:
        return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnsupportfunctions.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

typedef struct
{{
{fields}
    int32_t expected;
}} TestVector;

static const TestVector kVectors[] = {{
{vectors}
}};

LOOM_NOINLINE int32_t {wrapper_symbol}({parameters})
{{
    return {symbol}({wrapper_arguments});
}}

int main(void)
{{
    const size_t vector_count = sizeof(kVectors) / sizeof(kVectors[0]);
    for (size_t index = 0; index < vector_count; ++index)
    {{
        const TestVector *vector = &kVectors[index];
        if ({wrapper_symbol}({vector_arguments}) != vector->expected)
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""

    return render


def _header_packed_read_renderer(
    symbol: str,
    element_type: str,
    unsigned_type: str,
    lane_count: int,
    incrementing: bool,
) -> Callable[[str], str]:
    bit_width = 8 if element_type == "int8_t" else 16
    initializer = (
        "(int8_t)((int32_t)((index * 37) % 251) - 125)"
        if element_type == "int8_t"
        else "(int16_t)((int32_t)((index * 733) % 65521) - 32760)"
    )

    def render(wrapper_symbol: str) -> str:
        wrapper_parameter = (
            f"const {element_type} **input"
            if incrementing
            else f"const {element_type} *input"
        )
        main_setup = (
            f"    const {element_type} *cursor = input;\n" if incrementing else ""
        )
        main_call = (
            f"{wrapper_symbol}(&cursor)"
            if incrementing
            else f"{wrapper_symbol}(expected_cursor)"
        )
        cursor_check = (
            """        if (cursor != expected_cursor)
        {
            return 1;
        }
"""
            if incrementing
            else ""
        )
        return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnsupportfunctions.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

enum {{ kGroupCount = 9, kLaneCount = {lane_count} }};

LOOM_NOINLINE int32_t {wrapper_symbol}({wrapper_parameter})
{{
    return {symbol}(input);
}}

int main(void)
{{
    {element_type} input[kGroupCount * kLaneCount];
    for (size_t index = 0; index < kGroupCount * kLaneCount; ++index)
    {{
        input[index] = {initializer};
    }}
{main_setup}
    for (size_t group = 0; group < kGroupCount; ++group)
    {{
        const {element_type} *expected_cursor =
            input + (group + {1 if incrementing else 0}) * kLaneCount;
        const uint32_t actual = (uint32_t){main_call};
{cursor_check}        uint32_t expected = 0;
        for (size_t lane = 0; lane < kLaneCount; ++lane)
        {{
            const size_t source = group * kLaneCount + lane;
            expected |= (uint32_t)({unsigned_type})input[source]
                        << ({bit_width} * lane);
        }}
        if (actual != expected)
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""

    return render


def _header_packed_write_renderer(
    symbol: str,
    element_type: str,
    unsigned_type: str,
    lane_count: int,
) -> Callable[[str], str]:
    bit_width = 8 if element_type == "int8_t" else 16
    mask = "UINT8_MAX" if element_type == "int8_t" else "UINT16_MAX"

    def render(wrapper_symbol: str) -> str:
        return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnsupportfunctions.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

enum {{ kGroupCount = 9, kLaneCount = {lane_count} }};

static const uint32_t kValues[kGroupCount] = {{
    0x00000000u, 0x01020304u, 0x7f80ff00u,
    0x89abcdefu, 0xffffffffu, 0x13579bdfu,
    0x2468ace0u, 0x80000001u, 0x55aa33ccu,
}};

LOOM_NOINLINE void {wrapper_symbol}({element_type} **output, int32_t value)
{{
    {symbol}(output, value);
}}

int main(void)
{{
    {element_type} output[kGroupCount * kLaneCount] = {{0}};
    {element_type} *cursor = output;
    for (size_t group = 0; group < kGroupCount; ++group)
    {{
        {wrapper_symbol}(&cursor, (int32_t)kValues[group]);
        const {element_type} *expected_cursor =
            output + (group + 1) * kLaneCount;
        if (cursor != expected_cursor)
        {{
            return 1;
        }}
    }}
    for (size_t group = 0; group < kGroupCount; ++group)
    {{
        for (size_t lane = 0; lane < kLaneCount; ++lane)
        {{
            const {unsigned_type} expected = ({unsigned_type})(
                (kValues[group] >> ({bit_width} * lane)) & {mask});
            if (({unsigned_type})output[group * kLaneCount + lane] != expected)
            {{
                return 1;
            }}
        }}
    }}
    return 0;
}}
"""

    return render


_RENDERERS: dict[tuple[str, str], Callable[[str], str]] = {
    ("arm_relu_q7", "void (int8_t *, uint16_t)"): _render_relu_q7,
    ("arm_relu_q15", "void (int16_t *, uint16_t)"): _render_relu_q15,
    ("arm_relu6_s8", "void (int8_t *, uint16_t)"): _render_relu6_s8,
    (
        "arm_reshape_s8",
        "void (const int8_t *, int8_t *, const uint32_t)",
    ): _render_reshape_s8,
    (
        "arm_q7_to_q15_with_offset",
        "void (const int8_t *, int16_t *, int32_t, int16_t)",
    ): _render_q7_to_q15_with_offset,
    (
        "arm_memcpy_s8",
        "void (int8_t *restrict, const int8_t *restrict, uint32_t)",
    ): _render_memcpy_s8,
    (
        "arm_memcpy_q15",
        "void (int16_t *restrict, const int16_t *restrict, uint32_t)",
    ): _render_memcpy_q15,
    (
        "arm_memset_s8",
        "void (int8_t *, const int8_t, uint32_t)",
    ): _render_memset_s8,
    (
        "arm_concatenation_s8_w",
        "void (const int8_t *, const uint16_t, const uint16_t, const uint16_t, "
        "const uint16_t, int8_t *, const uint32_t)",
    ): _render_concatenation_w,
    (
        "arm_concatenation_s8_x",
        "void (const int8_t *, const uint16_t, const uint16_t, const uint16_t, "
        "const uint16_t, int8_t *, const uint16_t, const uint32_t)",
    ): _render_concatenation_x,
    (
        "arm_concatenation_s8_y",
        "void (const int8_t *, const uint16_t, const uint16_t, const uint16_t, "
        "const uint16_t, int8_t *, const uint16_t, const uint32_t)",
    ): _render_concatenation_y,
    (
        "arm_concatenation_s8_z",
        "void (const int8_t *, const uint16_t, const uint16_t, const uint16_t, "
        "const uint16_t, int8_t *, const uint16_t, const uint32_t)",
    ): _render_concatenation_z,
    (
        "arm_avgpool_s8_get_buffer_size_dsp",
        "int32_t (const int, const int)",
    ): _query_renderer(
        "arm_avgpool_s8_get_buffer_size_dsp",
        "kOutputWidth, kChannels",
        "kChannels * sizeof(int32_t)",
    ),
    (
        "arm_avgpool_s8_get_buffer_size_mve",
        "int32_t (const int, const int)",
    ): _query_renderer(
        "arm_avgpool_s8_get_buffer_size_mve",
        "kOutputWidth, kChannels",
        "0",
    ),
    (
        "arm_avgpool_s16_get_buffer_size_dsp",
        "int32_t (const int, const int)",
    ): _query_renderer(
        "arm_avgpool_s16_get_buffer_size_dsp",
        "kOutputWidth, kChannels",
        "kChannels * sizeof(int32_t)",
    ),
    (
        "arm_avgpool_s16_get_buffer_size_mve",
        "int32_t (const int, const int)",
    ): _query_renderer(
        "arm_avgpool_s16_get_buffer_size_mve",
        "kOutputWidth, kChannels",
        "0",
    ),
    (
        "arm_fully_connected_s8_get_buffer_size_dsp",
        "int32_t (const cmsis_nn_dims *)",
    ): _query_renderer(
        "arm_fully_connected_s8_get_buffer_size_dsp",
        "&kDimensions",
        "0",
    ),
    (
        "arm_fully_connected_s8_get_buffer_size_mve",
        "int32_t (const cmsis_nn_dims *)",
    ): _query_renderer(
        "arm_fully_connected_s8_get_buffer_size_mve",
        "&kDimensions",
        "kDimensions.c * sizeof(int32_t)",
    ),
    (
        "arm_fully_connected_s16_get_buffer_size_dsp",
        "int32_t (const cmsis_nn_dims *)",
    ): _query_renderer(
        "arm_fully_connected_s16_get_buffer_size_dsp",
        "&kDimensions",
        "0",
    ),
    (
        "arm_fully_connected_s16_get_buffer_size_mve",
        "int32_t (const cmsis_nn_dims *)",
    ): _query_renderer(
        "arm_fully_connected_s16_get_buffer_size_mve",
        "&kDimensions",
        "0",
    ),
    (
        "arm_svdf_s8_get_buffer_size_dsp",
        "int32_t (const cmsis_nn_dims *)",
    ): _query_renderer(
        "arm_svdf_s8_get_buffer_size_dsp",
        "&kDimensions",
        "0",
    ),
    (
        "arm_svdf_s8_get_buffer_size_mve",
        "int32_t (const cmsis_nn_dims *)",
    ): _query_renderer(
        "arm_svdf_s8_get_buffer_size_mve",
        "&kDimensions",
        "kDimensions.n * sizeof(int32_t)",
    ),
    (
        "arm_elementwise_mul_s16_batch_offset",
        "arm_cmsis_nn_status (const int16_t *, const int16_t *, int16_t *, "
        "const int32_t, const int32_t, const int32_t, const int32_t, "
        "const int32_t, const int32_t)",
    ): _render_elementwise_mul_batch(
        "arm_elementwise_mul_s16_batch_offset",
        "int16_t",
        -32768,
        32767,
    ),
    (
        "arm_elementwise_mul_s16_s8",
        "arm_cmsis_nn_status (const int16_t *, const int16_t *, int8_t *, "
        "const int32_t, const int32_t, const int32_t, const int32_t, "
        "const int32_t, const int32_t)",
    ): _render_elementwise_mul_batch(
        "arm_elementwise_mul_s16_s8",
        "int8_t",
        -128,
        127,
    ),
    (
        "arm_elementwise_mul_acc_s16",
        "arm_cmsis_nn_status (const int16_t *, const int16_t *, const int32_t, "
        "const int32_t, int16_t *, const int32_t, const int32_t, "
        "const int32_t, const int32_t, const int32_t, const int32_t)",
    ): _render_elementwise_mul_acc_s16,
    (
        "arm_nn_doubling_high_mult",
        "int32_t (const int32_t, const int32_t)",
    ): _header_scalar_renderer(
        "arm_nn_doubling_high_mult",
        "int32_t lhs, int32_t rhs",
        "    int32_t lhs;\n    int32_t rhs;\n",
        "lhs, rhs",
        "vector->lhs, vector->rhs",
        """    {0, 123456789, 0},
    {1073741824, 1073741824, 536870912},
    {-1073741824, 1073741824, -536870912},
    {INT32_MIN, INT32_MIN, 2147483647},
    {123456789, 987654321, 56779306},
    {-123456789, 987654321, -56779306},
    {INT32_MAX, INT32_MAX, 2147483646},""",
    ),
    (
        "arm_nn_doubling_high_mult_no_sat",
        "int32_t (int32_t, int32_t)",
    ): _header_scalar_renderer(
        "arm_nn_doubling_high_mult_no_sat",
        "int32_t lhs, int32_t rhs",
        "    int32_t lhs;\n    int32_t rhs;\n",
        "lhs, rhs",
        "vector->lhs, vector->rhs",
        """    {0, 123456789, 0},
    {1073741824, 1073741824, 536870912},
    {-1073741824, 1073741824, -536870912},
    {123456789, 987654321, 56779306},
    {-123456789, 987654321, -56779306},
    {INT32_MAX, INT32_MAX, 2147483646},""",
    ),
    (
        "arm_nn_divide_by_power_of_two",
        "int32_t (const int32_t, const int32_t)",
    ): _header_scalar_renderer(
        "arm_nn_divide_by_power_of_two",
        "int32_t value, int32_t exponent",
        "    int32_t value;\n    int32_t exponent;\n",
        "value, exponent",
        "vector->value, vector->exponent",
        """    {-17, 2, -4},
    {-7, 1, -4},
    {-5, 1, -3},
    {-3, 1, -2},
    {-1, 1, -1},
    {0, 7, 0},
    {1, 1, 1},
    {3, 1, 2},
    {5, 1, 3},
    {7, 1, 4},
    {17, 2, 4},
    {123456789, 7, 964506},""",
    ),
    (
        "arm_nn_requantize",
        "int32_t (const int32_t, const int32_t, const int32_t)",
    ): _header_scalar_renderer(
        "arm_nn_requantize",
        "int32_t value, int32_t multiplier, int32_t shift",
        "    int32_t value;\n    int32_t multiplier;\n    int32_t shift;\n",
        "value, multiplier, shift",
        "vector->value, vector->multiplier, vector->shift",
        """    {12345, 1073741824, 1, 12345},
    {-12345, 1073741824, 1, -12345},
    {12345, 1073741824, 0, 6173},
    {-12345, 1073741824, 0, -6172},
    {12345, 1073741824, -1, 3087},
    {-12345, 1073741824, -1, -3086},
    {7654321, 123456789, 3, 3520317},
    {-7654321, 123456789, -4, -27503},""",
    ),
    (
        "arm_nn_requantize_s64",
        "int32_t (const int64_t, const int32_t, const int32_t)",
    ): _header_scalar_renderer(
        "arm_nn_requantize_s64",
        "int64_t value, int32_t multiplier, int32_t shift",
        "    int64_t value;\n    int32_t multiplier;\n    int32_t shift;\n",
        "value, multiplier, shift",
        "vector->value, vector->multiplier, vector->shift",
        """    {0, 32767, 0, 0},
    {123456789, 12345, 0, 46511049},
    {-123456789, 12345, 0, -46511049},
    {123456789, 16384, -3, 7716049},
    {-123456789, 16384, -3, -7716049},
    {987654321, 32767, -4, 61726511},
    {-987654321, 32767, -4, -61726511},
    {1048576, 32767, 7, 134213632},
    {-1048576, 32767, 7, -134213632},""",
    ),
    (
        "arm_nn_read_q15x2_ia",
        "int32_t (const int16_t **)",
    ): _header_packed_read_renderer(
        "arm_nn_read_q15x2_ia", "int16_t", "uint16_t", 2, True
    ),
    (
        "arm_nn_read_s16x2",
        "int32_t (const int16_t *)",
    ): _header_packed_read_renderer(
        "arm_nn_read_s16x2", "int16_t", "uint16_t", 2, False
    ),
    (
        "arm_nn_read_s8x4_ia",
        "int32_t (const int8_t **)",
    ): _header_packed_read_renderer(
        "arm_nn_read_s8x4_ia", "int8_t", "uint8_t", 4, True
    ),
    (
        "arm_nn_read_s8x4",
        "int32_t (const int8_t *)",
    ): _header_packed_read_renderer("arm_nn_read_s8x4", "int8_t", "uint8_t", 4, False),
    (
        "arm_nn_write_q15x2_ia",
        "void (int16_t **, int32_t)",
    ): _header_packed_write_renderer("arm_nn_write_q15x2_ia", "int16_t", "uint16_t", 2),
    (
        "arm_nn_write_s8x4_ia",
        "void (int8_t **, int32_t)",
    ): _header_packed_write_renderer("arm_nn_write_s8x4_ia", "int8_t", "uint8_t", 4),
}


_HEADER_ONLY_PROTOCOLS = {
    "arm_nn_divide_by_power_of_two",
    "arm_nn_doubling_high_mult",
    "arm_nn_doubling_high_mult_no_sat",
    "arm_nn_requantize",
    "arm_nn_requantize_s64",
    "arm_nn_read_q15x2_ia",
    "arm_nn_read_s16x2",
    "arm_nn_read_s8x4",
    "arm_nn_read_s8x4_ia",
    "arm_nn_write_q15x2_ia",
    "arm_nn_write_s8x4_ia",
    "arm_memcpy_s8",
    "arm_memcpy_q15",
    "arm_memset_s8",
}


def _owned_path(raw_path: str, external_root: Path) -> Path:
    path = Path(raw_path)
    try:
        relative = path.relative_to(Path("externals/cmsis-nn"))
    except ValueError as exc:
        raise WorkloadProviderError(
            f"generated CMSIS-NN owner escapes its source root: {path}"
        ) from exc
    resolved = external_root / "cmsis-nn" / relative
    if not resolved.is_file():
        raise WorkloadProviderError(f"generated CMSIS-NN owner is unavailable: {path}")
    return resolved


def _declaration_owner(
    producer: corpus_inventory.CmsisNnGeneratedWorkloadProducer,
    external_root: Path,
) -> Path:
    declaration = re.compile(
        rf"\b{re.escape(producer.public_symbol)}\s*\([^;{{}}]*\)\s*(?:;|{{)",
        re.DOTALL,
    )
    owners = [
        owner
        for owner in (
            _owned_path(definition, external_root)
            for definition in producer.definitions
        )
        if declaration.search(owner.read_text(encoding="utf-8"))
    ]
    if len(owners) != 1:
        raise WorkloadProviderError(
            "generated CMSIS-NN protocol must resolve one public declaration: "
            f"{producer.public_symbol}"
        )
    return owners[0]


def render_generated_cmsis_nn_protocol(
    workload: corpus_inventory.ProgramWorkload,
    external_root: Path,
    wrapper_symbol: str,
) -> GeneratedCmsisNnProtocol:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisNnGeneratedWorkloadProducer):
        raise WorkloadProviderError(
            f"workload is not a generated CMSIS-NN protocol: {workload.identity}"
        )
    renderer = _renderer_for(workload)
    if renderer is None:
        raise WorkloadProviderError(
            f"generated CMSIS-NN protocol is unsupported: {workload.identity}"
        )
    header_only = producer.public_symbol in _HEADER_ONLY_PROTOCOLS
    expected_sources = 0 if header_only else 1
    if len(workload.sources) != expected_sources:
        raise WorkloadProviderError(
            "generated CMSIS-NN protocol has an invalid implementation closure: "
            f"{workload.identity}"
        )
    compiled_owner = (
        None if header_only else _owned_path(workload.sources[0], external_root)
    )
    return GeneratedCmsisNnProtocol(
        source=renderer(wrapper_symbol),
        protocol_symbol=wrapper_symbol if header_only else producer.public_symbol,
        compiled_owner=compiled_owner,
        authoritative_owner=_declaration_owner(producer, external_root),
    )


def _renderer_for(
    workload: corpus_inventory.ProgramWorkload,
) -> Callable[[str], str] | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisNnGeneratedWorkloadProducer):
        return None
    calls = tuple((call.symbol, call.signature) for call in workload.protocol)
    if (
        len(calls) != 1
        or producer.public_symbol != calls[0][0]
        or workload.target_profile != corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        or workload.oracle.kind != "generated-native-reference"
        or workload.oracle.path != producer.public_symbol
    ):
        return None
    return _RENDERERS.get(calls[0])


def supports_generated_cmsis_nn_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> bool:
    return _renderer_for(workload) is not None
