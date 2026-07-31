#!/usr/bin/env python3
"""Typed generated CMSIS-NN transpose-convolution workloads."""

from __future__ import annotations

from typing import Callable


_ROW_S8_S32_CALL = (
    "arm_nn_transpose_conv_row_s8_s32",
    "arm_cmsis_nn_status (const int8_t *, const int8_t *, int32_t *, "
    "const int32_t, const int32_t, const int32_t, const int32_t, "
    "const int32_t, const int32_t, const int32_t, const int32_t, "
    "const int32_t, const int32_t, const int32_t, const int32_t)",
)


def _render_row_s8_s32(_wrapper_symbol: str) -> str:
    return """#include <stddef.h>
#include <stdint.h>

#include "arm_nnsupportfunctions.h"

enum {
    kInputX = 5,
    kInputChannels = 3,
    kOutputChannels = 2,
    kRhsRows = 1,
    kRhsColumns = 2,
    kInputOffset = 3,
    kStrideX = 1,
    kOutputIndex = 1,
    kOutputMax = 64,
    kRowOffset = 13,
    kOutputCount =
        kOutputIndex + ((kInputX - 1) * kStrideX + kRhsColumns) *
            kOutputChannels,
};

static const int8_t kLhs[kInputX * kInputChannels] = {
    7, -3, 12,
    -5, 9, 2,
    4, -8, 6,
    -1, 11, -4,
    3, 5, -7,
};
static const int8_t kRhs[
    kOutputChannels * kRhsRows * kRhsColumns * kInputChannels] = {
    3, -2, 5, 1, -4, 6,
    -6, 7, 2, -3, 4, -5,
};

int main(void)
{
    int32_t output[kOutputCount];
    int32_t expected[kOutputCount];
    for (size_t index = 0; index < kOutputCount; ++index)
    {
        output[index] = (int32_t)index - 7;
        expected[index] = output[index];
    }

    const arm_cmsis_nn_status status = arm_nn_transpose_conv_row_s8_s32(
        kLhs, kRhs, output, kOutputIndex, kOutputMax, kRhsRows,
        kRhsColumns, kInputChannels, kOutputChannels, kInputOffset,
        kRowOffset, kInputX, kStrideX, 0, 0);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return 1;
    }

    for (size_t input_x = 0; input_x < kInputX; ++input_x)
    {
        for (size_t output_channel = 0;
             output_channel < kOutputChannels;
             ++output_channel)
        {
            for (size_t filter_x = 0; filter_x < kRhsColumns; ++filter_x)
            {
                const size_t spatial_index = input_x * kStrideX + filter_x;
                const size_t output_index =
                    kOutputIndex + spatial_index * kOutputChannels +
                    output_channel;
                for (size_t input_channel = 0;
                     input_channel < kInputChannels;
                     ++input_channel)
                {
                    const int32_t lhs =
                        kLhs[input_x * kInputChannels + input_channel] +
                        kInputOffset;
                    const size_t rhs_index =
                        (output_channel * kRhsColumns + filter_x) *
                            kInputChannels +
                        input_channel;
                    expected[output_index] += lhs * kRhs[rhs_index];
                }
            }
        }
    }

    for (size_t index = 0; index < kOutputCount; ++index)
    {
        if (output[index] != expected[index])
        {
            return 1;
        }
    }
    return 0;
}
"""


def renderer_for(call: tuple[str, str]) -> Callable[[str], str] | None:
    if call == _ROW_S8_S32_CALL:
        return _render_row_s8_s32
    return None
