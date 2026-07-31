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
_TRANSPOSE_S8_CALL = (
    "arm_transpose_conv_s8",
    "arm_cmsis_nn_status (const cmsis_nn_context *, "
    "const cmsis_nn_context *, const cmsis_nn_transpose_conv_params *, "
    "const cmsis_nn_per_channel_quant_params *, const cmsis_nn_dims *, "
    "const int8_t *, const cmsis_nn_dims *, const int8_t *, "
    "const cmsis_nn_dims *, const int32_t *, const cmsis_nn_dims *, "
    "int8_t *)",
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


def _render_transpose_s8(_wrapper_symbol: str) -> str:
    return """#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

enum {
    kInputHeight = 2,
    kInputWidth = 3,
    kInputChannels = 2,
    kOutputHeight = 3,
    kOutputWidth = 4,
    kOutputChannels = 2,
    kFilterHeight = 2,
    kFilterWidth = 2,
    kInputOffset = 3,
    kOutputOffset = 5,
    kMultiplier = 1 << 30,
    kShift = 1,
    kActivationMin = -120,
    kActivationMax = 120,
    kOutputCount = kOutputHeight * kOutputWidth * kOutputChannels,
    kScratchElements =
        kFilterHeight * kOutputWidth * kOutputChannels,
};

static const int8_t kInput[
    kInputHeight * kInputWidth * kInputChannels] = {
    7, -3, -5, 9, 2, 4,
    -8, 6, -1, 11, 3, -7,
};
static const int8_t kFilter[
    kOutputChannels * kFilterHeight * kFilterWidth * kInputChannels] = {
    3, -2, 5, 1, -4, 6, 2, -3,
    -6, 7, 1, -5, 4, -2, 3, 5,
};
static const int32_t kBias[kOutputChannels] = {11, -23};
static int32_t kMultipliers[kOutputChannels] = {
    kMultiplier, kMultiplier,
};
static int32_t kShifts[kOutputChannels] = {kShift, kShift};

int main(void)
{
    int32_t scratch[kScratchElements];
    int8_t output[kOutputCount] = {0};
    int32_t expected[kOutputCount];
    for (size_t index = 0; index < kOutputCount; ++index)
    {
        expected[index] = kBias[index % kOutputChannels];
    }

    cmsis_nn_context context = {
        .buf = scratch,
        .size = sizeof(scratch),
    };
    const cmsis_nn_transpose_conv_params parameters = {
        .input_offset = kInputOffset,
        .output_offset = kOutputOffset,
        .stride = {.w = 1, .h = 1},
        .padding = {.w = 0, .h = 0},
        .padding_offsets = {.w = 0, .h = 0},
        .dilation = {.w = 1, .h = 1},
        .activation = {.min = kActivationMin, .max = kActivationMax},
    };
    const cmsis_nn_per_channel_quant_params quantization = {
        .multiplier = kMultipliers,
        .shift = kShifts,
    };
    const cmsis_nn_dims input_dimensions = {
        .n = 1, .h = kInputHeight, .w = kInputWidth, .c = kInputChannels,
    };
    const cmsis_nn_dims filter_dimensions = {
        .n = kOutputChannels,
        .h = kFilterHeight,
        .w = kFilterWidth,
        .c = kInputChannels,
    };
    const cmsis_nn_dims bias_dimensions = {0};
    const cmsis_nn_dims output_dimensions = {
        .n = 1,
        .h = kOutputHeight,
        .w = kOutputWidth,
        .c = kOutputChannels,
    };

    const arm_cmsis_nn_status status = arm_transpose_conv_s8(
        &context, NULL, &parameters, &quantization, &input_dimensions,
        kInput, &filter_dimensions, kFilter, &bias_dimensions, kBias,
        &output_dimensions, output);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return 1;
    }

    for (size_t input_y = 0; input_y < kInputHeight; ++input_y)
    {
        for (size_t input_x = 0; input_x < kInputWidth; ++input_x)
        {
            for (size_t filter_y = 0; filter_y < kFilterHeight; ++filter_y)
            {
                for (size_t filter_x = 0; filter_x < kFilterWidth; ++filter_x)
                {
                    const size_t output_y = input_y + filter_y;
                    const size_t output_x = input_x + filter_x;
                    for (size_t output_channel = 0;
                         output_channel < kOutputChannels;
                         ++output_channel)
                    {
                        const size_t output_index =
                            (output_y * kOutputWidth + output_x) *
                                kOutputChannels +
                            output_channel;
                        for (size_t input_channel = 0;
                             input_channel < kInputChannels;
                             ++input_channel)
                        {
                            const size_t input_index =
                                (input_y * kInputWidth + input_x) *
                                    kInputChannels +
                                input_channel;
                            const size_t filter_index =
                                ((output_channel * kFilterHeight + filter_y) *
                                     kFilterWidth +
                                 filter_x) *
                                    kInputChannels +
                                input_channel;
                            expected[output_index] +=
                                (kInput[input_index] + kInputOffset) *
                                kFilter[filter_index];
                        }
                    }
                }
            }
        }
    }

    for (size_t index = 0; index < kOutputCount; ++index)
    {
        int32_t value = expected[index] + kOutputOffset;
        if (value < kActivationMin)
        {
            value = kActivationMin;
        }
        if (value > kActivationMax)
        {
            value = kActivationMax;
        }
        if (output[index] != value)
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
    if call == _TRANSPOSE_S8_CALL:
        return _render_transpose_s8
    return None
