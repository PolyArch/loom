#!/usr/bin/env python3
"""Typed generated CMSIS-NN layout workloads."""

from __future__ import annotations

from typing import Callable


_PAD_S8_CALL = (
    "arm_pad_s8",
    "arm_cmsis_nn_status (const int8_t *, int8_t *, const int8_t, "
    "const cmsis_nn_dims *, const cmsis_nn_dims *, const cmsis_nn_dims *)",
)
_TRANSPOSE_S8_CALL = (
    "arm_transpose_s8",
    "arm_cmsis_nn_status (const int8_t *, int8_t *const, "
    "const cmsis_nn_dims *const, const cmsis_nn_dims *const, "
    "const cmsis_nn_transpose_params *const)",
)


def _render_pad_s8(_wrapper_symbol: str) -> str:
    return """#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"

enum {
    kInputN = 1,
    kInputH = 2,
    kInputW = 3,
    kInputC = 2,
    kPreN = 0,
    kPreH = 1,
    kPreW = 1,
    kPreC = 1,
    kPostN = 0,
    kPostH = 1,
    kPostW = 0,
    kPostC = 1,
    kOutputN = kPreN + kInputN + kPostN,
    kOutputH = kPreH + kInputH + kPostH,
    kOutputW = kPreW + kInputW + kPostW,
    kOutputC = kPreC + kInputC + kPostC,
    kInputCount = kInputN * kInputH * kInputW * kInputC,
    kOutputCount = kOutputN * kOutputH * kOutputW * kOutputC,
    kPadValue = -37,
};

static const int8_t kInput[kInputCount] = {
    -128, -97, -31, -1, 0, 1, 7, 31, 63, 95, 126, 127,
};

static size_t offset4(
    size_t n, size_t h, size_t w, size_t c,
    size_t height, size_t width, size_t channels)
{
    return ((n * height + h) * width + w) * channels + c;
}

int main(void)
{
    int8_t output[kOutputCount];
    int8_t expected[kOutputCount];
    for (size_t index = 0; index < kOutputCount; ++index)
    {
        output[index] = 0;
        expected[index] = kPadValue;
    }
    for (size_t n = 0; n < kInputN; ++n)
    {
        for (size_t h = 0; h < kInputH; ++h)
        {
            for (size_t w = 0; w < kInputW; ++w)
            {
                for (size_t c = 0; c < kInputC; ++c)
                {
                    const size_t input_index =
                        offset4(n, h, w, c, kInputH, kInputW, kInputC);
                    const size_t output_index = offset4(
                        n + kPreN, h + kPreH, w + kPreW, c + kPreC,
                        kOutputH, kOutputW, kOutputC);
                    expected[output_index] = kInput[input_index];
                }
            }
        }
    }

    const cmsis_nn_dims input_dimensions = {
        kInputN, kInputH, kInputW, kInputC,
    };
    const cmsis_nn_dims pre_padding = {kPreN, kPreH, kPreW, kPreC};
    const cmsis_nn_dims post_padding = {kPostN, kPostH, kPostW, kPostC};
    const arm_cmsis_nn_status status = arm_pad_s8(
        kInput, output, kPadValue,
        &input_dimensions, &pre_padding, &post_padding);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return 1;
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
    kInputN = 1,
    kInputH = 2,
    kInputW = 3,
    kInputC = 2,
    kOutputN = kInputN,
    kOutputH = kInputW,
    kOutputW = kInputH,
    kOutputC = kInputC,
    kElementCount = kInputN * kInputH * kInputW * kInputC,
};

static const int8_t kInput[kElementCount] = {
    -128, -97, -31, -1, 0, 1, 7, 31, 63, 95, 126, 127,
};
static const uint32_t kPermutation[4] = {0, 2, 1, 3};

static size_t offset4(
    size_t n, size_t h, size_t w, size_t c,
    size_t height, size_t width, size_t channels)
{
    return ((n * height + h) * width + w) * channels + c;
}

int main(void)
{
    int8_t output[kElementCount] = {0};
    int8_t expected[kElementCount] = {0};
    for (size_t n = 0; n < kInputN; ++n)
    {
        for (size_t h = 0; h < kInputH; ++h)
        {
            for (size_t w = 0; w < kInputW; ++w)
            {
                for (size_t c = 0; c < kInputC; ++c)
                {
                    const size_t input_index =
                        offset4(n, h, w, c, kInputH, kInputW, kInputC);
                    const size_t output_index =
                        offset4(n, w, h, c, kOutputH, kOutputW, kOutputC);
                    expected[output_index] = kInput[input_index];
                }
            }
        }
    }

    const cmsis_nn_dims input_dimensions = {
        kInputN, kInputH, kInputW, kInputC,
    };
    const cmsis_nn_dims output_dimensions = {
        kOutputN, kOutputH, kOutputW, kOutputC,
    };
    const cmsis_nn_transpose_params parameters = {4, kPermutation};
    const arm_cmsis_nn_status status = arm_transpose_s8(
        kInput, output, &input_dimensions, &output_dimensions, &parameters);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return 1;
    }
    for (size_t index = 0; index < kElementCount; ++index)
    {
        if (output[index] != expected[index])
        {
            return 1;
        }
    }
    return 0;
}
"""


_RENDERERS: dict[tuple[str, str], Callable[[str], str]] = {
    _PAD_S8_CALL: _render_pad_s8,
    _TRANSPOSE_S8_CALL: _render_transpose_s8,
}


def renderer_for(call: tuple[str, str]) -> Callable[[str], str] | None:
    return _RENDERERS.get(call)
