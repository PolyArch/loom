#!/usr/bin/env python3
"""Typed generated CMSIS-NN matrix workload protocols."""

from __future__ import annotations

from typing import Callable


_S32_NT_T_CALL = (
    "arm_nn_mat_mult_nt_t_s8_s32",
    "arm_cmsis_nn_status (const int8_t *, const int8_t *, int32_t *, "
    "const int32_t, const int32_t, const int32_t, const int32_t, "
    "const int32_t)",
)


def _render_s32_nt_t(_wrapper_symbol: str) -> str:
    return """#include <stddef.h>
#include <stdint.h>

#include "arm_nnsupportfunctions.h"

enum {
    kLhsRows = 3,
    kRhsRows = 5,
    kRhsColumns = 4,
    kLhsOffset = 3,
    kDstIndexOffset = 2,
    kOutputCount = kLhsRows * kRhsColumns * kDstIndexOffset,
};

static const int8_t kLhs[kLhsRows * kRhsRows] = {
    -11, 4, 7, -3, 9,
    5, -8, 2, 6, -1,
    10, -4, -7, 3, 8,
};
static const int8_t kRhs[kRhsColumns * kRhsRows] = {
    6, -2, 5, 1, -7,
    -3, 8, -4, 2, 9,
    7, 0, -6, 4, -1,
    -5, 3, 10, -8, 2,
};

int main(void)
{
    int32_t output[kOutputCount];
    int32_t expected[kOutputCount];
    for (size_t index = 0; index < kOutputCount; ++index)
    {
        output[index] = (int32_t)((index * 13) % 29) - 14;
        expected[index] = output[index];
    }

    const arm_cmsis_nn_status status = arm_nn_mat_mult_nt_t_s8_s32(
        kLhs, kRhs, output, kLhsRows, kRhsRows, kRhsColumns,
        kLhsOffset, kDstIndexOffset);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return 1;
    }

    for (size_t row = 0; row < kLhsRows; ++row)
    {
        for (size_t column = 0; column < kRhsColumns; ++column)
        {
            const size_t destination =
                (row * kRhsColumns + column) * kDstIndexOffset;
            for (size_t depth = 0; depth < kRhsRows; ++depth)
            {
                expected[destination] +=
                    (kLhs[row * kRhsRows + depth] + kLhsOffset) *
                    kRhs[column * kRhsRows + depth];
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
    if call == _S32_NT_T_CALL:
        return _render_s32_nt_t
    return None
