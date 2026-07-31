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
_S16_VEC_MAT_CALLS = {
    (
        "arm_nn_vec_mat_mult_t_s16",
        "arm_cmsis_nn_status (const int16_t *, const int8_t *, "
        "const int64_t *, int16_t *, const int32_t, const int32_t, "
        "const int32_t, const int32_t, const int32_t, const int32_t)",
    ): ("arm_nn_vec_mat_mult_t_s16", "int8_t"),
    (
        "arm_nn_vec_mat_mult_t_s16_s16",
        "arm_cmsis_nn_status (const int16_t *, const int16_t *, "
        "const int64_t *, int16_t *, const int32_t, const int32_t, "
        "const int32_t, const int32_t, const int32_t, const int32_t)",
    ): ("arm_nn_vec_mat_mult_t_s16_s16", "int16_t"),
}
_ACCUMULATING_VEC_MAT_CALLS = {
    (
        "arm_nn_vec_mat_mul_result_acc_s16",
        "arm_cmsis_nn_status (const int16_t *, const int8_t *, "
        "const int64_t *, int16_t *, const int32_t, const int32_t, "
        "const int32_t, const int32_t, const int32_t, const int32_t)",
    ): ("arm_nn_vec_mat_mul_result_acc_s16", "int16_t", "int64_t"),
    (
        "arm_nn_vec_mat_mul_result_acc_s8_s16",
        "arm_cmsis_nn_status (const int8_t *, const int8_t *, "
        "const int32_t *, int16_t *, const int32_t, const int32_t, "
        "const int32_t, const int32_t, const int32_t, const int32_t)",
    ): ("arm_nn_vec_mat_mul_result_acc_s8_s16", "int8_t", "int32_t"),
}


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


def _s16_vec_mat_renderer(symbol: str, rhs_type: str) -> Callable[[str], str]:
    def render(_wrapper_symbol: str) -> str:
        rhs_values = (
            "3, -2, 5, 1, -4, -6, 7, 2, -3, 4, 8, 1, -5, 6, -2"
            if rhs_type == "int8_t"
            else "300, -200, 50, 100, -40, -60, 700, 20, -30, 40, "
            "80, 10, -50, 60, -20"
        )
        return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnsupportfunctions.h"

enum {{
    kColumnCount = 5,
    kRowCount = 3,
    kReducedMultiplier = 16384,
    kShift = 1,
    kActivationMin = -32768,
    kActivationMax = 32767,
}};

static const int16_t kLhs[kColumnCount] = {{7, -3, 12, -5, 9}};
static const {rhs_type} kRhs[kRowCount * kColumnCount] = {{
    {rhs_values},
}};
static const int64_t kBias[kRowCount] = {{11, -23, 37}};

int main(void)
{{
    int16_t output[kRowCount] = {{0}};
    const arm_cmsis_nn_status status = {symbol}(
        kLhs, kRhs, kBias, output, kReducedMultiplier, kShift,
        kColumnCount, kRowCount, kActivationMin, kActivationMax);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {{
        return 1;
    }}

    for (size_t row = 0; row < kRowCount; ++row)
    {{
        int64_t expected = kBias[row];
        for (size_t column = 0; column < kColumnCount; ++column)
        {{
            expected += kLhs[column] *
                kRhs[row * kColumnCount + column];
        }}
        if (expected < kActivationMin)
        {{
            expected = kActivationMin;
        }}
        if (expected > kActivationMax)
        {{
            expected = kActivationMax;
        }}
        if (output[row] != (int16_t)expected)
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""

    return render


def _accumulating_vec_mat_renderer(
    symbol: str, lhs_type: str, bias_type: str
) -> Callable[[str], str]:
    def render(_wrapper_symbol: str) -> str:
        return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnsupportfunctions.h"

enum {{
    kColumnCount = 5,
    kRowCount = 3,
    kBatchCount = 2,
    kBatchOffset = 2,
    kMultiplier = 1 << 30,
    kShift = 1,
    kOutputCount = kBatchCount * kRowCount,
}};

static const {lhs_type} kLhs[
    kColumnCount * (1 + (kBatchCount - 1) * kBatchOffset)] = {{
    7, -3, 12, -5, 9, 2, -8, 4, 11, -6, 5, 1, -9, 3, 10,
}};
static const int8_t kRhs[kRowCount * kColumnCount] = {{
    3, -2, 5, 1, -4,
    -6, 7, 2, -3, 4,
    8, 1, -5, 6, -2,
}};
static const {bias_type} kBias[kRowCount] = {{11, -23, 37}};

int main(void)
{{
    int16_t output[kOutputCount] = {{5, -7, 9, -11, 13, -15}};
    int16_t expected[kOutputCount];
    for (size_t index = 0; index < kOutputCount; ++index)
    {{
        expected[index] = output[index];
    }}

    const arm_cmsis_nn_status status = {symbol}(
        kLhs, kRhs, kBias, output, kMultiplier, kShift,
        kColumnCount, kRowCount, kBatchCount, kBatchOffset);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {{
        return 1;
    }}

    for (size_t batch = 0; batch < kBatchCount; ++batch)
    {{
        for (size_t row = 0; row < kRowCount; ++row)
        {{
            int64_t accumulated = kBias[row];
            for (size_t column = 0; column < kColumnCount; ++column)
            {{
                accumulated +=
                    kLhs[batch * kColumnCount * kBatchOffset + column] *
                    kRhs[row * kColumnCount + column];
            }}
            const size_t destination = batch * kRowCount + row;
            expected[batch * kRowCount + row] += (int16_t)accumulated;
            if (expected[destination] < -32768)
            {{
                expected[destination] = -32768;
            }}
            if (expected[destination] > 32767)
            {{
                expected[destination] = 32767;
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

    return render


def renderer_for(call: tuple[str, str]) -> Callable[[str], str] | None:
    if call == _S32_NT_T_CALL:
        return _render_s32_nt_t
    vec_mat = _S16_VEC_MAT_CALLS.get(call)
    if vec_mat is not None:
        return _s16_vec_mat_renderer(*vec_mat)
    accumulating = _ACCUMULATING_VEC_MAT_CALLS.get(call)
    if accumulating is not None:
        return _accumulating_vec_mat_renderer(*accumulating)
    return None
