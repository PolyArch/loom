#!/usr/bin/env python3
"""Typed generated CMSIS-NN convolution matrix-kernel workloads."""

from __future__ import annotations

from typing import Callable


_CALLS = {
    (
        "arm_nn_mat_mult_kernel_s8_s16",
        "int8_t *(const int8_t *, const int16_t *, const uint16_t, "
        "const int32_t *, const int32_t *, const int32_t, const int16_t, "
        "const int16_t, const int32_t, const int32_t, "
        "const int32_t *const, int8_t *)",
    ): ("arm_nn_mat_mult_kernel_s8_s16", "s8"),
    (
        "arm_nn_mat_mult_kernel_row_offset_s8_s16",
        "int8_t *(const int8_t *, const int16_t *, const uint16_t, "
        "const int32_t *, const int32_t *, const int32_t, const int16_t, "
        "const int16_t, const int32_t, const int32_t, "
        "const int32_t *const, const int32_t, int8_t *)",
    ): ("arm_nn_mat_mult_kernel_row_offset_s8_s16", "row-offset-s8"),
    (
        "arm_nn_mat_mult_kernel_s16",
        "int16_t *(const int8_t *, const int16_t *, const int32_t, "
        "const int32_t *, const int32_t *, const int32_t, const int32_t, "
        "const int32_t, const cmsis_nn_bias_data *const, int16_t *)",
    ): ("arm_nn_mat_mult_kernel_s16", "s16"),
}


def _renderer(symbol: str, flavor: str) -> Callable[[str], str]:
    def render(_wrapper_symbol: str) -> str:
        output_type = "int16_t" if flavor == "s16" else "int8_t"
        output_min = "-20000" if flavor == "s16" else "-100"
        output_max = "20000" if flavor == "s16" else "100"
        sentinel = "-30000" if flavor == "s16" else "-101"
        aligned_columns = "kColumns" if flavor == "s16" else "7"
        row_address_offset = (
            "5" if flavor == "row-offset-s8" else "kOutputChannels"
        )
        output_offset = "" if flavor == "s16" else "+ kOutputOffset"

        if flavor == "s16":
            bias_setup = """const cmsis_nn_bias_data bias_data = {
        .data = kBias,
        .is_int32_bias = true,
    };"""
            input_b = """static const int16_t kInputB[2 * kAlignedColumns] = {
    7, -3, 12, -5, 9,
    -6, 8, 2, 4, -1,
};"""
            call_arguments = """kInputA, kInputB, kOutputChannels,
        kShifts, kMultipliers, kActivationMin, kActivationMax,
        kColumns, &bias_data, output"""
        else:
            bias_setup = ""
            input_b = """static const int16_t kInputB[2 * kAlignedColumns] = {
    7, -3, 12, -5, 9, 30000, 30001,
    -6, 8, 2, 4, -1, 30002, 30003,
};"""
            row_argument = (
                ", kRowAddressOffset" if flavor == "row-offset-s8" else ""
            )
            call_arguments = """kInputA, kInputB, kOutputChannels,
        kShifts, kMultipliers, kOutputOffset, kActivationMin,
        kActivationMax, kColumns, kAlignedColumns, kBias@ROW_ARGUMENT@,
        output""".replace("@ROW_ARGUMENT@", row_argument)

        template = """#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

enum {
    kOutputChannels = 3,
    kColumns = 5,
    kAlignedColumns = @ALIGNED_COLUMNS@,
    kRowAddressOffset = @ROW_ADDRESS_OFFSET@,
    kOutputSpan = 2 * kRowAddressOffset,
    kOutputOffset = 5,
    kMultiplier = 1 << 30,
    kShift = 1,
    kActivationMin = @OUTPUT_MIN@,
    kActivationMax = @OUTPUT_MAX@,
};

static const int8_t kInputA[kOutputChannels * kColumns] = {
    3, -2, 5, 1, -4,
    -6, 7, 2, -3, 4,
    7, 1, -5, 6, -2,
};
@INPUT_B@
static const int32_t kBias[kOutputChannels] = {11, -23, 37};
static const int32_t kMultipliers[kOutputChannels] = {
    kMultiplier, kMultiplier, kMultiplier,
};
static const int32_t kShifts[kOutputChannels] = {
    kShift, kShift, kShift,
};

int main(void)
{
    @OUTPUT_TYPE@ output[kOutputSpan];
    @OUTPUT_TYPE@ expected[kOutputSpan];
    for (size_t index = 0; index < kOutputSpan; ++index)
    {
        output[index] = @SENTINEL@;
        expected[index] = @SENTINEL@;
    }
    @BIAS_SETUP@

    @OUTPUT_TYPE@ *returned = @SYMBOL@(
        @CALL_ARGUMENTS@);
    if (returned != output + kOutputSpan)
    {
        return 1;
    }

    for (size_t vector = 0; vector < 2; ++vector)
    {
        for (size_t channel = 0; channel < kOutputChannels; ++channel)
        {
            int32_t value = kBias[channel];
            for (size_t column = 0; column < kColumns; ++column)
            {
                value += kInputA[channel * kColumns + column] *
                    kInputB[vector * kAlignedColumns + column];
            }
            value = value @OUTPUT_OFFSET@;
            if (value < kActivationMin)
            {
                value = kActivationMin;
            }
            if (value > kActivationMax)
            {
                value = kActivationMax;
            }
            expected[vector * kRowAddressOffset + channel] =
                (@OUTPUT_TYPE@)value;
        }
    }

    for (size_t index = 0; index < kOutputSpan; ++index)
    {
        if (output[index] != expected[index])
        {
            return 1;
        }
    }
    return 0;
}
"""
        return (
            template.replace("@ALIGNED_COLUMNS@", aligned_columns)
            .replace("@ROW_ADDRESS_OFFSET@", row_address_offset)
            .replace("@OUTPUT_MIN@", output_min)
            .replace("@OUTPUT_MAX@", output_max)
            .replace("@OUTPUT_TYPE@", output_type)
            .replace("@SENTINEL@", sentinel)
            .replace("@INPUT_B@", input_b)
            .replace("@BIAS_SETUP@", bias_setup)
            .replace("@SYMBOL@", symbol)
            .replace("@CALL_ARGUMENTS@", call_arguments)
            .replace("@OUTPUT_OFFSET@", output_offset)
        )

    return render


def renderer_for(call: tuple[str, str]) -> Callable[[str], str] | None:
    configuration = _CALLS.get(call)
    if configuration is None:
        return None
    return _renderer(*configuration)
