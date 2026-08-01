#!/usr/bin/env python3
"""Typed generated CMSIS-NN convolution protocols."""

from __future__ import annotations

from typing import Callable

import corpus_inventory


_CONVOLUTION_SIGNATURE = (
    "arm_cmsis_nn_status (const cmsis_nn_context *, const cmsis_nn_conv_params *, "
    "const cmsis_nn_per_channel_quant_params *, const cmsis_nn_dims *, "
    "const int8_t *, const cmsis_nn_dims *, const int8_t *, "
    "const cmsis_nn_dims *, const int32_t *, const cmsis_nn_dims *, int8_t *)"
)
_DEPTHWISE_SIGNATURE = (
    "arm_cmsis_nn_status (const cmsis_nn_context *, "
    "const cmsis_nn_dw_conv_params *, "
    "const cmsis_nn_per_channel_quant_params *, const cmsis_nn_dims *, "
    "const int8_t *, const cmsis_nn_dims *, const int8_t *, "
    "const cmsis_nn_dims *, const int32_t *, const cmsis_nn_dims *, int8_t *)"
)

_CONVOLVE_1X1_CALLS = (
    (
        "arm_convolve_1x1_s8_fast_get_buffer_size",
        "int32_t (const cmsis_nn_dims *)",
    ),
    ("arm_convolve_1x1_s8_fast", _CONVOLUTION_SIGNATURE),
)
_CONVOLVE_1_X_N_CALLS = (
    (
        "arm_convolve_1_x_n_s8_get_buffer_size",
        "int32_t (const cmsis_nn_conv_params *, const cmsis_nn_dims *, "
        "const cmsis_nn_dims *, const cmsis_nn_dims *)",
    ),
    ("arm_convolve_1_x_n_s8", _CONVOLUTION_SIGNATURE),
)
_CONVOLVE_S4_STRIDE_X_CALLS = (
    (
        "arm_convolve_1x1_s4_fast_get_buffer_size",
        "int32_t (const cmsis_nn_dims *)",
    ),
    ("arm_convolve_1x1_s4_fast", _CONVOLUTION_SIGNATURE),
    ("arm_convolve_1x1_s4", _CONVOLUTION_SIGNATURE),
)
_CONVOLVE_S4_DIRECT_CALLS = (
    ("arm_convolve_1x1_s4", _CONVOLUTION_SIGNATURE),
)
_CONVOLVE_S4_WRAPPER_CALLS = (
    (
        "arm_convolve_wrapper_s4_get_buffer_size",
        "int32_t (const cmsis_nn_conv_params *, const cmsis_nn_dims *, "
        "const cmsis_nn_dims *, const cmsis_nn_dims *)",
    ),
    ("arm_convolve_wrapper_s4", _CONVOLUTION_SIGNATURE),
    ("arm_convolve_1x1_s4", _CONVOLUTION_SIGNATURE),
)
_DEPTHWISE_WRAPPER_CALLS = (
    ("arm_depthwise_conv_3x3_s8", _DEPTHWISE_SIGNATURE),
    (
        "arm_depthwise_conv_wrapper_s8_get_buffer_size",
        "int32_t (const cmsis_nn_dw_conv_params *, const cmsis_nn_dims *, "
        "const cmsis_nn_dims *, const cmsis_nn_dims *)",
    ),
    ("arm_depthwise_conv_wrapper_s8", _DEPTHWISE_SIGNATURE),
)


def _render_convolution(
    wrapper_symbol: str,
    *,
    data_name: str,
    macro_name: str,
    query_symbol: str,
    query_arguments: str,
    operation_symbol: str,
) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "TestCases/TestData/{data_name}/test_data.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

enum {{ kScratchCapacity = 4096 }};

LOOM_NOINLINE arm_cmsis_nn_status {wrapper_symbol}(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *biases,
    const int32_t *multipliers,
    const int32_t *shifts,
    void *scratch,
    size_t scratch_capacity,
    int8_t *output)
{{
    cmsis_nn_context context = {{0}};
    cmsis_nn_conv_params parameters = {{0}};
    cmsis_nn_per_channel_quant_params quantization = {{0}};
    cmsis_nn_dims input_dimensions = {{0}};
    cmsis_nn_dims filter_dimensions = {{0}};
    cmsis_nn_dims bias_dimensions = {{0}};
    cmsis_nn_dims output_dimensions = {{0}};

    input_dimensions.n = {macro_name}_INPUT_BATCHES;
    input_dimensions.h = {macro_name}_INPUT_H;
    input_dimensions.w = {macro_name}_INPUT_W;
    input_dimensions.c = {macro_name}_IN_CH;
    filter_dimensions.n = {macro_name}_OUT_CH;
    filter_dimensions.h = {macro_name}_FILTER_Y;
    filter_dimensions.w = {macro_name}_FILTER_X;
    filter_dimensions.c = {macro_name}_IN_CH;
    bias_dimensions.n = 1;
    bias_dimensions.h = 1;
    bias_dimensions.w = 1;
    bias_dimensions.c = {macro_name}_OUT_CH;
    output_dimensions.n = {macro_name}_INPUT_BATCHES;
    output_dimensions.h = {macro_name}_OUTPUT_H;
    output_dimensions.w = {macro_name}_OUTPUT_W;
    output_dimensions.c = {macro_name}_OUT_CH;

    parameters.padding.h = {macro_name}_PAD_Y;
    parameters.padding.w = {macro_name}_PAD_X;
    parameters.stride.h = {macro_name}_STRIDE_Y;
    parameters.stride.w = {macro_name}_STRIDE_X;
    parameters.dilation.h = {macro_name}_DILATION_Y;
    parameters.dilation.w = {macro_name}_DILATION_X;
    parameters.input_offset = {macro_name}_INPUT_OFFSET;
    parameters.output_offset = {macro_name}_OUTPUT_OFFSET;
    parameters.activation.min = {macro_name}_OUT_ACTIVATION_MIN;
    parameters.activation.max = {macro_name}_OUT_ACTIVATION_MAX;
    quantization.multiplier = (int32_t *)multipliers;
    quantization.shift = (int32_t *)shifts;

    const int32_t required = {query_symbol}({query_arguments});
    if (required < 0 || (size_t)required > scratch_capacity)
    {{
        return ARM_CMSIS_NN_ARG_ERROR;
    }}
    context.buf = required == 0 ? NULL : scratch;
    context.size = required;
    return {operation_symbol}(&context,
                              &parameters,
                              &quantization,
                              &input_dimensions,
                              input,
                              &filter_dimensions,
                              weights,
                              &bias_dimensions,
                              biases,
                              &output_dimensions,
                              output);
}}

int main(void)
{{
    int8_t output[{macro_name}_DST_SIZE] = {{0}};
    uint8_t scratch[kScratchCapacity] = {{0}};
    const arm_cmsis_nn_status status = {wrapper_symbol}(
        {data_name}_input,
        {data_name}_weights,
        {data_name}_biases,
        {data_name}_output_mult,
        {data_name}_output_shift,
        scratch,
        sizeof(scratch),
        output);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {{
        return 1;
    }}
    for (size_t index = 0; index < {macro_name}_DST_SIZE; ++index)
    {{
        if (output[index] != {data_name}_output_ref[index])
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""


def _render_convolve_1x1(wrapper_symbol: str) -> str:
    return _render_convolution(
        wrapper_symbol,
        data_name="kernel1x1",
        macro_name="KERNEL1X1",
        query_symbol="arm_convolve_1x1_s8_fast_get_buffer_size",
        query_arguments="&input_dimensions",
        operation_symbol="arm_convolve_1x1_s8_fast",
    )


def _render_convolve_1_x_n(wrapper_symbol: str) -> str:
    return _render_convolution(
        wrapper_symbol,
        data_name="conv_1_x_n_6_generic",
        macro_name="CONV_1_X_N_6_GENERIC",
        query_symbol="arm_convolve_1_x_n_s8_get_buffer_size",
        query_arguments=(
            "&parameters, &input_dimensions, &filter_dimensions, "
            "&output_dimensions"
        ),
        operation_symbol="arm_convolve_1_x_n_s8",
    )


def _render_s4_convolution(
    wrapper_symbol: str,
    *,
    data_name: str,
    macro_name: str,
    protocol: str,
) -> str:
    if protocol == "fast-then-direct":
        operation = f"""
    const int32_t required =
        arm_convolve_1x1_s4_fast_get_buffer_size(&input_dimensions);
    if (required < 0 || (size_t)required > scratch_capacity)
    {{
        return ARM_CMSIS_NN_ARG_ERROR;
    }}
    context.buf = NULL;
    context.size = 0;
    arm_cmsis_nn_status status = arm_convolve_1x1_s4_fast(
        &context, &parameters, &quantization, &input_dimensions, input,
        &filter_dimensions, weights, &bias_dimensions, biases,
        &output_dimensions, primary_output);
    if (status != ARM_CMSIS_NN_ARG_ERROR)
    {{
        return ARM_CMSIS_NN_ARG_ERROR;
    }}
    return arm_convolve_1x1_s4(
        &context, &parameters, &quantization, &input_dimensions, input,
        &filter_dimensions, weights, &bias_dimensions, biases,
        &output_dimensions, primary_output);"""
        compare_secondary = False
    elif protocol == "direct":
        operation = """
    context.buf = NULL;
    context.size = 0;
    return arm_convolve_1x1_s4(
        &context, &parameters, &quantization, &input_dimensions, input,
        &filter_dimensions, weights, &bias_dimensions, biases,
        &output_dimensions, primary_output);"""
        compare_secondary = False
    elif protocol == "wrapper-then-direct":
        operation = """
    const int32_t required = arm_convolve_wrapper_s4_get_buffer_size(
        &parameters, &input_dimensions, &filter_dimensions, &output_dimensions);
    if (required < 0 || (size_t)required > scratch_capacity)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    context.buf = required == 0 ? NULL : scratch;
    context.size = required;
    arm_cmsis_nn_status status = arm_convolve_wrapper_s4(
        &context, &parameters, &quantization, &input_dimensions, input,
        &filter_dimensions, weights, &bias_dimensions, biases,
        &output_dimensions, secondary_output);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return status;
    }
    context.buf = NULL;
    context.size = 0;
    return arm_convolve_1x1_s4(
        &context, &parameters, &quantization, &input_dimensions, input,
        &filter_dimensions, weights, &bias_dimensions, biases,
        &output_dimensions, primary_output);"""
        compare_secondary = True
    else:
        raise ValueError(f"unknown s4 convolution protocol: {protocol}")

    secondary_check = ""
    if compare_secondary:
        secondary_check = (
            f" || secondary_output[index] != {data_name}_output_ref[index]"
        )

    return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "TestCases/TestData/{data_name}/test_data.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

enum {{ kScratchCapacity = 4096 }};

LOOM_NOINLINE arm_cmsis_nn_status {wrapper_symbol}(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *biases,
    const int32_t *multipliers,
    const int32_t *shifts,
    void *scratch,
    size_t scratch_capacity,
    int8_t *primary_output,
    int8_t *secondary_output)
{{
    cmsis_nn_context context = {{0}};
    cmsis_nn_conv_params parameters = {{0}};
    cmsis_nn_per_channel_quant_params quantization = {{0}};
    cmsis_nn_dims input_dimensions = {{0}};
    cmsis_nn_dims filter_dimensions = {{0}};
    cmsis_nn_dims bias_dimensions = {{0}};
    cmsis_nn_dims output_dimensions = {{0}};

    input_dimensions.n = {macro_name}_INPUT_BATCHES;
    input_dimensions.h = {macro_name}_INPUT_H;
    input_dimensions.w = {macro_name}_INPUT_W;
    input_dimensions.c = {macro_name}_IN_CH;
    filter_dimensions.n = {macro_name}_OUT_CH;
    filter_dimensions.h = {macro_name}_FILTER_Y;
    filter_dimensions.w = {macro_name}_FILTER_X;
    filter_dimensions.c = {macro_name}_IN_CH;
    bias_dimensions.n = 1;
    bias_dimensions.h = 1;
    bias_dimensions.w = 1;
    bias_dimensions.c = {macro_name}_OUT_CH;
    output_dimensions.n = {macro_name}_INPUT_BATCHES;
    output_dimensions.h = {macro_name}_OUTPUT_H;
    output_dimensions.w = {macro_name}_OUTPUT_W;
    output_dimensions.c = {macro_name}_OUT_CH;

    parameters.padding.h = {macro_name}_PAD_Y;
    parameters.padding.w = {macro_name}_PAD_X;
    parameters.stride.h = {macro_name}_STRIDE_Y;
    parameters.stride.w = {macro_name}_STRIDE_X;
    parameters.dilation.h = {macro_name}_DILATION_Y;
    parameters.dilation.w = {macro_name}_DILATION_X;
    parameters.input_offset = {macro_name}_INPUT_OFFSET;
    parameters.output_offset = {macro_name}_OUTPUT_OFFSET;
    parameters.activation.min = {macro_name}_OUT_ACTIVATION_MIN;
    parameters.activation.max = {macro_name}_OUT_ACTIVATION_MAX;
    quantization.multiplier = (int32_t *)multipliers;
    quantization.shift = (int32_t *)shifts;
    (void)secondary_output;
{operation}
}}

int main(void)
{{
    int8_t primary_output[{macro_name}_DST_SIZE] = {{0}};
    int8_t secondary_output[{macro_name}_DST_SIZE] = {{0}};
    uint8_t scratch[kScratchCapacity] = {{0}};
    const arm_cmsis_nn_status status = {wrapper_symbol}(
        {data_name}_input,
        {data_name}_weights,
        {data_name}_biases,
        {data_name}_output_mult,
        {data_name}_output_shift,
        scratch,
        sizeof(scratch),
        primary_output,
        secondary_output);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {{
        return 1;
    }}
    for (size_t index = 0; index < {macro_name}_DST_SIZE; ++index)
    {{
        if (primary_output[index] != {data_name}_output_ref[index]{secondary_check})
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""


def _render_s4_stride_x(wrapper_symbol: str) -> str:
    return _render_s4_convolution(
        wrapper_symbol,
        data_name="kernel1x1_stride_x_int4",
        macro_name="KERNEL1X1_STRIDE_X_INT4",
        protocol="fast-then-direct",
    )


def _render_s4_direct(wrapper_symbol: str) -> str:
    return _render_s4_convolution(
        wrapper_symbol,
        data_name="kernel1x1_stride_x_y_1_int4",
        macro_name="KERNEL1X1_STRIDE_X_Y_1_INT4",
        protocol="direct",
    )


def _render_s4_wrapper(wrapper_symbol: str) -> str:
    return _render_s4_convolution(
        wrapper_symbol,
        data_name="kernel1x1_stride_x_y_int4",
        macro_name="KERNEL1X1_STRIDE_X_Y_INT4",
        protocol="wrapper-then-direct",
    )


def _render_depthwise_wrapper(wrapper_symbol: str) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "TestCases/TestData/depthwise_kernel_3x3/test_data.h"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

enum {{ kScratchCapacity = 4096 }};

LOOM_NOINLINE arm_cmsis_nn_status {wrapper_symbol}(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *biases,
    const int32_t *multipliers,
    const int32_t *shifts,
    void *scratch,
    size_t scratch_capacity,
    int8_t *direct_output,
    int8_t *wrapper_output)
{{
    cmsis_nn_context direct_context = {{0}};
    cmsis_nn_context wrapper_context = {{0}};
    cmsis_nn_dw_conv_params parameters = {{0}};
    cmsis_nn_per_channel_quant_params quantization = {{0}};
    cmsis_nn_dims input_dimensions = {{0}};
    cmsis_nn_dims filter_dimensions = {{0}};
    cmsis_nn_dims bias_dimensions = {{0}};
    cmsis_nn_dims output_dimensions = {{0}};

    input_dimensions.n = DEPTHWISE_KERNEL_3X3_INPUT_BATCHES;
    input_dimensions.h = DEPTHWISE_KERNEL_3X3_INPUT_H;
    input_dimensions.w = DEPTHWISE_KERNEL_3X3_INPUT_W;
    input_dimensions.c = DEPTHWISE_KERNEL_3X3_IN_CH;
    filter_dimensions.n = DEPTHWISE_KERNEL_3X3_OUT_CH;
    filter_dimensions.h = DEPTHWISE_KERNEL_3X3_FILTER_Y;
    filter_dimensions.w = DEPTHWISE_KERNEL_3X3_FILTER_X;
    filter_dimensions.c = DEPTHWISE_KERNEL_3X3_IN_CH;
    output_dimensions.n = DEPTHWISE_KERNEL_3X3_INPUT_BATCHES;
    output_dimensions.h = DEPTHWISE_KERNEL_3X3_OUTPUT_H;
    output_dimensions.w = DEPTHWISE_KERNEL_3X3_OUTPUT_W;
    output_dimensions.c = DEPTHWISE_KERNEL_3X3_OUT_CH;

    parameters.padding.h = DEPTHWISE_KERNEL_3X3_PAD_Y;
    parameters.padding.w = DEPTHWISE_KERNEL_3X3_PAD_X;
    parameters.stride.h = DEPTHWISE_KERNEL_3X3_STRIDE_Y;
    parameters.stride.w = DEPTHWISE_KERNEL_3X3_STRIDE_X;
    parameters.dilation.h = DEPTHWISE_KERNEL_3X3_DILATION_Y;
    parameters.dilation.w = DEPTHWISE_KERNEL_3X3_DILATION_X;
    parameters.ch_mult = DEPTHWISE_KERNEL_3X3_CH_MULT;
    parameters.input_offset = DEPTHWISE_KERNEL_3X3_INPUT_OFFSET;
    parameters.output_offset = DEPTHWISE_KERNEL_3X3_OUTPUT_OFFSET;
    parameters.activation.min = DEPTHWISE_KERNEL_3X3_OUT_ACTIVATION_MIN;
    parameters.activation.max = DEPTHWISE_KERNEL_3X3_OUT_ACTIVATION_MAX;
    quantization.multiplier = (int32_t *)multipliers;
    quantization.shift = (int32_t *)shifts;

    arm_cmsis_nn_status status = arm_depthwise_conv_3x3_s8(
        &direct_context, &parameters, &quantization,
        &input_dimensions, input, &filter_dimensions, weights,
        &bias_dimensions, biases, &output_dimensions, direct_output);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {{
        return status;
    }}
    const int32_t required = arm_depthwise_conv_wrapper_s8_get_buffer_size(
        &parameters, &input_dimensions, &filter_dimensions, &output_dimensions);
    if (required < 0 || (size_t)required > scratch_capacity)
    {{
        return ARM_CMSIS_NN_ARG_ERROR;
    }}
    wrapper_context.buf = required == 0 ? NULL : scratch;
    wrapper_context.size = required;
    return arm_depthwise_conv_wrapper_s8(
        &wrapper_context, &parameters, &quantization,
        &input_dimensions, input, &filter_dimensions, weights,
        &bias_dimensions, biases, &output_dimensions, wrapper_output);
}}

int main(void)
{{
    int8_t direct_output[DEPTHWISE_KERNEL_3X3_DST_SIZE] = {{0}};
    int8_t wrapper_output[DEPTHWISE_KERNEL_3X3_DST_SIZE] = {{0}};
    uint8_t scratch[kScratchCapacity] = {{0}};
    const arm_cmsis_nn_status status = {wrapper_symbol}(
        depthwise_kernel_3x3_input,
        depthwise_kernel_3x3_weights,
        depthwise_kernel_3x3_biases,
        depthwise_kernel_3x3_output_mult,
        depthwise_kernel_3x3_output_shift,
        scratch,
        sizeof(scratch),
        direct_output,
        wrapper_output);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {{
        return 1;
    }}
    for (size_t index = 0; index < DEPTHWISE_KERNEL_3X3_DST_SIZE; ++index)
    {{
        const int8_t expected = depthwise_kernel_3x3_output_ref[index];
        if (direct_output[index] != expected || wrapper_output[index] != expected)
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""


_RENDERERS: dict[
    tuple[tuple[str, str], ...],
    Callable[[str], str],
] = {
    _CONVOLVE_1X1_CALLS: _render_convolve_1x1,
    _CONVOLVE_1_X_N_CALLS: _render_convolve_1_x_n,
    _CONVOLVE_S4_STRIDE_X_CALLS: _render_s4_stride_x,
    _CONVOLVE_S4_DIRECT_CALLS: _render_s4_direct,
    _CONVOLVE_S4_WRAPPER_CALLS: _render_s4_wrapper,
    _DEPTHWISE_WRAPPER_CALLS: _render_depthwise_wrapper,
}


def renderer_for(
    workload: corpus_inventory.ProgramWorkload,
) -> Callable[[str], str] | None:
    calls = tuple((call.symbol, call.signature) for call in workload.protocol)
    return _RENDERERS.get(calls)
