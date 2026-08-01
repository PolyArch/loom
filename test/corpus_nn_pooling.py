#!/usr/bin/env python3
"""Typed generated CMSIS-NN pooling protocols."""

from __future__ import annotations

from typing import Callable


_MAX_POOL_S8_SIGNATURE = (
    "arm_cmsis_nn_status (const cmsis_nn_context *, const cmsis_nn_pool_params *, "
    "const cmsis_nn_dims *, const int8_t *, const cmsis_nn_dims *, "
    "const cmsis_nn_dims *, int8_t *)"
)
_MAX_POOL_S16_SIGNATURE = (
    "arm_cmsis_nn_status (const cmsis_nn_context *, const cmsis_nn_pool_params *, "
    "const cmsis_nn_dims *, const int16_t *, const cmsis_nn_dims *, "
    "const cmsis_nn_dims *, int16_t *)"
)


def _render_max_pool(
    symbol: str,
    *,
    element_type: str,
    data_name: str,
    macro_name: str,
) -> Callable[[str], str]:
    def render(_wrapper_symbol: str) -> str:
        return f"""#include <stddef.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "TestCases/TestData/{data_name}/test_data.h"

int main(void)
{{
    cmsis_nn_context context = {{0}};
    cmsis_nn_pool_params parameters = {{0}};
    cmsis_nn_dims input_dimensions = {{0}};
    cmsis_nn_dims filter_dimensions = {{0}};
    cmsis_nn_dims output_dimensions = {{0}};
    {element_type} output[sizeof({data_name}_output) /
                          sizeof({data_name}_output[0])] = {{0}};

    input_dimensions.n = {macro_name}_BATCH_SIZE;
    input_dimensions.h = {macro_name}_INPUT_H;
    input_dimensions.w = {macro_name}_INPUT_W;
    input_dimensions.c = {macro_name}_INPUT_C;
    filter_dimensions.n = 1;
    filter_dimensions.h = {macro_name}_FILTER_H;
    filter_dimensions.w = {macro_name}_FILTER_W;
    filter_dimensions.c = 1;
    output_dimensions.n = {macro_name}_BATCH_SIZE;
    output_dimensions.h = {macro_name}_OUTPUT_H;
    output_dimensions.w = {macro_name}_OUTPUT_W;
    output_dimensions.c = {macro_name}_OUTPUT_C;
    parameters.padding.h = {macro_name}_PADDING_H;
    parameters.padding.w = {macro_name}_PADDING_W;
    parameters.stride.h = {macro_name}_STRIDE_H;
    parameters.stride.w = {macro_name}_STRIDE_W;
    parameters.activation.min = {macro_name}_ACTIVATION_MIN;
    parameters.activation.max = {macro_name}_ACTIVATION_MAX;

    const arm_cmsis_nn_status status = {symbol}(
        &context,
        &parameters,
        &input_dimensions,
        {data_name}_input_tensor,
        &filter_dimensions,
        &output_dimensions,
        output);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {{
        return 1;
    }}
    for (size_t index = 0;
         index < sizeof(output) / sizeof(output[0]);
         ++index)
    {{
        if (output[index] != {data_name}_output[index])
        {{
            return 1;
        }}
    }}
    return 0;
}}
"""

    return render


_RENDERERS: dict[tuple[str, str], Callable[[str], str]] = {
    (
        "arm_max_pool_s8",
        _MAX_POOL_S8_SIGNATURE,
    ): _render_max_pool(
        "arm_max_pool_s8",
        element_type="int8_t",
        data_name="maxpooling_1",
        macro_name="MAXPOOLING_1",
    ),
    (
        "arm_max_pool_s16",
        _MAX_POOL_S16_SIGNATURE,
    ): _render_max_pool(
        "arm_max_pool_s16",
        element_type="int16_t",
        data_name="maxpool_int16_1",
        macro_name="MAXPOOL_INT16_1",
    ),
}


def renderer_for(call: tuple[str, str]) -> Callable[[str], str] | None:
    return _RENDERERS.get(call)
