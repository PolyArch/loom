#!/usr/bin/env python3
"""Typed generated CMSIS-NN LSTM protocol workloads."""

from __future__ import annotations

from typing import Callable


_GATE_S16_CALL = (
    "arm_nn_lstm_calculate_gate_s16",
    "arm_cmsis_nn_status (const int16_t *, const int16_t *, "
    "const cmsis_nn_lstm_gate *, const cmsis_nn_lstm_params *, int16_t *, "
    "const int32_t)",
)
_GATE_S8_S16_CALL = (
    "arm_nn_lstm_calculate_gate_s8_s16",
    "arm_cmsis_nn_status (const int8_t *, const int8_t *, "
    "const cmsis_nn_lstm_gate *, const cmsis_nn_lstm_params *, int16_t *, "
    "const int32_t)",
)
_STEP_S16_CALL = (
    "arm_nn_lstm_step_s16",
    "arm_cmsis_nn_status (const int16_t *, const int16_t *, int16_t *, "
    "const cmsis_nn_lstm_params *, cmsis_nn_lstm_context *, const int32_t)",
)
_STEP_S8_CALL = (
    "arm_nn_lstm_step_s8",
    "arm_cmsis_nn_status (const int8_t *, const int8_t *, int8_t *, "
    "const cmsis_nn_lstm_params *, cmsis_nn_lstm_context *, const int32_t)",
)


def _render_protocol(*, scalar: str, operation: str) -> str:
    if scalar == "s16":
        input_type = "int16_t"
        output_type = "int16_t"
        bias_type = "int64_t"
        gate_symbol = "arm_nn_lstm_calculate_gate_s16"
        vec_mat_symbol = "arm_nn_vec_mat_mul_result_acc_s16"
        step_symbol = "arm_nn_lstm_step_s16"
        output_mul_call = """arm_elementwise_mul_s16_batch_offset(
        first_gate, second_gate, hidden_out, params->output_offset,
        params->output_multiplier, params->output_shift,
        params->hidden_size, params->batch_size, batch_offset);"""
        data_values = "1200, -700, 350, -900, 1500, 600"
        hidden_values = "400, -800, 1000, -300"
        reference_gate = "reference_gate_s16"
    elif scalar == "s8":
        input_type = "int8_t"
        output_type = "int8_t"
        bias_type = "int32_t"
        gate_symbol = "arm_nn_lstm_calculate_gate_s8_s16"
        vec_mat_symbol = "arm_nn_vec_mat_mul_result_acc_s8_s16"
        step_symbol = "arm_nn_lstm_step_s8"
        output_mul_call = """arm_elementwise_mul_s16_s8(
        first_gate, second_gate, hidden_out, params->output_offset,
        params->output_multiplier, params->output_shift,
        params->hidden_size, params->batch_size, batch_offset);"""
        data_values = "12, -7, 3, -9, 15, 6"
        hidden_values = "4, -8, 10, -3"
        reference_gate = "reference_gate_s8_s16"
    else:
        raise ValueError(f"unsupported LSTM scalar type: {scalar}")

    reference_step = f"reference_step_{scalar}"
    target_symbol = gate_symbol if operation == "gate" else step_symbol
    main_body = (
        f"""int16_t output[kStateElements] = {{0}};
    int16_t expected[kStateElements] = {{0}};
    const arm_cmsis_nn_status status = {target_symbol}(
        kData, kHidden, &params.input_gate, &params, output, kBatchOffset);
    {reference_gate}(
        kData, kHidden, &params.input_gate, &params, expected,
        kBatchOffset);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {{
        return 1;
    }}
    for (size_t index = 0; index < kStateElements; ++index)
    {{
        if (output[index] != expected[index])
        {{
            return 1;
        }}
    }}"""
        if operation == "gate"
        else f"""int16_t target_temp1[kStateElements] = {{0}};
    int16_t target_temp2[kStateElements] = {{0}};
    int16_t target_cell[kStateElements] = {{1024, -2048, 3072, -4096}};
    {output_type} target_output[kStateElements] = {{0}};
    cmsis_nn_lstm_context target_context = {{
        .temp1 = target_temp1,
        .temp2 = target_temp2,
        .cell_state = target_cell,
    }};

    int16_t reference_temp1[kStateElements] = {{0}};
    int16_t reference_temp2[kStateElements] = {{0}};
    int16_t reference_cell[kStateElements] = {{1024, -2048, 3072, -4096}};
    {output_type} reference_output[kStateElements] = {{0}};
    cmsis_nn_lstm_context reference_context = {{
        .temp1 = reference_temp1,
        .temp2 = reference_temp2,
        .cell_state = reference_cell,
    }};

    const arm_cmsis_nn_status status = {target_symbol}(
        kData, kHidden, target_output, &params, &target_context,
        kBatchOffset);
    {reference_step}(
        kData, kHidden, reference_output, &params, &reference_context,
        kBatchOffset);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {{
        return 1;
    }}
    for (size_t index = 0; index < kStateElements; ++index)
    {{
        if (target_output[index] != reference_output[index] ||
            target_cell[index] != reference_cell[index])
        {{
            return 1;
        }}
        }}"""
    )
    reference_step_source = ""
    if operation == "step":
        reference_step_source = f"""static void {reference_step}(
    const {input_type} *data_in,
    const {input_type} *hidden_in,
    {output_type} *hidden_out,
    const cmsis_nn_lstm_params *params,
    cmsis_nn_lstm_context *buffers,
    int32_t batch_offset)
{{
    int16_t *first_gate = buffers->temp1;
    int16_t *second_gate = buffers->temp2;
    int16_t *cell_state = buffers->cell_state;
    const int32_t element_count = params->hidden_size * params->batch_size;

    {reference_gate}(
        data_in, hidden_in, &params->forget_gate, params, first_gate,
        batch_offset);
    arm_elementwise_mul_s16(
        first_gate, cell_state, 0, 0, cell_state, 0,
        params->forget_to_cell_multiplier, params->forget_to_cell_shift,
        INT16_MIN, INT16_MAX, element_count);

    {reference_gate}(
        data_in, hidden_in, &params->input_gate, params, first_gate,
        batch_offset);
    {reference_gate}(
        data_in, hidden_in, &params->cell_gate, params, second_gate,
        batch_offset);
    arm_elementwise_mul_acc_s16(
        first_gate, second_gate, 0, 0, cell_state, 0,
        params->input_to_cell_multiplier, params->input_to_cell_shift,
        -params->cell_clip, params->cell_clip, element_count);

    {reference_gate}(
        data_in, hidden_in, &params->output_gate, params, first_gate,
        batch_offset);
    arm_nn_activation_s16(
        cell_state, second_gate, element_count,
        params->cell_scale_power + 12, ARM_TANH);
    {output_mul_call}
}}

"""

    return f"""#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

enum {{
    kInputSize = 3,
    kHiddenSize = 2,
    kBatchCount = 2,
    kBatchOffset = 1,
    kStateElements = kHiddenSize * kBatchCount,
}};

static const {input_type} kData[kInputSize * kBatchCount] = {{
    {data_values},
}};
static const {input_type} kHidden[kHiddenSize * kBatchCount] = {{
    {hidden_values},
}};
static const int8_t kInputWeights[kHiddenSize * kInputSize] = {{
    3, -2, 1, -4, 5, 2,
}};
static const int8_t kHiddenWeights[kHiddenSize * kHiddenSize] = {{
    2, -3, 4, 1,
}};
static const {bias_type} kInputBias[kHiddenSize] = {{400, -700}};
static const {bias_type} kHiddenBias[kHiddenSize] = {{-300, 500}};

static cmsis_nn_lstm_gate make_gate(arm_nn_activation_type activation)
{{
    cmsis_nn_lstm_gate gate = {{0}};
    gate.input_multiplier = 1 << 30;
    gate.input_shift = 1;
    gate.input_weights = kInputWeights;
    gate.input_effective_bias = kInputBias;
    gate.hidden_multiplier = 1 << 30;
    gate.hidden_shift = 1;
    gate.hidden_weights = kHiddenWeights;
    gate.hidden_effective_bias = kHiddenBias;
    gate.activation_type = activation;
    return gate;
}}

static cmsis_nn_lstm_params make_params(void)
{{
    cmsis_nn_lstm_params params = {{0}};
    params.batch_size = kBatchCount;
    params.time_steps = 1;
    params.input_size = kInputSize;
    params.hidden_size = kHiddenSize;
    params.forget_to_cell_multiplier = 1 << 30;
    params.forget_to_cell_shift = 1;
    params.input_to_cell_multiplier = 1 << 30;
    params.input_to_cell_shift = 1;
    params.cell_clip = INT16_MAX;
    params.cell_scale_power = -12;
    params.output_multiplier = 1 << 30;
    params.output_shift = 1;
    params.output_offset = 0;
    params.forget_gate = make_gate(ARM_SIGMOID);
    params.input_gate = make_gate(ARM_SIGMOID);
    params.cell_gate = make_gate(ARM_TANH);
    params.output_gate = make_gate(ARM_SIGMOID);
    return params;
}}

static void {reference_gate}(
    const {input_type} *data_in,
    const {input_type} *hidden_in,
    const cmsis_nn_lstm_gate *gate,
    const cmsis_nn_lstm_params *params,
    int16_t *output,
    int32_t batch_offset)
{{
    memset(output, 0,
           (size_t)params->hidden_size * params->batch_size * sizeof(*output));
    {vec_mat_symbol}(
        data_in, gate->input_weights, gate->input_effective_bias, output,
        gate->input_multiplier, gate->input_shift, params->input_size,
        params->hidden_size, params->batch_size, batch_offset);
    if (hidden_in != NULL)
    {{
        {vec_mat_symbol}(
            hidden_in, gate->hidden_weights, gate->hidden_effective_bias,
            output, gate->hidden_multiplier, gate->hidden_shift,
            params->hidden_size, params->hidden_size, params->batch_size,
            batch_offset);
    }}
    arm_nn_activation_s16(
        output, output, params->hidden_size * params->batch_size, 0,
        gate->activation_type);
}}

{reference_step_source}
int main(void)
{{
    const cmsis_nn_lstm_params params = make_params();
    {main_body}
    return 0;
}}
"""


def renderer_for(call: tuple[str, str]) -> Callable[[str], str] | None:
    if call == _GATE_S16_CALL:
        return lambda _wrapper_symbol: _render_protocol(
            scalar="s16", operation="gate"
        )
    if call == _GATE_S8_S16_CALL:
        return lambda _wrapper_symbol: _render_protocol(
            scalar="s8", operation="gate"
        )
    if call == _STEP_S16_CALL:
        return lambda _wrapper_symbol: _render_protocol(
            scalar="s16", operation="step"
        )
    if call == _STEP_S8_CALL:
        return lambda _wrapper_symbol: _render_protocol(
            scalar="s8", operation="step"
        )
    return None
