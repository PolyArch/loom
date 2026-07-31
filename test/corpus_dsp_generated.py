#!/usr/bin/env python3
"""Typed generated protocols for CMSIS-DSP census workloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


class TransformQueryKind(Enum):
    CFFT_OUTPUT = "cfft-output"
    CFFT_TEMPORARY = "cfft-temporary"
    CIFFT_OUTPUT = "cifft-output"
    MFCC_TEMPORARY = "mfcc-temporary"
    RFFT_OUTPUT = "rfft-output"
    RFFT_TEMPORARY = "rfft-temporary"
    RIFFT_INPUT = "rifft-input"


class LifecycleKind(Enum):
    MATRIX_INIT = "matrix-init"
    PID_RESET = "pid-reset"


@dataclass(frozen=True)
class TransformQueryProtocol:
    kind: TransformQueryKind
    symbol: str
    signature: str

    @property
    def owner_header(self) -> str:
        return "transform_functions.h"


@dataclass(frozen=True)
class LifecycleProtocol:
    kind: LifecycleKind
    symbol: str
    signature: str
    instance_type: str
    value_type: str
    owner_header: str
    target_profile: str


_TRANSFORM_QUERY_PROTOCOLS = (
    TransformQueryProtocol(
        TransformQueryKind.CFFT_OUTPUT,
        "arm_cfft_output_buffer_size",
        "i32(i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.CFFT_TEMPORARY,
        "arm_cfft_tmp_buffer_size",
        "i32(i32,i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.CIFFT_OUTPUT,
        "arm_cifft_output_buffer_size",
        "i32(i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.MFCC_TEMPORARY,
        "arm_mfcc_tmp_buffer_size",
        "i32(i32,i32,i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.RFFT_OUTPUT,
        "arm_rfft_output_buffer_size",
        "i32(i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.RFFT_TEMPORARY,
        "arm_rfft_tmp_buffer_size",
        "i32(i32,i32,i32,i32)",
    ),
    TransformQueryProtocol(
        TransformQueryKind.RIFFT_INPUT,
        "arm_rifft_input_buffer_size",
        "i32(i32,i32,i32)",
    ),
)
_TRANSFORM_QUERY_BY_CALL = {
    (protocol.symbol, protocol.signature): protocol
    for protocol in _TRANSFORM_QUERY_PROTOCOLS
}

_LIFECYCLE_PROTOCOLS = (
    LifecycleProtocol(
        LifecycleKind.MATRIX_INIT,
        "arm_mat_init_f16",
        "void(ptr,i16,i16,ptr)",
        "arm_matrix_instance_f16",
        "float16_t",
        "matrix_functions_f16.h",
        corpus_inventory.STANDARD_FLOAT16_TARGET_PROFILE,
    ),
    LifecycleProtocol(
        LifecycleKind.MATRIX_INIT,
        "arm_mat_init_f32",
        "void(ptr,i16,i16,ptr)",
        "arm_matrix_instance_f32",
        "float32_t",
        "matrix_functions.h",
        corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
    ),
    LifecycleProtocol(
        LifecycleKind.MATRIX_INIT,
        "arm_mat_init_f64",
        "void(ptr,i16,i16,ptr)",
        "arm_matrix_instance_f64",
        "float64_t",
        "matrix_functions.h",
        corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
    ),
    LifecycleProtocol(
        LifecycleKind.MATRIX_INIT,
        "arm_mat_init_q15",
        "void(ptr,i16,i16,ptr)",
        "arm_matrix_instance_q15",
        "q15_t",
        "matrix_functions.h",
        corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
    ),
    LifecycleProtocol(
        LifecycleKind.MATRIX_INIT,
        "arm_mat_init_q31",
        "void(ptr,i16,i16,ptr)",
        "arm_matrix_instance_q31",
        "q31_t",
        "matrix_functions.h",
        corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
    ),
    LifecycleProtocol(
        LifecycleKind.MATRIX_INIT,
        "arm_mat_init_q7",
        "void(ptr,i16,i16,ptr)",
        "arm_matrix_instance_q7",
        "q7_t",
        "matrix_functions.h",
        corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
    ),
    LifecycleProtocol(
        LifecycleKind.PID_RESET,
        "arm_pid_reset_f32",
        "void(ptr)",
        "arm_pid_instance_f32",
        "float32_t",
        "controller_functions.h",
        corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
    ),
    LifecycleProtocol(
        LifecycleKind.PID_RESET,
        "arm_pid_reset_q15",
        "void(ptr)",
        "arm_pid_instance_q15",
        "q15_t",
        "controller_functions.h",
        corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
    ),
    LifecycleProtocol(
        LifecycleKind.PID_RESET,
        "arm_pid_reset_q31",
        "void(ptr)",
        "arm_pid_instance_q31",
        "q31_t",
        "controller_functions.h",
        corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE,
    ),
)
_LIFECYCLE_BY_CALL = {
    (protocol.symbol, protocol.signature): protocol for protocol in _LIFECYCLE_PROTOCOLS
}


def transform_query_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> TransformQueryProtocol | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspGeneratedWorkloadProducer):
        return None
    if (
        workload.suite != "cmsis-dsp"
        or workload.target_profile != corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        or producer.selector_kind != "transform-query"
        or len(workload.protocol) != 1
    ):
        return None
    call = workload.protocol[0]
    protocol = _TRANSFORM_QUERY_BY_CALL.get((call.symbol, call.signature))
    if protocol is None:
        return None
    if workload.vector_identity != f"transform-query:{protocol.symbol}:0":
        return None
    return protocol


def lifecycle_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> LifecycleProtocol | None:
    producer = workload.producer
    if not isinstance(producer, corpus_inventory.CmsisDspGeneratedWorkloadProducer):
        return None
    if (
        workload.suite != "cmsis-dsp"
        or producer.selector_kind != "lifecycle-completion"
        or len(workload.protocol) != 1
    ):
        return None
    call = workload.protocol[0]
    protocol = _LIFECYCLE_BY_CALL.get((call.symbol, call.signature))
    if protocol is None or workload.target_profile != protocol.target_profile:
        return None
    if workload.vector_identity != f"lifecycle-completion:{protocol.symbol}:0":
        return None
    return protocol


def _query_body(kind: TransformQueryKind) -> str:
    if kind in {TransformQueryKind.CFFT_OUTPUT, TransformQueryKind.CIFFT_OUTPUT}:
        symbol = (
            "arm_cfft_output_buffer_size"
            if kind is TransformQueryKind.CFFT_OUTPUT
            else "arm_cifft_output_buffer_size"
        )
        return f"""  int failures = 0;
  failures += {symbol}(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                       sample_count) != 2 * sample_count;
  failures += {symbol}(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                       sample_count + 1) != 2 * (sample_count + 1);
  return failures != 0;
"""
    if kind is TransformQueryKind.CFFT_TEMPORARY:
        return """  int failures = 0;
  failures += arm_cfft_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                       sample_count, 1) != 0;
  failures += arm_cfft_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                                       sample_count, 2) != 0;
  return failures != 0;
"""
    if kind is TransformQueryKind.RFFT_OUTPUT:
        return """  int failures = 0;
  failures += arm_rfft_output_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                          sample_count) != sample_count;
  failures += arm_rfft_output_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                                          sample_count) != 2 * sample_count;
  return failures != 0;
"""
    if kind is TransformQueryKind.RFFT_TEMPORARY:
        return """  int failures = 0;
  failures += arm_rfft_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                       sample_count, 1) != 0;
  failures += arm_rfft_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                                       sample_count, 2) != 0;
  return failures != 0;
"""
    if kind is TransformQueryKind.RIFFT_INPUT:
        return """  int failures = 0;
  failures += arm_rifft_input_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                          sample_count) != sample_count;
  failures += arm_rifft_input_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                                          sample_count) != sample_count + 2;
  return failures != 0;
"""
    if kind is TransformQueryKind.MFCC_TEMPORARY:
        return """  const std::uint32_t buf_id = 1;
  const std::uint32_t use_cfft = 1;
  int failures = 0;
  failures += arm_mfcc_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                       sample_count, buf_id,
                                       use_cfft) != 2 * sample_count;
  failures += arm_mfcc_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                       sample_count, buf_id, 0) != sample_count;
  failures += arm_mfcc_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_Q15,
                                       sample_count, buf_id,
                                       0) != 2 * sample_count;
  failures += arm_mfcc_tmp_buffer_size(ARM_MATH_SCALAR_ARCH, ARM_MATH_F32,
                                       sample_count, 2, 0) != 0;
  return failures != 0;
"""
    raise WorkloadProviderError(f"unknown transform query kind: {kind.value}")


def render_transform_query_protocol(
    workload: corpus_inventory.ProgramWorkload,
    protocol_symbol: str,
) -> str:
    protocol = transform_query_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no transform query provider: {workload.identity}"
        )
    body = _query_body(protocol.kind)
    return f"""#include <cstdint>

#include "dsp/transform_functions.h"

#if defined(__GNUC__) || defined(__clang__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

extern "C" {{
std::uint32_t loom_corpus_sample_count = 64U;
}}

extern "C" LOOM_NOINLINE int {protocol_symbol}(
    std::uint32_t sample_count) {{
{body}}}

int main() {{
  return {protocol_symbol}(loom_corpus_sample_count);
}}
"""


def render_lifecycle_protocol(
    workload: corpus_inventory.ProgramWorkload,
    protocol_symbol: str,
) -> str:
    protocol = lifecycle_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no lifecycle provider: {workload.identity}"
        )
    if protocol.kind is LifecycleKind.MATRIX_INIT:
        body = f"""  {protocol.value_type} data[6];
  {protocol.instance_type} instance;
  {protocol.symbol}(&instance, rows, columns, data);
  int failures = 0;
  failures += instance.numRows != rows;
  failures += instance.numCols != columns;
  failures += instance.pData != data;
  return failures != 0;
"""
        parameters = "std::uint16_t rows, std::uint16_t columns"
        globals_and_main = f"""extern "C" {{
std::uint16_t loom_corpus_rows = 2U;
std::uint16_t loom_corpus_columns = 3U;
}}

int main() {{
  return {protocol_symbol}(loom_corpus_rows, loom_corpus_columns);
}}
"""
    elif protocol.kind is LifecycleKind.PID_RESET:
        body = f"""  {protocol.instance_type} instance;
  for (std::size_t index = 0; index < 3; ++index) {{
    instance.state[index] = static_cast<{protocol.value_type}>(state_seed + index);
  }}
  {protocol.symbol}(&instance);
  for (std::size_t index = 0; index < 3; ++index) {{
    if (instance.state[index] != 0) {{
      return 1;
    }}
  }}
  return 0;
"""
        parameters = "std::int32_t state_seed"
        globals_and_main = f"""extern "C" {{
std::int32_t loom_corpus_state_seed = 1;
}}

int main() {{
  return {protocol_symbol}(loom_corpus_state_seed);
}}
"""
    else:
        raise WorkloadProviderError(f"unknown lifecycle kind: {protocol.kind.value}")

    return f"""#include <cstddef>
#include <cstdint>

#include "dsp/{protocol.owner_header}"

#if defined(__GNUC__) || defined(__clang__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif

extern "C" LOOM_NOINLINE int {protocol_symbol}({parameters}) {{
{body}}}

{globals_and_main}"""
