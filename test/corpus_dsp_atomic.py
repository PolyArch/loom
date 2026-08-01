#!/usr/bin/env python3
"""Typed atomic protocols for multi-call CMSIS-DSP workloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import corpus_inventory
from corpus_workload_errors import WorkloadProviderError


class SupportProtocolKind(Enum):
    SORT = "sort"
    MERGE_SORT = "merge-sort"
    SPLINE = "spline"
    DTW = "dtw"


@dataclass(frozen=True)
class AtomicProtocol:
    kind: SupportProtocolKind
    test_class: str
    test_method: str
    calls: tuple[tuple[str, str], ...]
    owner_header: str


_PROTOCOLS = (
    AtomicProtocol(
        SupportProtocolKind.SORT,
        "SupportTestsF32",
        "test_bitonic_sort_const_f32",
        (
            ("arm_sort_init_f32", "void(ptr,i32,i32)"),
            ("arm_sort_f32", "void(ptr,ptr,ptr,i32)"),
        ),
        "support_functions.h",
    ),
    AtomicProtocol(
        SupportProtocolKind.MERGE_SORT,
        "SupportTestsF32",
        "test_merge_sort_const_f32",
        (
            ("arm_merge_sort_init_f32", "void(ptr,i32,ptr)"),
            ("arm_merge_sort_f32", "void(ptr,ptr,ptr,i32)"),
        ),
        "support_functions.h",
    ),
    AtomicProtocol(
        SupportProtocolKind.SPLINE,
        "InterpolationTestsF32",
        "test_spline_ramp_f32",
        (
            ("arm_spline_init_f32", "void(ptr,i32,ptr,ptr,i32,ptr,ptr)"),
            ("arm_spline_f32", "void(ptr,ptr,ptr,i32)"),
        ),
        "interpolation_functions.h",
    ),
    AtomicProtocol(
        SupportProtocolKind.DTW,
        "DistanceTestsF32",
        "test_dtw_distance_f32",
        (
            ("arm_dtw_distance_f32", "i32(ptr,ptr,ptr,ptr)"),
            ("arm_dtw_path_f32", "void(ptr,ptr,ptr)"),
            ("arm_dtw_init_window_q7", "i32(i32,i32,ptr)"),
        ),
        "distance_functions.h",
    ),
)


def atomic_protocol(
    workload: corpus_inventory.ProgramWorkload,
) -> AtomicProtocol | None:
    producer = workload.producer
    if (
        workload.suite != "cmsis-dsp"
        or workload.target_profile
        != corpus_inventory.PORTABLE_SCALAR_TARGET_PROFILE
        or not isinstance(producer, corpus_inventory.CmsisDspWorkloadProducer)
        or producer.selector_kind != "official"
        or producer.vector_ordinal != 0
    ):
        return None
    calls = tuple((call.symbol, call.signature) for call in workload.protocol)
    for protocol in _PROTOCOLS:
        if (
            producer.test_class == protocol.test_class
            and producer.test_method == protocol.test_method
            and calls == protocol.calls
        ):
            return protocol
    return None


def _preamble(owner_header: str) -> str:
    return f"""#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dsp/{owner_header}"

#if defined(__clang__) || defined(__GNUC__)
#define LOOM_NOINLINE __attribute__((noinline))
#else
#define LOOM_NOINLINE
#endif
"""


def _render_sort(protocol_symbol: str) -> str:
    return f"""{_preamble("support_functions.h")}
namespace {{
constexpr std::size_t kCount = 8;
constexpr float32_t kInput[kCount] = {{3.0f, -1.0f, 2.0f, 0.0f,
                                      7.0f, 4.0f, -5.0f, 6.0f}};
constexpr float32_t kExpected[kCount] = {{-5.0f, -1.0f, 0.0f, 2.0f,
                                         3.0f, 4.0f, 6.0f, 7.0f}};
}}

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    arm_sort_instance_f32 *instance, float32_t *input, float32_t *output) {{
  arm_sort_init_f32(instance, ARM_SORT_BITONIC, ARM_SORT_ASCENDING);
  arm_sort_f32(instance, input, output, kCount);
}}

int main() {{
  arm_sort_instance_f32 instance;
  float32_t input[kCount];
  float32_t output[kCount]{{}};
  for (std::size_t index = 0; index < kCount; ++index) input[index] = kInput[index];
  {protocol_symbol}(&instance, input, output);
  for (std::size_t index = 0; index < kCount; ++index)
    if (output[index] != kExpected[index]) return 1;
  return 0;
}}
"""


def _render_merge_sort(protocol_symbol: str) -> str:
    return f"""{_preamble("support_functions.h")}
namespace {{
constexpr std::size_t kCount = 7;
constexpr float32_t kInput[kCount] = {{3.0f, -1.0f, 2.0f, 0.0f, 7.0f, -5.0f, 6.0f}};
constexpr float32_t kExpected[kCount] = {{-5.0f, -1.0f, 0.0f, 2.0f, 3.0f, 6.0f, 7.0f}};
}}

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    arm_merge_sort_instance_f32 *instance, float32_t *input,
    float32_t *output, float32_t *scratch) {{
  arm_merge_sort_init_f32(instance, ARM_SORT_ASCENDING, scratch);
  arm_merge_sort_f32(instance, input, output, kCount);
}}

int main() {{
  arm_merge_sort_instance_f32 instance;
  float32_t input[kCount];
  float32_t output[kCount]{{}};
  float32_t scratch[kCount]{{}};
  for (std::size_t index = 0; index < kCount; ++index) input[index] = kInput[index];
  {protocol_symbol}(&instance, input, output, scratch);
  for (std::size_t index = 0; index < kCount; ++index)
    if (output[index] != kExpected[index]) return 1;
  return 0;
}}
"""


def _render_spline(protocol_symbol: str) -> str:
    return f"""{_preamble("interpolation_functions.h")}
namespace {{
constexpr std::size_t kKnownCount = 3;
constexpr std::size_t kQueryCount = 5;
constexpr float32_t kKnownX[kKnownCount] = {{0.0f, 1.0f, 2.0f}};
constexpr float32_t kKnownY[kKnownCount] = {{1.0f, 3.0f, 5.0f}};
constexpr float32_t kQueryX[kQueryCount] = {{0.0f, 0.5f, 1.0f, 1.5f, 2.0f}};
constexpr float32_t kExpected[kQueryCount] = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}};
}}

extern "C" LOOM_NOINLINE void {protocol_symbol}(
    arm_spline_instance_f32 *instance, float32_t *coefficients,
    float32_t *scratch, float32_t *output) {{
  arm_spline_init_f32(instance, ARM_SPLINE_PARABOLIC_RUNOUT,
                      kKnownX, kKnownY, kKnownCount, coefficients, scratch);
  arm_spline_f32(instance, kQueryX, output, kQueryCount);
}}

int main() {{
  arm_spline_instance_f32 instance;
  float32_t coefficients[3 * (kKnownCount - 1)]{{}};
  float32_t scratch[2 * kKnownCount - 1]{{}};
  float32_t output[kQueryCount]{{}};
  {protocol_symbol}(&instance, coefficients, scratch, output);
  for (std::size_t index = 0; index < kQueryCount; ++index)
    if (std::fabs(output[index] - kExpected[index]) > 1.0e-6f) return 1;
  return 0;
}}
"""


def _render_dtw(protocol_symbol: str) -> str:
    return f"""{_preamble("distance_functions.h")}
namespace {{
constexpr std::size_t kExtent = 2;
constexpr float32_t kDistances[kExtent * kExtent] = {{0.0f, 4.0f, 4.0f, 0.0f}};
}}

extern "C" LOOM_NOINLINE arm_status {protocol_symbol}(
    arm_matrix_instance_f32 *distances, arm_matrix_instance_f32 *costs,
    arm_matrix_instance_q7 *window,
    float32_t *distance, int16_t *path, std::uint32_t *path_length) {{
  const arm_status distance_status =
      arm_dtw_distance_f32(distances, nullptr, costs, distance);
  if (distance_status != ARM_MATH_SUCCESS) return distance_status;
  arm_dtw_path_f32(costs, path, path_length);
  return arm_dtw_init_window_q7(ARM_DTW_SAKOE_CHIBA_WINDOW, 0, window);
}}

int main() {{
  float32_t costs_data[kExtent * kExtent]{{}};
  arm_matrix_instance_f32 distances{{kExtent, kExtent,
      const_cast<float32_t *>(kDistances)}};
  arm_matrix_instance_f32 costs{{kExtent, kExtent, costs_data}};
  float32_t distance = -1.0f;
  int16_t path[2 * (kExtent + kExtent)]{{}};
  std::uint32_t path_length = 0;
  q7_t window_data[kExtent * kExtent]{{}};
  arm_matrix_instance_q7 window{{kExtent, kExtent, window_data}};
  const arm_status status =
      {protocol_symbol}(&distances, &costs, &window,
                        &distance, path, &path_length);
  if (status != ARM_MATH_SUCCESS || distance != 0.0f || path_length != 2) return 1;
  if (path[0] != 0 || path[1] != 0 || path[2] != 1 || path[3] != 1) return 1;
  const bool window_matches =
      window_data[0] == 1 && window_data[1] == 0 &&
      window_data[2] == 0 && window_data[3] == 1;
  return window_matches ? 0 : 1;
}}
"""


def render_atomic_protocol(
    workload: corpus_inventory.ProgramWorkload, protocol_symbol: str
) -> str:
    protocol = atomic_protocol(workload)
    if protocol is None:
        raise WorkloadProviderError(
            f"CMSIS-DSP workload has no atomic protocol: {workload.identity}"
        )
    if protocol.kind is SupportProtocolKind.SORT:
        return _render_sort(protocol_symbol)
    if protocol.kind is SupportProtocolKind.MERGE_SORT:
        return _render_merge_sort(protocol_symbol)
    if protocol.kind is SupportProtocolKind.SPLINE:
        return _render_spline(protocol_symbol)
    if protocol.kind is SupportProtocolKind.DTW:
        return _render_dtw(protocol_symbol)
    raise WorkloadProviderError("unknown CMSIS-DSP atomic protocol kind")
