// Loom kernel: edit_distance_step
#ifndef EDIT_DISTANCE_STEP_H
#define EDIT_DISTANCE_STEP_H

#include <cstdint>
#include <cstddef>

void edit_distance_step_cpu(const uint32_t* __restrict__ input_left, const uint32_t* __restrict__ input_top, const uint32_t* __restrict__ input_diag, const uint32_t* __restrict__ input_char_a, const uint32_t* __restrict__ input_char_b, uint32_t* __restrict__ output_result, const uint32_t N);

void edit_distance_step_dsa(const uint32_t* __restrict__ input_left, const uint32_t* __restrict__ input_top, const uint32_t* __restrict__ input_diag, const uint32_t* __restrict__ input_char_a, const uint32_t* __restrict__ input_char_b, uint32_t* __restrict__ output_result, const uint32_t N);

#endif // EDIT_DISTANCE_STEP_H
