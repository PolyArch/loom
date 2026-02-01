// Loom kernel: quat_mult
#ifndef QUAT_MULT_H
#define QUAT_MULT_H

#include <cstdint>
#include <cstddef>

void quat_mult_cpu(const float* __restrict__ input_quat_1, const float* __restrict__ input_quat_2, float* __restrict__ output_quat, const uint32_t N);

void quat_mult_dsa(const float* __restrict__ input_quat_1, const float* __restrict__ input_quat_2, float* __restrict__ output_quat, const uint32_t N);

#endif // QUAT_MULT_H
