// Loom kernel: batchnorm
#ifndef BATCHNORM_H
#define BATCHNORM_H

#include <cstdint>
#include <cstddef>

void batchnorm_cpu(const float* __restrict__ input, const float* __restrict__ mean, const float* __restrict__ variance, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ output, const uint32_t C, const uint32_t H, const uint32_t W, const float epsilon);

void batchnorm_dsa(const float* __restrict__ input, const float* __restrict__ mean, const float* __restrict__ variance, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ output, const uint32_t C, const uint32_t H, const uint32_t W, const float epsilon);

#endif // BATCHNORM_H
