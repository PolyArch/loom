// Loom kernel: covariance
#ifndef COVARIANCE_H
#define COVARIANCE_H

#include <cstdint>
#include <cstddef>

float covariance_cpu(const float* __restrict__ X, const float* __restrict__ Y, const uint32_t N);

float covariance_dsa(const float* __restrict__ X, const float* __restrict__ Y, const uint32_t N);

#endif // COVARIANCE_H
