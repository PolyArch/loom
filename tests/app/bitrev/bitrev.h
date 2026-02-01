// Loom kernel: bitrev
#ifndef BITREV_H
#define BITREV_H

#include <cstdint>
#include <cstddef>

void bitrev_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

void bitrev_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

#endif // BITREV_H
