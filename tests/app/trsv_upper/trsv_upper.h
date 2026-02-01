// Loom kernel: trsv_upper
#ifndef TRSV_UPPER_H
#define TRSV_UPPER_H

#include <cstdint>
#include <cstddef>

void trsv_upper_cpu(const uint32_t* __restrict__ U, const uint32_t* __restrict__ b, uint32_t* __restrict__ x, const uint32_t N);

void trsv_upper_dsa(const uint32_t* __restrict__ U, const uint32_t* __restrict__ b, uint32_t* __restrict__ x, const uint32_t N);

#endif // TRSV_UPPER_H
