// Loom kernel: trsv_lower
#ifndef TRSV_LOWER_H
#define TRSV_LOWER_H

#include <cstdint>
#include <cstddef>

void trsv_lower_cpu(const uint32_t* __restrict__ L, const uint32_t* __restrict__ b, uint32_t* __restrict__ x, const uint32_t N);

void trsv_lower_dsa(const uint32_t* __restrict__ L, const uint32_t* __restrict__ b, uint32_t* __restrict__ x, const uint32_t N);

#endif // TRSV_LOWER_H
