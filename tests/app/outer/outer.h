// Loom kernel: outer
#ifndef OUTER_H
#define OUTER_H

#include <cstdint>
#include <cstddef>

void outer_cpu(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ C, const uint32_t M, const uint32_t N);

void outer_dsa(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ C, const uint32_t M, const uint32_t N);

#endif // OUTER_H
