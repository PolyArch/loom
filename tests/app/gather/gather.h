// Loom kernel: gather
#ifndef GATHER_H
#define GATHER_H

#include <cstdint>
#include <cstddef>

void gather_cpu(const uint32_t* __restrict__ src, const uint32_t* __restrict__ indices, uint32_t* __restrict__ dst, const uint32_t N, const uint32_t src_size);

void gather_dsa(const uint32_t* __restrict__ src, const uint32_t* __restrict__ indices, uint32_t* __restrict__ dst, const uint32_t N, const uint32_t src_size);

#endif // GATHER_H
