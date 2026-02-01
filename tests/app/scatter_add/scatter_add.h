// Loom kernel: scatter_add
#ifndef SCATTER_ADD_H
#define SCATTER_ADD_H

#include <cstdint>
#include <cstddef>

void scatter_add_cpu(const uint32_t* __restrict__ src, const uint32_t* __restrict__ indices, uint32_t* __restrict__ dst, const uint32_t N, const uint32_t dst_size);

void scatter_add_dsa(const uint32_t* __restrict__ src, const uint32_t* __restrict__ indices, uint32_t* __restrict__ dst, const uint32_t N, const uint32_t dst_size);

#endif // SCATTER_ADD_H
