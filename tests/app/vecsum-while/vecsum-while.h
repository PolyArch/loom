// Loom kernel: vecsum-while
#ifndef VECSUM_WHILE_H
#define VECSUM_WHILE_H

#include <cstdint>
#include <cstddef>

uint32_t vecsum_cpu(const uint32_t* __restrict__ A, const uint32_t init_value, const uint32_t N);

uint32_t vecsum_dsa(const uint32_t* __restrict__ A, const uint32_t init_value, const uint32_t N);

#endif // VECSUM_WHILE_H
