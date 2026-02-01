// Loom kernel: compact
#ifndef COMPACT_H
#define COMPACT_H

#include <cstdint>
#include <cstddef>

uint32_t compact_cpu(const uint32_t* __restrict__ input, uint32_t* __restrict__ output, const uint32_t N);

uint32_t compact_dsa(const uint32_t* __restrict__ input, uint32_t* __restrict__ output, const uint32_t N);

#endif // COMPACT_H
