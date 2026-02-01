// Loom kernel: compact_predicate
#ifndef COMPACT_PREDICATE_H
#define COMPACT_PREDICATE_H

#include <cstdint>
#include <cstddef>

uint32_t compact_predicate_cpu(const uint32_t* __restrict__ input, const uint32_t* __restrict__ predicate, uint32_t* __restrict__ output, const uint32_t N);

uint32_t compact_predicate_dsa(const uint32_t* __restrict__ input, const uint32_t* __restrict__ predicate, uint32_t* __restrict__ output, const uint32_t N);

#endif // COMPACT_PREDICATE_H
