// Loom kernel: wildcard_match
#ifndef WILDCARD_MATCH_H
#define WILDCARD_MATCH_H

#include <cstdint>
#include <cstddef>

void wildcard_match_cpu(const uint32_t* __restrict__ input_text, const uint32_t* __restrict__ input_pattern, uint32_t* __restrict__ output_match, const uint32_t N, const uint32_t M);

void wildcard_match_dsa(const uint32_t* __restrict__ input_text, const uint32_t* __restrict__ input_pattern, uint32_t* __restrict__ output_match, const uint32_t N, const uint32_t M);

#endif // WILDCARD_MATCH_H
