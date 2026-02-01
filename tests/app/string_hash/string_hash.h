// Loom kernel: string_hash
#ifndef STRING_HASH_H
#define STRING_HASH_H

#include <cstdint>
#include <cstddef>

void string_hash_cpu(const uint32_t* __restrict__ input_str, uint32_t* __restrict__ output_hashes, const uint32_t N, const uint32_t window_size);

void string_hash_dsa(const uint32_t* __restrict__ input_str, uint32_t* __restrict__ output_hashes, const uint32_t N, const uint32_t window_size);

#endif // STRING_HASH_H
