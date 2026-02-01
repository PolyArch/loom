// Loom kernel: string_compare
#ifndef STRING_COMPARE_H
#define STRING_COMPARE_H

#include <cstdint>
#include <cstddef>

void string_compare_cpu(const uint32_t* __restrict__ input_str_a, const uint32_t* __restrict__ input_str_b, uint32_t* __restrict__ output_result, const uint32_t N);

void string_compare_dsa(const uint32_t* __restrict__ input_str_a, const uint32_t* __restrict__ input_str_b, uint32_t* __restrict__ output_result, const uint32_t N);

#endif // STRING_COMPARE_H
