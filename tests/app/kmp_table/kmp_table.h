// Loom kernel: kmp_table
#ifndef KMP_TABLE_H
#define KMP_TABLE_H

#include <cstdint>
#include <cstddef>

void kmp_table_cpu(const uint32_t* __restrict__ input_pattern, uint32_t* __restrict__ output_table, const uint32_t M);

void kmp_table_dsa(const uint32_t* __restrict__ input_pattern, uint32_t* __restrict__ output_table, const uint32_t M);

#endif // KMP_TABLE_H
