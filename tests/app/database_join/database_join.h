// Loom kernel: database_join
#ifndef DATABASE_JOIN_H
#define DATABASE_JOIN_H

#include <cstdint>
#include <cstddef>

uint32_t database_join_cpu(const int32_t* __restrict__ a_ids, const int32_t* __restrict__ b_ids, const int32_t* __restrict__ a_values, const int32_t* __restrict__ b_values, int32_t* __restrict__ output_ids, int32_t* __restrict__ output_a_values, int32_t* __restrict__ output_b_values, const uint32_t size_a, const uint32_t size_b);

uint32_t database_join_dsa(const int32_t* __restrict__ a_ids, const int32_t* __restrict__ b_ids, const int32_t* __restrict__ a_values, const int32_t* __restrict__ b_values, int32_t* __restrict__ output_ids, int32_t* __restrict__ output_a_values, int32_t* __restrict__ output_b_values, const uint32_t size_a, const uint32_t size_b);

#endif // DATABASE_JOIN_H
