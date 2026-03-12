// Loom kernel implementation: database_join
#include "database_join.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Database nested loop join
// Tests complete compilation chain with nested loops and conditional matching
// Test: A_ids=[1,2,3], B_ids=[2,3,4], A_vals=[10,20,30], B_vals=[200,300,400]
//       Matches: (2,20,200), (3,30,300) -> output_ids=[2,3], output_a_values=[20,30], output_b_values=[200,300]

// CPU implementation of database nested loop join
// Joins two tables on matching IDs, outputting both values side by side
// Returns the number of matching rows written to output
// TODO: Implement O(n + m) version in the future
uint32_t database_join_cpu(const int32_t* __restrict__ a_ids,
                           const int32_t* __restrict__ b_ids,
                           const int32_t* __restrict__ a_values,
                           const int32_t* __restrict__ b_values,
                           int32_t* __restrict__ output_ids,
                           int32_t* __restrict__ output_a_values,
                           int32_t* __restrict__ output_b_values,
                           const uint32_t size_a,
                           const uint32_t size_b) {
    uint32_t out_idx = 0;
    for (uint32_t i = 0; i < size_a; i++) {
        for (uint32_t j = 0; j < size_b; j++) {
            if (a_ids[i] == b_ids[j]) {
                output_ids[out_idx] = a_ids[i];
                output_a_values[out_idx] = a_values[i];
                output_b_values[out_idx] = b_values[j];
                out_idx++;
            }
        }
    }
    return out_idx;
}

// Database nested loop join: output rows where a_ids[i] == b_ids[j]
// Accelerator implementation of database nested loop join
LOOM_ACCEL()
uint32_t database_join_dsa(const int32_t* __restrict__ a_ids,
                           const int32_t* __restrict__ b_ids,
                           const int32_t* __restrict__ a_values,
                           const int32_t* __restrict__ b_values,
                           int32_t* __restrict__ output_ids,
                           int32_t* __restrict__ output_a_values,
                           int32_t* __restrict__ output_b_values,
                           const uint32_t size_a,
                           const uint32_t size_b) {
    uint32_t out_idx = 0;
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < size_a; i++) {
        for (uint32_t j = 0; j < size_b; j++) {
            if (a_ids[i] == b_ids[j]) {
                output_ids[out_idx] = a_ids[i];
                output_a_values[out_idx] = a_values[i];
                output_b_values[out_idx] = b_values[j];
                out_idx++;
            }
        }
    }
    return out_idx;
}
