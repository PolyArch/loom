// Loom kernel implementation: kmp_table
#include "kmp_table.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: KMP failure function (prefix table)
// Tests complete compilation chain with while loop and array indexing
// Test: pattern "ABABAC" → table [0,0,1,2,3,0]

// CPU implementation of KMP failure function computation
// Computes the failure function (prefix table) for KMP string matching
// Input: pattern string, Output: failure function table
void kmp_table_cpu(const uint32_t* __restrict__ input_pattern,
                   uint32_t* __restrict__ output_table,
                   const uint32_t M) {
    output_table[0] = 0;
    uint32_t j = 0;

    for (uint32_t i = 1; i < M; i++) {
        while (j > 0 && input_pattern[i] != input_pattern[j]) {
            j = output_table[j - 1];
        }

        if (input_pattern[i] == input_pattern[j]) {
            j++;
        }

        output_table[i] = j;
    }
}

// Accelerator implementation of KMP failure function computation
LOOM_ACCEL()
void kmp_table_dsa(const uint32_t* __restrict__ input_pattern,
                   uint32_t* __restrict__ output_table,
                   const uint32_t M) {
    output_table[0] = 0;
    uint32_t j = 0;

    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 1; i < M; i++) {
        while (j > 0 && input_pattern[i] != input_pattern[j]) {
            j = output_table[j - 1];
        }

        if (input_pattern[i] == input_pattern[j]) {
            j++;
        }

        output_table[i] = j;
    }
}

