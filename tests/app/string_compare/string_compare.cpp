// Loom kernel implementation: string_compare
#include "string_compare.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: String comparison
// Tests complete compilation chain with early exit on mismatch
// Test: "appletea" vs "appleton" → 4294967295 (0xFFFFFFFF = -1, means A < B)
// Differs at position 6: 'e' (101) < 'o' (111)






// CPU implementation of string comparison
// Compares two strings represented as uint32_t arrays (each element is a character)
// Returns: -1 if str_a < str_b, 0 if equal, 1 if str_a > str_b
void string_compare_cpu(const uint32_t* __restrict__ input_str_a,
                        const uint32_t* __restrict__ input_str_b,
                        uint32_t* __restrict__ output_result,
                        const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        if (input_str_a[i] < input_str_b[i]) {
            *output_result = 0xFFFFFFFF; // -1 as unsigned
            return;
        } else if (input_str_a[i] > input_str_b[i]) {
            *output_result = 1;
            return;
        }
    }
    *output_result = 0; // Equal
}

// Accelerator implementation of string comparison
LOOM_NO_ACCEL
void string_compare_dsa(const uint32_t* __restrict__ input_str_a,
                        const uint32_t* __restrict__ input_str_b,
                        uint32_t* __restrict__ output_result,
                        const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        if (input_str_a[i] < input_str_b[i]) {
            *output_result = 0xFFFFFFFF; // -1 as unsigned
            return;
        } else if (input_str_a[i] > input_str_b[i]) {
            *output_result = 1;
            return;
        }
    }
    *output_result = 0; // Equal
}


