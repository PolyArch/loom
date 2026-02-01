// Loom kernel implementation: edit_distance_step
#include "edit_distance_step.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Edit distance DP step
// Tests complete compilation chain with comparison and minimum operations
// Test: left=[1,2,3], top=[1,2,3], diag=[0,1,2], chars match=[T,F,F] → result=[0,2,2]

// CPU implementation of single edit distance DP step
// Computes one cell of the edit distance DP table
// Inputs: value from left cell, top cell, diagonal cell, and two characters
// Output: edit distance value for current cell
void edit_distance_step_cpu(const uint32_t* __restrict__ input_left,
                             const uint32_t* __restrict__ input_top,
                             const uint32_t* __restrict__ input_diag,
                             const uint32_t* __restrict__ input_char_a,
                             const uint32_t* __restrict__ input_char_b,
                             uint32_t* __restrict__ output_result,
                             const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t cost = (input_char_a[i] == input_char_b[i]) ? 0 : 1;

        // Minimum of: insert, delete, substitute
        uint32_t insert_cost = input_top[i] + 1;
        uint32_t delete_cost = input_left[i] + 1;
        uint32_t subst_cost = input_diag[i] + cost;

        uint32_t min_val = std::min(insert_cost, delete_cost);
        min_val = std::min(min_val, subst_cost);

        output_result[i] = min_val;
    }
}

// Accelerator implementation of single edit distance DP step
LOOM_ACCEL()
void edit_distance_step_dsa(const uint32_t* __restrict__ input_left,
                             const uint32_t* __restrict__ input_top,
                             const uint32_t* __restrict__ input_diag,
                             const uint32_t* __restrict__ input_char_a,
                             const uint32_t* __restrict__ input_char_b,
                             uint32_t* __restrict__ output_result,
                             const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t cost = (input_char_a[i] == input_char_b[i]) ? 0 : 1;

        // Minimum of: insert, delete, substitute
        uint32_t insert_cost = input_top[i] + 1;
        uint32_t delete_cost = input_left[i] + 1;
        uint32_t subst_cost = input_diag[i] + cost;

        uint32_t min_val = std::min(insert_cost, delete_cost);
        min_val = std::min(min_val, subst_cost);

        output_result[i] = min_val;
    }
}

