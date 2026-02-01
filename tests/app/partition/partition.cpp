// Loom kernel implementation: partition
#include "partition.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Partition array around pivot
// Tests complete compilation chain with two-pass partitioning algorithm
// Test: [5,2,8,3,9,1] pivot=5.0 → [5,2,3,1,8,9] pivot_idx=4






// CPU implementation of partition around pivot
void partition_cpu(const float* __restrict__ input,
                   float* __restrict__ output,
                   uint32_t* __restrict__ output_pivot_idx,
                   const uint32_t N,
                   const float pivot) {
    uint32_t write_pos = 0;
    
    // First pass: write elements less than or equal to pivot
    for (uint32_t i = 0; i < N; i++) {
        if (input[i] <= pivot) {
            output[write_pos++] = input[i];
        }
    }
    
    // Record pivot position
    *output_pivot_idx = write_pos;
    
    // Second pass: write elements greater than pivot
    for (uint32_t i = 0; i < N; i++) {
        if (input[i] > pivot) {
            output[write_pos++] = input[i];
        }
    }
}

// Accelerator implementation of partition around pivot
LOOM_ACCEL()
void partition_dsa(const float* __restrict__ input,
                   float* __restrict__ output,
                   uint32_t* __restrict__ output_pivot_idx,
                   const uint32_t N,
                   const float pivot) {
    uint32_t write_pos = 0;
    
    // First pass: write elements less than or equal to pivot
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < N; i++) {
        if (input[i] <= pivot) {
            output[write_pos++] = input[i];
        }
    }
    
    // Record pivot position
    *output_pivot_idx = write_pos;
    
    // Second pass: write elements greater than pivot
    for (uint32_t i = 0; i < N; i++) {
        if (input[i] > pivot) {
            output[write_pos++] = input[i];
        }
    }
}



