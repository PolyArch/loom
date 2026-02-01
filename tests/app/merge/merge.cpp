// Loom kernel implementation: merge
#include "merge.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Merge two sorted arrays
// Tests complete compilation chain with while loops and conditional logic
// Test: [1,3,5] + [2,4,6,7] → [1,2,3,4,5,6,7]






// CPU implementation of merge two sorted arrays
void merge_cpu(const float* __restrict__ input_a,
               const float* __restrict__ input_b,
               float* __restrict__ output,
               const uint32_t N,
               const uint32_t M) {
    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t k = 0;
    
    // Merge while both arrays have elements
    while (i < N && j < M) {
        if (input_a[i] <= input_b[j]) {
            output[k++] = input_a[i++];
        } else {
            output[k++] = input_b[j++];
        }
    }
    
    // Copy remaining elements from input_a
    while (i < N) {
        output[k++] = input_a[i++];
    }
    
    // Copy remaining elements from input_b
    while (j < M) {
        output[k++] = input_b[j++];
    }
}

// Accelerator implementation of merge two sorted arrays
LOOM_ACCEL()
void merge_dsa(const float* __restrict__ input_a,
               const float* __restrict__ input_b,
               float* __restrict__ output,
               const uint32_t N,
               const uint32_t M) {
    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t k = 0;
    
    // Merge while both arrays have elements
    LOOM_PARALLEL(4, interleaved)
    LOOM_TRIPCOUNT_RANGE(10, 1000)
    while (i < N && j < M) {
        if (input_a[i] <= input_b[j]) {
            output[k++] = input_a[i++];
        } else {
            output[k++] = input_b[j++];
        }
    }
    
    // Copy remaining elements from input_a
    while (i < N) {
        output[k++] = input_a[i++];
    }
    
    // Copy remaining elements from input_b
    while (j < M) {
        output[k++] = input_b[j++];
    }
}


