// Loom kernel implementation: sort_insertion
#include "sort_insertion.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Insertion sort
// Tests complete compilation chain with while loop and shifting
// Test: [3.0, 1.0, 4.0, 2.0] → [1.0, 2.0, 3.0, 4.0]

// CPU implementation of insertion sort
void sort_insertion_cpu(const float* __restrict__ input,
                        float* __restrict__ output,
                        const uint32_t N) {
    // Copy input to output
    for (uint32_t i = 0; i < N; i++) {
        output[i] = input[i];
    }

    // Insertion sort
    for (uint32_t i = 1; i < N; i++) {
        float key = output[i];
        int32_t j = static_cast<int32_t>(i) - 1;

        while (j >= 0 && output[j] > key) {
            output[j + 1] = output[j];
            j--;
        }
        output[j + 1] = key;
    }
}

// Accelerator implementation of insertion sort
LOOM_ACCEL()
void sort_insertion_dsa(const float* __restrict__ input,
                        float* __restrict__ output,
                        const uint32_t N) {
    // Copy input to output
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < N; i++) {
        output[i] = input[i];
    }

    // Insertion sort
    for (uint32_t i = 1; i < N; i++) {
        float key = output[i];
        int32_t j = static_cast<int32_t>(i) - 1;

        while (j >= 0 && output[j] > key) {
            output[j + 1] = output[j];
            j--;
        }
        output[j + 1] = key;
    }
}

