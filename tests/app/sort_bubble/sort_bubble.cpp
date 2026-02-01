// Loom kernel implementation: sort_bubble
#include "sort_bubble.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Bubble sort
// Tests complete compilation chain with nested loops and swapping
// Test: [3.0, 1.0, 4.0, 2.0] → [1.0, 2.0, 3.0, 4.0]

// CPU implementation of bubble sort
void sort_bubble_cpu(const float* __restrict__ input,
                     float* __restrict__ output,
                     const uint32_t N) {
    // Copy input to output
    for (uint32_t i = 0; i < N; i++) {
        output[i] = input[i];
    }

    // Bubble sort
    for (uint32_t i = 0; i < N - 1; i++) {
        for (uint32_t j = 0; j < N - i - 1; j++) {
            if (output[j] > output[j + 1]) {
                float temp = output[j];
                output[j] = output[j + 1];
                output[j + 1] = temp;
            }
        }
    }
}

// Accelerator implementation of bubble sort
LOOM_ACCEL()
void sort_bubble_dsa(const float* __restrict__ input,
                     float* __restrict__ output,
                     const uint32_t N) {
    // Copy input to output
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < N; i++) {
        output[i] = input[i];
    }

    // Bubble sort
    for (uint32_t i = 0; i < N - 1; i++) {
        for (uint32_t j = 0; j < N - i - 1; j++) {
            if (output[j] > output[j + 1]) {
                float temp = output[j];
                output[j] = output[j + 1];
                output[j + 1] = temp;
            }
        }
    }
}

