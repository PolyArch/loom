// Loom kernel implementation: sort_merge
#include "sort_merge.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Merge sort (iterative bottom-up)
// Tests complete compilation chain with nested loops and merge operations
// Test: [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0] -> [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]

// CPU implementation of merge sort (iterative bottom-up)
void sort_merge_cpu(const float* __restrict__ input,
                    float* __restrict__ output,
                    float* __restrict__ temp,
                    const uint32_t N) {
    // Copy input to output
    for (uint32_t i = 0; i < N; i++) {
        output[i] = input[i];
    }

    if (N <= 1) return;

    // Bottom-up merge sort: merge subarrays of size 1, 2, 4, 8, ...
    for (uint32_t width = 1; width < N; width *= 2) {
        // Merge pairs of subarrays
        for (uint32_t left = 0; left < N; left += 2 * width) {
            uint32_t mid = left + width;
            uint32_t right = left + 2 * width;

            // Clamp bounds
            if (mid > N) mid = N;
            if (right > N) right = N;

            // Merge [left, mid) and [mid, right) into temp
            uint32_t i = left;
            uint32_t j = mid;
            uint32_t k = left;

            while (i < mid && j < right) {
                if (output[i] <= output[j]) {
                    temp[k++] = output[i++];
                } else {
                    temp[k++] = output[j++];
                }
            }

            // Copy remaining from left half
            while (i < mid) {
                temp[k++] = output[i++];
            }

            // Copy remaining from right half
            while (j < right) {
                temp[k++] = output[j++];
            }

            // Copy back to output
            for (uint32_t idx = left; idx < right; idx++) {
                output[idx] = temp[idx];
            }
        }
    }
}

// Accelerator implementation of merge sort (iterative bottom-up)
LOOM_ACCEL()
void sort_merge_dsa(const float* __restrict__ input,
                    float* __restrict__ output,
                    float* __restrict__ temp,
                    const uint32_t N) {
    // Copy input to output
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < N; i++) {
        output[i] = input[i];
    }

    if (N <= 1) return;

    // Bottom-up merge sort: merge subarrays of size 1, 2, 4, 8, ...
    for (uint32_t width = 1; width < N; width *= 2) {
        // Merge pairs of subarrays
        for (uint32_t left = 0; left < N; left += 2 * width) {
            uint32_t mid = left + width;
            uint32_t right = left + 2 * width;

            // Clamp bounds
            if (mid > N) mid = N;
            if (right > N) right = N;

            // Merge [left, mid) and [mid, right) into temp
            uint32_t i = left;
            uint32_t j = mid;
            uint32_t k = left;

            while (i < mid && j < right) {
                if (output[i] <= output[j]) {
                    temp[k++] = output[i++];
                } else {
                    temp[k++] = output[j++];
                }
            }

            // Copy remaining from left half
            while (i < mid) {
                temp[k++] = output[i++];
            }

            // Copy remaining from right half
            while (j < right) {
                temp[k++] = output[j++];
            }

            // Copy back to output
            for (uint32_t idx = left; idx < right; idx++) {
                output[idx] = temp[idx];
            }
        }
    }
}
