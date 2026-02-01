// Loom kernel implementation: histogram_strided
#include "histogram_strided.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Strided histogram
// Tests complete compilation chain with division and indirect writes
// Test: input=[0,5,10,15,7,12,3,18,22,25], stride=6, 5 bins → [3,2,2,2,1]

// CPU implementation of histogram with custom bin mapping
// Maps input values to bins using a stride
// hist[input[i] / stride]++
void histogram_strided_cpu(const uint32_t* __restrict__ input,
                           uint32_t* __restrict__ hist,
                           const uint32_t N,
                           const uint32_t num_bins,
                           const uint32_t stride) {
    // Initialize histogram to zero
    for (uint32_t i = 0; i < num_bins; i++) {
        hist[i] = 0;
    }

    // Count occurrences with striding
    for (uint32_t i = 0; i < N; i++) {
        LOOM_REDUCE(+)
        uint32_t bin = input[i] / stride;
        if (bin < num_bins) {
            hist[bin]++;
        }
    }
}

// Accelerator implementation of strided histogram
LOOM_TARGET("temporal")
LOOM_ACCEL()
void histogram_strided_dsa(const uint32_t* __restrict__ input,
                           uint32_t* __restrict__ hist,
                           const uint32_t N,
                           const uint32_t num_bins,
                           const uint32_t stride) {
    // Initialize histogram to zero
    LOOM_PARALLEL(4)
    for (uint32_t i = 0; i < num_bins; i++) {
        hist[i] = 0;
    }

    // Count occurrences with striding
    for (uint32_t i = 0; i < N; i++) {
        uint32_t bin = input[i] / stride;
        if (bin < num_bins) {
            hist[bin]++;
        }
    }
}

