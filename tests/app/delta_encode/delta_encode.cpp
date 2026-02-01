// Loom kernel implementation: delta_encode
#include "delta_encode.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Delta encoding (compute differences)
// Tests complete compilation chain with difference operation
// Test: input=[100,101,103,106,110], N=5 → output_deltas=[100,1,2,3,4]

// CPU implementation of delta encoding
void delta_encode_cpu(const uint32_t* __restrict__ input_data,
                      uint32_t* __restrict__ output_deltas,
                      const uint32_t N) {
    if (N == 0) {
        return;
    }

    // First element stays the same
    output_deltas[0] = input_data[0];

    // Subsequent elements are differences
    for (uint32_t i = 1; i < N; i++) {
        output_deltas[i] = input_data[i] - input_data[i - 1];
    }
}

// Accelerator implementation of delta encoding
LOOM_ACCEL()
void delta_encode_dsa(const uint32_t* __restrict__ input_data,
                      uint32_t* __restrict__ output_deltas,
                      const uint32_t N) {
    if (N == 0) {
        return;
    }

    // First element stays the same
    output_deltas[0] = input_data[0];

    // Subsequent elements are differences
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t i = 1; i < N; i++) {
        output_deltas[i] = input_data[i] - input_data[i - 1];
    }
}

