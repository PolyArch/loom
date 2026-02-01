// Loom kernel implementation: upsample
#include "upsample.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Upsampling (zero insertion)
// Tests complete compilation chain with two loops: initialization and strided write (opposite of downsample)

// CPU implementation of upsampling (zero insertion)
// Insert (factor-1) zeros between each sample
// Output size = input_size * factor
void upsample_cpu(const float* __restrict__ input,
                  float* __restrict__ output,
                  const uint32_t input_size,
                  const uint32_t factor) {
    uint32_t output_size = input_size * factor;

    // Initialize all outputs to zero
    for (uint32_t i = 0; i < output_size; i++) {
        output[i] = 0.0f;
    }

    // Place input samples at every factor-th position
    for (uint32_t i = 0; i < input_size; i++) {
        output[i * factor] = input[i];
    }
}

// Upsample: output[i*factor] = input[i], other positions = 0 (strided write with zero-fill)
// Accelerator implementation of upsampling
LOOM_ACCEL()
void upsample_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const float* __restrict__ input,
                  LOOM_STREAM float* __restrict__ output,
                  const uint32_t input_size,
                  const uint32_t factor) {
    uint32_t output_size = input_size * factor;

    for (uint32_t i = 0; i < output_size; i++) {
        output[i] = 0.0f;
    }

    for (uint32_t i = 0; i < input_size; i++) {
        output[i * factor] = input[i];
    }
}

