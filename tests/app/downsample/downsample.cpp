// Loom kernel implementation: downsample
#include "downsample.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Downsampling (decimation by factor)
// Tests complete compilation chain with strided read pattern (i*factor stride, not sequential)
// Test: input=[1,2,3,4,5,6,7,8], factor=2 → output=[1,3,5,7]

// CPU implementation of downsampling (decimation)
// Take every Nth sample: output[i] = input[i * factor]
// Output size = input_size / factor
void downsample_cpu(const float* __restrict__ input,
                    float* __restrict__ output,
                    const uint32_t input_size,
                    const uint32_t factor) {
    uint32_t output_size = input_size / factor;

    for (uint32_t i = 0; i < output_size; i++) {
        output[i] = input[i * factor];
    }
}

// Downsample: output[i] = input[i * factor] (strided read with stride=factor)
// Accelerator implementation of downsampling
LOOM_ACCEL()
void downsample_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const float* __restrict__ input,
                    LOOM_STREAM float* __restrict__ output,
                    const uint32_t input_size,
                    const uint32_t factor) {
    uint32_t output_size = input_size / factor;

    for (uint32_t i = 0; i < output_size; i++) {
        output[i] = input[i * factor];
    }
}

