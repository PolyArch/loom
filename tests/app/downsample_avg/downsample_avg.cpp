// Loom kernel implementation: downsample_avg
#include "downsample_avg.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Downsampling with averaging
// Tests complete compilation chain with averaged decimation
// Test: input=[1,2,3,4,5,6,7,8], factor=2 → output=[1.5,3.5,5.5,7.5]

// CPU implementation of downsampling with averaging
// Average consecutive samples before decimation
void downsample_avg_cpu(const float* __restrict__ input,
                        float* __restrict__ output,
                        const uint32_t input_size,
                        const uint32_t factor) {
    uint32_t output_size = input_size / factor;

    for (uint32_t i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < factor; j++) {
            sum += input[i * factor + j];
        }
        output[i] = sum / factor;
    }
}

// Accelerator implementation of downsampling with averaging
LOOM_ACCEL()
void downsample_avg_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const float* __restrict__ input,
                        LOOM_STREAM float* __restrict__ output,
                        const uint32_t input_size,
                        const uint32_t factor) {
    uint32_t output_size = input_size / factor;

    for (uint32_t i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < factor; j++) {
            sum += input[i * factor + j];
        }
        output[i] = sum / factor;
    }
}

