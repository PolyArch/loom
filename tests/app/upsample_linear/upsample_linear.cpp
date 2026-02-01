// Loom kernel implementation: upsample_linear
#include "upsample_linear.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// CPU implementation of upsampling with linear interpolation
// Linearly interpolate between samples
void upsample_linear_cpu(const float* __restrict__ input,
                         float* __restrict__ output,
                         const uint32_t input_size,
                         const uint32_t factor) {
    if (input_size == 0) return;

    // Handle last segment separately
    for (uint32_t i = 0; i < input_size - 1; i++) {
        output[i * factor] = input[i];

        // Linear interpolation between input[i] and input[i+1]
        for (uint32_t j = 1; j < factor; j++) {
            float alpha = static_cast<float>(j) / factor;
            output[i * factor + j] = (1.0f - alpha) * input[i] + alpha * input[i + 1];
        }
    }

    // Last sample and padding
    output[(input_size - 1) * factor] = input[input_size - 1];
    for (uint32_t j = 1; j < factor; j++) {
        output[(input_size - 1) * factor + j] = input[input_size - 1];
    }
}

// Accelerator implementation of upsampling with linear interpolation
LOOM_ACCEL()
void upsample_linear_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const float* __restrict__ input,
                         LOOM_STREAM float* __restrict__ output,
                         const uint32_t input_size,
                         const uint32_t factor) {
    if (input_size == 0) return;

    for (uint32_t i = 0; i < input_size - 1; i++) {
        output[i * factor] = input[i];

        for (uint32_t j = 1; j < factor; j++) {
            float alpha = static_cast<float>(j) / factor;
            output[i * factor + j] = (1.0f - alpha) * input[i] + alpha * input[i + 1];
        }
    }

    output[(input_size - 1) * factor] = input[input_size - 1];
    for (uint32_t j = 1; j < factor; j++) {
        output[(input_size - 1) * factor + j] = input[input_size - 1];
    }
}

