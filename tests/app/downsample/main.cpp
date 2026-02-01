#include <cstdio>

#include "downsample.h"
#include <cmath>

int main() {
    const uint32_t input_size = 256;
    const uint32_t factor = 4;
    const uint32_t output_size = input_size / factor;

    // Input signal
    float input[input_size];

    // Output arrays
    float expect_output[output_size];
    float calculated_output[output_size];

    // Initialize input signal
    for (uint32_t i = 0; i < input_size; i++) {
        input[i] = sinf(2.0f * 3.14159f * i / 32.0f) + 0.1f * cosf(2.0f * 3.14159f * i / 8.0f);
    }

    // Test simple downsampling
    downsample_cpu(input, expect_output, input_size, factor);
    downsample_dsa(input, calculated_output, input_size, factor);

    for (uint32_t i = 0; i < output_size; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-6f) {
            printf("downsample: FAILED\n");
            return 1;
        }
    }

    printf("downsample: PASSED\n");
    return 0;
}

