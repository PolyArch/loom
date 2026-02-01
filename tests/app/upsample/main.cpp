#include <cstdio>

#include "upsample.h"
#include <cmath>

int main() {
    const uint32_t input_size = 64;
    const uint32_t factor = 4;
    const uint32_t output_size = input_size * factor;

    // Input signal
    float input[input_size];

    // Output arrays
    float expect_output[output_size];
    float calculated_output[output_size];

    // Initialize input signal
    for (uint32_t i = 0; i < input_size; i++) {
        input[i] = sinf(2.0f * 3.14159f * i / 16.0f);
    }

    // Test zero-insertion upsampling
    upsample_cpu(input, expect_output, input_size, factor);
    upsample_dsa(input, calculated_output, input_size, factor);

    for (uint32_t i = 0; i < output_size; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-6f) {
            printf("upsample: FAILED\n");
            return 1;
        }
    }

    printf("upsample: PASSED\n");
    return 0;
}

