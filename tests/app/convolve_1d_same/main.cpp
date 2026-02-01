#include <cstdio>

#include "convolve_1d_same.h"
#include <cmath>

int main() {
    const uint32_t input_size = 128;
    const uint32_t kernel_size = 7;

    // Input signal
    float input[input_size];

    // Convolution kernel
    float kernel[kernel_size];

    // Output arrays (same size as input)
    float expect_output[input_size];
    float calculated_output[input_size];

    // Initialize input signal
    for (uint32_t i = 0; i < input_size; i++) {
        input[i] = sinf(2.0f * 3.14159f * i / 32.0f);
    }

    // Initialize kernel (simple moving average)
    for (uint32_t i = 0; i < kernel_size; i++) {
        kernel[i] = 1.0f / kernel_size;
    }

    // Test same-size convolution
    convolve_1d_same_cpu(input, kernel, expect_output, input_size, kernel_size);
    convolve_1d_same_dsa(input, kernel, calculated_output, input_size, kernel_size);

    for (uint32_t i = 0; i < input_size; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-5f) {
            printf("convolve_1d_same: FAILED\n");
            return 1;
        }
    }

    printf("convolve_1d_same: PASSED\n");
    return 0;
}

