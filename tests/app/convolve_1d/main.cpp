#include <cstdio>

#include "convolve_1d.h"
#include <cmath>

int main() {
    const uint32_t input_size = 128;
    const uint32_t kernel_size = 7;
    const uint32_t output_size = input_size - kernel_size + 1;
    
    // Input signal
    float input[input_size];
    
    // Convolution kernel
    float kernel[kernel_size];
    
    // Output arrays for valid convolution
    float expect_output[output_size];
    float calculated_output[output_size];
    
    // Initialize input signal
    for (uint32_t i = 0; i < input_size; i++) {
        input[i] = sinf(2.0f * 3.14159f * i / 32.0f);
    }
    
    // Initialize kernel (simple moving average)
    for (uint32_t i = 0; i < kernel_size; i++) {
        kernel[i] = 1.0f / kernel_size;
    }
    
    // Test valid convolution
    convolve_1d_cpu(input, kernel, expect_output, input_size, kernel_size);
    convolve_1d_dsa(input, kernel, calculated_output, input_size, kernel_size);
    
    for (uint32_t i = 0; i < output_size; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-5f) {
            printf("convolve_1d: FAILED\n");
            return 1;
        }
    }
    
    printf("convolve_1d: PASSED\n");
    return 0;
}

