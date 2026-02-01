// Loom kernel implementation: convolve_1d
#include "convolve_1d.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: 1D convolution (valid mode)
// Tests complete compilation chain with nested loops and sliding window
// Test: input=[1,2,3,4,5,6], kernel=[1,0.5,0.25], sizes=(6,3) → output=[2.75,4.5,6.25,8]






// CPU implementation of 1D convolution (time-domain)
// NOTE: This implementation performs cross-correlation, which is commonly
// referred to as convolution in many libraries (e.g., deep learning frameworks).
// output[n] = sum(input[n+k] * kernel[k]) for k = 0 to kernel_size-1
// Output size = input_size - kernel_size + 1 (valid convolution)
void convolve_1d_cpu(const float* __restrict__ input,
                     const float* __restrict__ kernel,
                     float* __restrict__ output,
                     const uint32_t input_size,
                     const uint32_t kernel_size) {
    uint32_t output_size = input_size - kernel_size + 1;
    
    for (uint32_t n = 0; n < output_size; n++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < kernel_size; k++) {
            sum += input[n + k] * kernel[k];
        }
        output[n] = sum;
    }
}

// Accelerator implementation of 1D convolution
LOOM_ACCEL()
void convolve_1d_dsa(const float* __restrict__ input,
                     const float* __restrict__ kernel,
                     float* __restrict__ output,
                     const uint32_t input_size,
                     const uint32_t kernel_size) {
    uint32_t output_size = input_size - kernel_size + 1;
    
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t n = 0; n < output_size; n++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < kernel_size; k++) {
            sum += input[n + k] * kernel[k];
        }
        output[n] = sum;
    }
}



