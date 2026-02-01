// Loom kernel implementation: convolve_1d_same
#include "convolve_1d_same.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: 1D convolution with zero-padding (same size output)
// Tests complete compilation chain with boundary checking and zero-padding
// Test: input=[1,2,3,4,5], kernel=[1,0.5,0.25], sizes=(5,3) → output=[1,2.75,4.5,6.25,6.5]






// CPU implementation of 1D convolution with zero-padding (same size output)
void convolve_1d_same_cpu(const float* __restrict__ input,
                          const float* __restrict__ kernel,
                          float* __restrict__ output,
                          const uint32_t input_size,
                          const uint32_t kernel_size) {
    int32_t pad = kernel_size / 2;
    
    for (uint32_t n = 0; n < input_size; n++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < kernel_size; k++) {
            int32_t idx = static_cast<int32_t>(n) - pad + static_cast<int32_t>(k);
            if (idx >= 0 && idx < static_cast<int32_t>(input_size)) {
                sum += input[idx] * kernel[k];
            }
        }
        output[n] = sum;
    }
}

// Accelerator implementation of 1D convolution with zero-padding
LOOM_ACCEL()
void convolve_1d_same_dsa(const float* __restrict__ input,
                          const float* __restrict__ kernel,
                          float* __restrict__ output,
                          const uint32_t input_size,
                          const uint32_t kernel_size) {
    int32_t pad = kernel_size / 2;
    
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t n = 0; n < input_size; n++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < kernel_size; k++) {
            int32_t idx = static_cast<int32_t>(n) - pad + static_cast<int32_t>(k);
            if (idx >= 0 && idx < static_cast<int32_t>(input_size)) {
                sum += input[idx] * kernel[k];
            }
        }
        output[n] = sum;
    }
}



