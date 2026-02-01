#include <cstdio>

#include "relu.h"
#include <cmath>

int main() {
    const uint32_t N = 1024;
    
    // Input array
    float input[N];
    
    // Output arrays
    float expect_output[N];
    float calculated_output[N];
    
    // Initialize input with positive and negative values
    for (uint32_t i = 0; i < N; i++) {
        input[i] = static_cast<float>(i) - 512.0f;  // Range: -512 to 511
    }
    
    // Compute expected result with CPU version
    relu_cpu(input, expect_output, N);
    
    // Compute result with accelerator version
    relu_dsa(input, calculated_output, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-6f) {
            printf("relu: FAILED\n");
            return 1;
        }
    }
    
    printf("relu: PASSED\n");
    return 0;
}

