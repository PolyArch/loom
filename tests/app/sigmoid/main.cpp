#include <cstdio>

#include "sigmoid.h"
#include <cmath>

int main() {
    const uint32_t N = 1024;
    
    // Input array
    float input[N];
    
    // Output arrays
    float expect_output[N];
    float calculated_output[N];
    
    // Initialize input with range [-5, 5]
    for (uint32_t i = 0; i < N; i++) {
        input[i] = (static_cast<float>(i) / N - 0.5f) * 10.0f;
    }
    
    // Compute expected result with CPU version
    sigmoid_cpu(input, expect_output, N);
    
    // Compute result with accelerator version
    sigmoid_dsa(input, calculated_output, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-6f) {
            printf("sigmoid: FAILED\n");
            return 1;
        }
    }
    
    printf("sigmoid: PASSED\n");
    return 0;
}

