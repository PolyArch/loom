#include <cstdio>

#include "moving_avg.h"
#include <cmath>

int main() {
    const uint32_t N = 1024;
    const uint32_t window_size = 5;
    
    // Allocate and initialize input
    float input[N];
    for (uint32_t i = 0; i < N; i++) {
        input[i] = static_cast<float>(i % 100);
    }
    
    // Allocate output arrays
    float expect_output[N];
    float calculated_output[N];
    
    // Compute expected result with CPU version
    moving_avg_cpu(input, expect_output, N, window_size);
    
    // Compute result with DSA version
    moving_avg_dsa(input, calculated_output, N, window_size);
    
    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-5f) {
            printf("moving_avg: FAILED\n");
            return 1;
        }
    }
    
    printf("moving_avg: PASSED\n");
    return 0;
}

