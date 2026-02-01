#include <cstdio>

#include "sort_bubble.h"
#include <cmath>

int main() {
    const uint32_t N = 256;  // Use smaller size for bubble sort
    
    // Allocate and initialize input
    float input[N];
    for (uint32_t i = 0; i < N; i++) {
        input[i] = static_cast<float>(N - i);  // Reverse order
    }
    
    // Allocate output arrays
    float expect_output[N];
    float calculated_output[N];
    
    // Compute expected result with CPU version
    sort_bubble_cpu(input, expect_output, N);
    
    // Compute result with DSA version
    sort_bubble_dsa(input, calculated_output, N);
    
    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-5f) {
            printf("sort_bubble: FAILED\n");
            return 1;
        }
    }
    
    // Verify sortedness
    for (uint32_t i = 0; i < N - 1; i++) {
        if (expect_output[i] > expect_output[i + 1]) {
            printf("sort_bubble: FAILED\n");
            return 1;
        }
    }
    
    printf("sort_bubble: PASSED\n");
    return 0;
}

