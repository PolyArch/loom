#include <cstdio>

#include "variance.h"
#include <cmath>

int main() {
    const uint32_t N = 1024;
    
    // Allocate and initialize input
    float input[N];
    for (uint32_t i = 0; i < N; i++) {
        input[i] = static_cast<float>(i % 100) + 0.5f;
    }
    
    // Compute expected result with CPU version
    float expect_result = variance_cpu(input, N);
    
    // Compute result with DSA version
    float calculated_result = variance_dsa(input, N);
    
    // Compare results with tolerance
    if (fabsf(expect_result - calculated_result) > 1e-3f) {
        printf("variance: FAILED\n");
        return 1;
    }
    
    printf("variance: PASSED\n");
    return 0;
}

