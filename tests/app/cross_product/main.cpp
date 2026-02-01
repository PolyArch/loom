#include <cstdio>

#include "cross_product.h"
#include <cmath>

int main() {
    const uint32_t N = 64;
    
    // Input vectors (N pairs of 3D vectors)
    float vec_a[N * 3];
    float vec_b[N * 3];
    
    for (uint32_t i = 0; i < N; i++) {
        vec_a[i * 3 + 0] = 1.0f + i * 0.1f;
        vec_a[i * 3 + 1] = 0.0f;
        vec_a[i * 3 + 2] = 0.0f;
        
        vec_b[i * 3 + 0] = 0.0f;
        vec_b[i * 3 + 1] = 1.0f + i * 0.1f;
        vec_b[i * 3 + 2] = 0.0f;
    }
    
    // Output arrays
    float expect_result[N * 3];
    float calculated_result[N * 3];
    
    // Compute expected result with CPU version
    cross_product_cpu(vec_a, vec_b, expect_result, N);
    
    // Compute result with accelerator version
    cross_product_dsa(vec_a, vec_b, calculated_result, N);
    
    // Compare results with tolerance
    for (uint32_t i = 0; i < N * 3; i++) {
        if (fabsf(expect_result[i] - calculated_result[i]) > 1e-5f) {
            printf("cross_product: FAILED\n");
            return 1;
        }
    }
    
    printf("cross_product: PASSED\n");
    return 0;
}

