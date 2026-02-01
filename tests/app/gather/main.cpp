#include <cstdio>

#include "gather.h"

int main() {
    const uint32_t N = 1024;
    const uint32_t src_size = 256;
    
    // Source array
    uint32_t src[src_size];
    
    // Index array
    uint32_t indices[N];
    
    // Output arrays
    uint32_t expect_dst[N];
    uint32_t calculated_dst[N];
    
    // Initialize source array
    for (uint32_t i = 0; i < src_size; i++) {
        src[i] = i * 2;
    }
    
    // Initialize index array
    for (uint32_t i = 0; i < N; i++) {
        indices[i] = (i * 3) % src_size;
    }
    
    // Compute expected result with CPU version
    gather_cpu(src, indices, expect_dst, N, src_size);
    
    // Compute result with accelerator version
    gather_dsa(src, indices, calculated_dst, N, src_size);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_dst[i] != calculated_dst[i]) {
            printf("gather: FAILED\n");
            return 1;
        }
    }
    
    printf("gather: PASSED\n");
    return 0;
}

