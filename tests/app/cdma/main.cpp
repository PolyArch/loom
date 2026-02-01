#include <cstdio>

#include "cdma.h"

int main() {
    const size_t N = 1024;
    
    // Input array
    uint32_t SRC[N];
    
    // Output arrays
    uint32_t expect_DST[N];
    uint32_t calculated_DST[N];
    
    // Initialize input array
    for (size_t i = 0; i < N; i++) {
        SRC[i] = i * 3 + 7;  // Some arbitrary pattern
    }
    
    // Compute expected result with CPU version
    cdma_cpu(SRC, expect_DST, N);
    
    // Compute result with accelerator version
    cdma_dsa(SRC, calculated_DST, N);
    
    // Compare results
    for (size_t i = 0; i < N; i++) {
        if (expect_DST[i] != calculated_DST[i]) {
            printf("cdma: FAILED\n");
            return 1;
        }
    }
    
    printf("cdma: PASSED\n");
    return 0;
}

