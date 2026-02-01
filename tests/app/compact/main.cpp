#include <cstdio>

#include "compact.h"

int main() {
    const uint32_t N = 1024;
    
    // Input array with some zeros
    uint32_t input[N];
    
    // Output arrays (large enough to hold all non-zero elements)
    uint32_t expect_output[N];
    uint32_t calculated_output[N];
    
    // Initialize input with pattern that has zeros
    for (uint32_t i = 0; i < N; i++) {
        input[i] = (i % 5 == 0) ? 0 : (i % 100);
    }
    
    // Test basic compaction (non-zero filter)
    uint32_t expect_count = compact_cpu(input, expect_output, N);
    uint32_t calculated_count = compact_dsa(input, calculated_output, N);
    
    if (expect_count != calculated_count) {
        printf("compact: FAILED\n");
        return 1;
    }
    
    for (uint32_t i = 0; i < expect_count; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("compact: FAILED\n");
            return 1;
        }
    }
    
    printf("compact: PASSED\n");
    return 0;
}

