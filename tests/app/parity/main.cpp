#include <cstdio>

#include "parity.h"

int main() {
    const uint32_t N = 256;
    
    // Input data with various bit patterns
    uint32_t input[N];
    input[0] = 0x00000000;  // 0 bits set -> even parity (0)
    input[1] = 0x00000001;  // 1 bit set -> odd parity (1)
    input[2] = 0x00000003;  // 2 bits set -> even parity (0)
    input[3] = 0x00000007;  // 3 bits set -> odd parity (1)
    input[4] = 0xFFFFFFFF;  // 32 bits set -> even parity (0)
    
    for (uint32_t i = 5; i < N; i++) {
        input[i] = i * 0x9ABCDEF0;
    }
    
    // Output arrays
    uint32_t expect_parity[N];
    uint32_t calculated_parity[N];
    
    // Compute expected result with CPU version
    parity_cpu(input, expect_parity, N);
    
    // Compute result with accelerator version
    parity_dsa(input, calculated_parity, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_parity[i] != calculated_parity[i]) {
            printf("parity: FAILED\n");
            return 1;
        }
    }
    
    printf("parity: PASSED\n");
    return 0;
}

