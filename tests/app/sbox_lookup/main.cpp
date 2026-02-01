#include <cstdio>

#include "sbox_lookup.h"

int main() {
    const uint32_t N = 1024;
    const uint32_t SBOX_SIZE = 256;
    
    // S-box table (256 entries)
    uint32_t sbox[SBOX_SIZE];
    
    // Input data array
    uint32_t input_data[N];
    
    // Output arrays
    uint32_t expect_output[N];
    uint32_t calculated_output[N];
    
    // Initialize S-box with a simple permutation
    for (uint32_t i = 0; i < SBOX_SIZE; i++) {
        sbox[i] = (i * 7 + 31) & 0xFF;
    }
    
    // Initialize input data
    for (uint32_t i = 0; i < N; i++) {
        input_data[i] = (i * 13 + 17) & 0xFF;
    }
    
    // Compute expected result with CPU version
    sbox_lookup_cpu(input_data, sbox, expect_output, N);
    
    // Compute result with accelerator version
    sbox_lookup_dsa(input_data, sbox, calculated_output, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("sbox_lookup: FAILED\n");
            return 1;
        }
    }
    
    printf("sbox_lookup: PASSED\n");
    return 0;
}

