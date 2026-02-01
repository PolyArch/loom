#include <cstdio>

#include "delta_encode.h"

int main() {
    const uint32_t N = 10;
    
    // Input array (monotonically increasing)
    uint32_t input[N] = {100, 102, 105, 110, 115, 122, 130, 135, 142, 150};
    
    // Output arrays
    uint32_t expect_output[N];
    uint32_t calculated_output[N];
    
    // Compute expected result with CPU version
    delta_encode_cpu(input, expect_output, N);
    
    // Compute result with accelerator version
    delta_encode_dsa(input, calculated_output, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("delta_encode: FAILED\n");
            return 1;
        }
    }
    
    printf("delta_encode: PASSED\n");
    return 0;
}


