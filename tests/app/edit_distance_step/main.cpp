#include <cstdio>

#include "edit_distance_step.h"

int main() {
    const uint32_t N = 64;
    
    // Input DP values and characters
    uint32_t left[N];
    uint32_t top[N];
    uint32_t diag[N];
    uint32_t char_a[N];
    uint32_t char_b[N];
    
    for (uint32_t i = 0; i < N; i++) {
        left[i] = i + 1;
        top[i] = i + 2;
        diag[i] = i;
        char_a[i] = 'a' + (i % 2);
        char_b[i] = 'a' + ((i + 1) % 2);
    }
    
    // Output arrays
    uint32_t expect_result[N];
    uint32_t calculated_result[N];
    
    // Compute expected result with CPU version
    edit_distance_step_cpu(left, top, diag, char_a, char_b, expect_result, N);
    
    // Compute result with accelerator version
    edit_distance_step_dsa(left, top, diag, char_a, char_b, calculated_result, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_result[i] != calculated_result[i]) {
            printf("edit_distance_step: FAILED\n");
            return 1;
        }
    }
    
    printf("edit_distance_step: PASSED\n");
    return 0;
}

