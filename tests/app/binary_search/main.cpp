#include <cstdio>

#include "binary_search.h"

int main() {
    const uint32_t N = 10;
    const uint32_t M = 5;
    
    // Sorted input array
    float input_sorted[N] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f, 17.0f, 19.0f};
    
    // Target values to search for
    float input_targets[M] = {7.0f, 2.0f, 15.0f, 20.0f, 1.0f};
    
    // Output arrays
    uint32_t expect_indices[M];
    uint32_t calculated_indices[M];
    
    // Compute expected result with CPU version
    binary_search_cpu(input_sorted, input_targets, expect_indices, N, M);
    
    // Compute result with DSA version
    binary_search_dsa(input_sorted, input_targets, calculated_indices, N, M);
    
    // Compare results
    for (uint32_t i = 0; i < M; i++) {
        if (expect_indices[i] != calculated_indices[i]) {
            printf("binary_search: FAILED\n");
            return 1;
        }
    }
    
    printf("binary_search: PASSED\n");
    return 0;
}

