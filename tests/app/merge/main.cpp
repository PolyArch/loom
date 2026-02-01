#include <cstdio>

#include "merge.h"
#include <cmath>

int main() {
    const uint32_t N = 8;
    const uint32_t M = 6;
    
    // Two sorted input arrays
    float input_a[N] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f};
    float input_b[M] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
    
    // Output arrays
    float expect_output[N + M];
    float calculated_output[N + M];
    
    // Compute expected result with CPU version
    merge_cpu(input_a, input_b, expect_output, N, M);
    
    // Compute result with DSA version
    merge_dsa(input_a, input_b, calculated_output, N, M);
    
    // Compare results
    for (uint32_t i = 0; i < N + M; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-5f) {
            printf("merge: FAILED\n");
            return 1;
        }
    }
    
    printf("merge: PASSED\n");
    return 0;
}

