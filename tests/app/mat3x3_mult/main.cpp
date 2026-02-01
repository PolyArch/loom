#include <cstdio>

#include "mat3x3_mult.h"
#include <cmath>

int main() {
    const uint32_t N = 32;

    // Input matrices (N pairs of 3x3 matrices)
    float mat_a[N * 9];
    float mat_b[N * 9];

    for (uint32_t i = 0; i < N * 9; i++) {
        mat_a[i] = 1.0f + i * 0.1f;
        mat_b[i] = 0.5f + i * 0.05f;
    }

    // Output arrays
    float expect_result[N * 9];
    float calculated_result[N * 9];

    // Compute expected result with CPU version
    mat3x3_mult_cpu(mat_a, mat_b, expect_result, N);

    // Compute result with accelerator version
    mat3x3_mult_dsa(mat_a, mat_b, calculated_result, N);

    // Compare results with tolerance
    for (uint32_t i = 0; i < N * 9; i++) {
        if (fabsf(expect_result[i] - calculated_result[i]) > 1e-4f) {
            printf("mat3x3_mult: FAILED\n");
            return 1;
        }
    }

    printf("mat3x3_mult: PASSED\n");
    return 0;
}

