#include <cstdio>

#include "dot_product_3d.h"
#include <cmath>

int main() {
    const uint32_t N = 64;

    // Input vectors (N pairs of 3D vectors)
    float vec_a[N * 3];
    float vec_b[N * 3];

    for (uint32_t i = 0; i < N; i++) {
        vec_a[i * 3 + 0] = 1.0f + i * 0.1f;
        vec_a[i * 3 + 1] = 2.0f + i * 0.2f;
        vec_a[i * 3 + 2] = 3.0f + i * 0.3f;

        vec_b[i * 3 + 0] = 0.5f + i * 0.05f;
        vec_b[i * 3 + 1] = 1.5f + i * 0.15f;
        vec_b[i * 3 + 2] = 2.5f + i * 0.25f;
    }

    // Output arrays
    float expect_result[N];
    float calculated_result[N];

    // Compute expected result with CPU version
    dot_product_3d_cpu(vec_a, vec_b, expect_result, N);

    // Compute result with accelerator version
    dot_product_3d_dsa(vec_a, vec_b, calculated_result, N);

    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_result[i] - calculated_result[i]) > 1e-5f) {
            printf("dot_product_3d: FAILED\n");
            return 1;
        }
    }

    printf("dot_product_3d: PASSED\n");
    return 0;
}

