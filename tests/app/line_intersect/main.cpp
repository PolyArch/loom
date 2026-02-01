#include <cstdio>

#include "line_intersect.h"

int main() {
    const uint32_t N = 64;

    // Input line segments (N pairs of 2D line segments)
    float line_a[N * 4];
    float line_b[N * 4];

    // Test case 1: Lines that intersect at center
    line_a[0] = 0.0f;  line_a[1] = 0.0f;  line_a[2] = 2.0f;  line_a[3] = 2.0f;
    line_b[0] = 0.0f;  line_b[1] = 2.0f;  line_b[2] = 2.0f;  line_b[3] = 0.0f;

    // Test case 2: Lines that don't intersect
    line_a[4] = 0.0f;  line_a[5] = 0.0f;  line_a[6] = 1.0f;  line_a[7] = 0.0f;
    line_b[4] = 0.0f;  line_b[5] = 1.0f;  line_b[6] = 1.0f;  line_b[7] = 1.0f;

    // Test case 3: Parallel lines
    line_a[8] = 0.0f;  line_a[9] = 0.0f;  line_a[10] = 1.0f;  line_a[11] = 0.0f;
    line_b[8] = 0.0f;  line_b[9] = 1.0f;  line_b[10] = 1.0f;  line_b[11] = 1.0f;

    // Fill remaining with varied patterns
    for (uint32_t i = 3; i < N; i++) {
        float offset = i * 0.1f;
        line_a[i * 4 + 0] = 0.0f + offset;
        line_a[i * 4 + 1] = 0.0f;
        line_a[i * 4 + 2] = 2.0f + offset;
        line_a[i * 4 + 3] = 2.0f;

        line_b[i * 4 + 0] = 0.0f + offset;
        line_b[i * 4 + 1] = 2.0f;
        line_b[i * 4 + 2] = 2.0f + offset;
        line_b[i * 4 + 3] = 0.0f;
    }

    // Output arrays
    uint32_t expect_intersect[N];
    uint32_t calculated_intersect[N];

    // Compute expected result with CPU version
    line_intersect_cpu(line_a, line_b, expect_intersect, N);

    // Compute result with accelerator version
    line_intersect_dsa(line_a, line_b, calculated_intersect, N);

    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_intersect[i] != calculated_intersect[i]) {
            printf("line_intersect: FAILED\n");
            return 1;
        }
    }

    printf("line_intersect: PASSED\n");
    return 0;
}

