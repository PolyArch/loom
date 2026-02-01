#include <cstdio>

#include "compare_swap.h"
#include <cmath>

int main() {
    const uint32_t N = 16;

    // Input arrays
    float input_a[N] = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 3.0f, 7.0f, 4.0f,
                        6.0f, 10.0f, 15.0f, 12.0f, 11.0f, 14.0f, 13.0f, 16.0f};
    float input_b[N] = {3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 8.0f, 4.0f, 6.0f,
                        10.0f, 5.0f, 12.0f, 15.0f, 14.0f, 11.0f, 16.0f, 13.0f};

    // Output arrays for CPU
    float expect_min[N];
    float expect_max[N];

    // Output arrays for DSA
    float calculated_min[N];
    float calculated_max[N];

    // Compute expected result with CPU version
    compare_swap_cpu(input_a, input_b, expect_min, expect_max, N);

    // Compute result with DSA version
    compare_swap_dsa(input_a, input_b, calculated_min, calculated_max, N);

    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_min[i] - calculated_min[i]) > 1e-5f) {
            printf("compare_swap: FAILED\n");
            return 1;
        }
        if (fabsf(expect_max[i] - calculated_max[i]) > 1e-5f) {
            printf("compare_swap: FAILED\n");
            return 1;
        }
    }

    printf("compare_swap: PASSED\n");
    return 0;
}

