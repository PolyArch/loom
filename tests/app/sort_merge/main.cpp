#include <cstdio>
#include <cmath>

#include "sort_merge.h"

int main() {
    // Test: [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0] -> [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]
    const uint32_t N = 8;

    float input[8] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};

    // Output arrays
    float expect_output[8];
    float expect_temp[8];
    float calculated_output[8];
    float calculated_temp[8];

    // Compute expected result with CPU version
    sort_merge_cpu(input, expect_output, expect_temp, N);

    // Compute result with DSA version
    sort_merge_dsa(input, calculated_output, calculated_temp, N);

    // Compare results with tolerance for floating point
    const float epsilon = 1e-6f;
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > epsilon) {
            printf("sort_merge: FAILED (mismatch at index %u: expected %f, got %f)\n",
                   i, expect_output[i], calculated_output[i]);
            return 1;
        }
    }

    printf("sort_merge: PASSED\n");
    return 0;
}
