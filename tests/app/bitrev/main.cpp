#include <cstdio>

#include "bitrev.h"
#include <cmath>

int main() {
    const uint32_t N = 128;

    // Input and output arrays
    float input[N];
    float expect_output[N];
    float calculated_output[N];

    // Initialize input
    for (uint32_t i = 0; i < N; i++) {
        input[i] = static_cast<float>(i);
    }

    // Test bit-reversal
    bitrev_cpu(input, expect_output, N);
    bitrev_dsa(input, calculated_output, N);

    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-6f) {
            printf("bitrev: FAILED\n");
            return 1;
        }
    }

    printf("bitrev: PASSED\n");
    return 0;
}

