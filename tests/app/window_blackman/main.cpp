#include <cstdio>

#include "window_blackman.h"
#include <cmath>

int main() {
    const uint32_t N = 256;

    // Input signal
    float input[N];

    // Windowed output
    float expect_output[N];
    float calculated_output[N];

    // Initialize input signal
    for (uint32_t i = 0; i < N; i++) {
        input[i] = sinf(2.0f * 3.14159f * i / 32.0f);
    }

    // Test Blackman window application
    window_blackman_cpu(input, expect_output, N);
    window_blackman_dsa(input, calculated_output, N);

    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-6f) {
            printf("window_blackman: FAILED\n");
            return 1;
        }
    }

    printf("window_blackman: PASSED\n");
    return 0;
}

