#include <cstdio>

#include "window_hamming.h"
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

    // Test Hamming window application
    window_hamming_cpu(input, expect_output, N);
    window_hamming_dsa(input, calculated_output, N);

    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-6f) {
            printf("window_hamming: FAILED\n");
            return 1;
        }
    }

    printf("window_hamming: PASSED\n");
    return 0;
}

