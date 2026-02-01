#include <cstdio>

#include "fft_butterfly.h"
#include <cmath>

// Helper function to manually bit-reverse an index
static uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log2n; i++) {
        if (x & (1 << i)) {
            result |= 1 << (log2n - 1 - i);
        }
    }
    return result;
}

int main() {
    const uint32_t N = 16;  // FFT size (must be power of 2)
    const uint32_t log2n = 4;  // log2(16)

    // Input data (already bit-reversed order for the butterfly algorithm)
    float input_real[N];
    float input_imag[N];
    float expect_real[N];
    float expect_imag[N];
    float calculated_real[N];
    float calculated_imag[N];

    // Initialize input data with specific test values
    float input_real_data[16] = {-1.254599f, 4.507143f, 2.319939f, 0.986585f,
                                  -3.439814f, -3.440055f, -4.419164f, 3.661762f,
                                  1.011150f, 2.080726f, -4.794155f, 4.699099f,
                                  3.324426f, -2.876609f, -3.181750f, -3.165955f};

    float input_imag_data[16] = {-1.957578f, 0.247564f, -0.680550f, -2.087709f,
                                  1.118529f, -3.605061f, -2.078553f, -1.336382f,
                                  -0.439300f, 2.851760f, -3.003262f, 0.142344f,
                                  0.924146f, -4.535496f, 1.075449f, -3.294759f};

    for (uint32_t i = 0; i < N; i++) {
        input_real[i] = input_real_data[i];
        input_imag[i] = input_imag_data[i];
    }

    // Apply butterfly operations
    fft_butterfly_cpu(input_real, input_imag, expect_real, expect_imag, N);
    fft_butterfly_dsa(input_real, input_imag, calculated_real, calculated_imag, N);

    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_real[i] - calculated_real[i]) > 1e-4f ||
            fabsf(expect_imag[i] - calculated_imag[i]) > 1e-4f) {
            printf("fft_butterfly: FAILED\n");
            return 1;
        }
    }

    printf("fft_butterfly: PASSED\n");
    return 0;
}

