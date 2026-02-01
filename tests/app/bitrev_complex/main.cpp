#include <cstdio>

#include "bitrev_complex.h"
#include <cmath>

int main() {
    const uint32_t N = 128;

    // Complex input arrays
    float input_real[N];
    float input_imag[N];

    // Complex output arrays
    float expect_real[N];
    float expect_imag[N];
    float calculated_real[N];
    float calculated_imag[N];

    // Initialize input
    for (uint32_t i = 0; i < N; i++) {
        input_real[i] = static_cast<float>(i);
        input_imag[i] = static_cast<float>(N - i);
    }

    // Test complex bit-reversal
    bitrev_complex_cpu(input_real, input_imag, expect_real, expect_imag, N);
    bitrev_complex_dsa(input_real, input_imag, calculated_real, calculated_imag, N);

    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_real[i] - calculated_real[i]) > 1e-6f ||
            fabsf(expect_imag[i] - calculated_imag[i]) > 1e-6f) {
            printf("bitrev_complex: FAILED\n");
            return 1;
        }
    }

    printf("bitrev_complex: PASSED\n");
    return 0;
}

