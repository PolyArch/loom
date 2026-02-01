#include <cstdio>

#include "ifft_butterfly.h"
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
    const uint32_t N = 16;
    const uint32_t log2n = 4;  // log2(64)

    // Input and working arrays
    float input_real[N];
    float input_imag[N];
    float expect_real[N];
    float expect_imag[N];
    float calculated_real[N];
    float calculated_imag[N];

    // Initialize input data with specific test values
    float input_real_data[16] = {
        -3.981271f,11.75819f,1.067285f,-24.85930f,
        19.57854f,2.130239f,4.313792f,-6.751454f,
        1.824865f,-3.408288f,0.6815608f,4.723578f,
        8.814138f,-27.89848,-6.278558f,-1.788430f };

    float input_imag_data[16] = {
        -16.65886f,5.712252f,4.835214f,-10.16193f,
        -7.473514f,-2.492356f,-16.83604f,4.521151f,
        -4.100621f,6.368324f,9.526210f,12.11642f,
        10.31990f,-23.74220f,6.707603f,-9.962789f};

    for (uint32_t i = 0; i < N; i++) {
        input_real[i] = input_real_data[i];
        input_imag[i] = input_imag_data[i];
    }

    // Apply IFFT butterfly operations
    ifft_butterfly_cpu(input_real, input_imag, expect_real, expect_imag, N);
    ifft_butterfly_dsa(input_real, input_imag, calculated_real, calculated_imag, N);

    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_real[i] - calculated_real[i]) > 1e-4f ||
            fabsf(expect_imag[i] - calculated_imag[i]) > 1e-4f) {
            printf("ifft_butterfly: FAILED\n");
            return 1;
        }
    }

    printf("ifft_butterfly: PASSED\n");
    return 0;
}

