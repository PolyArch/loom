#include <cstdio>

#include "newton_iter.h"
#include <cmath>

int main() {
    const uint32_t N = 64;
    
    // Input current x, f(x), and f'(x) values
    float input_x[N];
    float input_f[N];
    float input_df[N];
    
    // Output updated x values
    float expect_x[N];
    float calculated_x[N];
    
    // Initialize inputs (solving x^2 - c = 0, so f(x) = x^2 - c, f'(x) = 2x)
    for (uint32_t i = 0; i < N; i++) {
        float c = (float)(i + 1);
        input_x[i] = c;  // Initial guess
        input_f[i] = input_x[i] * input_x[i] - c;  // f(x) = x^2 - c
        input_df[i] = 2.0f * input_x[i];  // f'(x) = 2x
    }
    
    // Compute expected result with CPU version
    newton_iter_cpu(input_x, input_f, input_df, expect_x, N);
    
    // Compute result with accelerator version
    newton_iter_dsa(input_x, input_f, input_df, calculated_x, N);
    
    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_x[i] - calculated_x[i]) > 1e-5f) {
            printf("newton_iter: FAILED\n");
            return 1;
        }
    }
    
    printf("newton_iter: PASSED\n");
    return 0;
}

