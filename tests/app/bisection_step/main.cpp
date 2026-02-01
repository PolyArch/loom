#include <cstdio>

#include "bisection_step.h"
#include <cmath>

int main() {
    const uint32_t N = 64;

    // Input interval endpoints and function values
    float input_a[N];
    float input_b[N];
    float input_fa[N];
    float input_fb[N];
    float input_fc[N];

    // Output updated interval endpoints
    float expect_a[N];
    float expect_b[N];
    float calculated_a[N];
    float calculated_b[N];

    // Initialize inputs to test all three branches
    // Branch 1 (fa*fc < 0): root in [a, c]
    // Branch 2 (fc*fb < 0): root in [c, b]
    // Branch 3 (default): no clear sign change
    for (uint32_t i = 0; i < N; i++) {
        if (i % 3 == 0) {
            // Test case 1: fa*fc < 0, root in [a, c]
            // Use f(x) = x - 1
            input_a[i] = 0.0f;
            input_b[i] = 2.0f;
            input_fa[i] = -1.0f;  // f(0) = -1
            input_fb[i] = 1.0f;   // f(2) = 1
            input_fc[i] = 0.0f;   // f(1) = 0
        } else if (i % 3 == 1) {
            // Test case 2: fa*fc > 0, fc*fb < 0, root in [c, b]
            // Use f(x) = x - 3
            input_a[i] = 1.0f;
            input_b[i] = 5.0f;
            input_fa[i] = -2.0f;  // f(1) = -2
            input_fb[i] = 2.0f;   // f(5) = 2
            input_fc[i] = 0.0f;   // f(3) = 0
        } else {
            // Test case 3: fa*fc > 0, fc*fb > 0, default to [c, b]
            input_a[i] = 2.0f;
            input_b[i] = 6.0f;
            input_fa[i] = 4.0f;
            input_fb[i] = 8.0f;
            input_fc[i] = 5.0f;
        }
    }

    // Compute expected result with CPU version
    bisection_step_cpu(input_a, input_b, input_fa, input_fb, input_fc, expect_a, expect_b, N);

    // Compute result with accelerator version
    bisection_step_dsa(input_a, input_b, input_fa, input_fb, input_fc, calculated_a, calculated_b, N);

    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_a[i] - calculated_a[i]) > 1e-5f) {
            printf("bisection_step: FAILED\n");
            return 1;
        }
        if (fabsf(expect_b[i] - calculated_b[i]) > 1e-5f) {
            printf("bisection_step: FAILED\n");
            return 1;
        }
    }

    printf("bisection_step: PASSED\n");
    return 0;
}

