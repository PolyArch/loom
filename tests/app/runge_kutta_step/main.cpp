#include <cstdio>

#include "runge_kutta_step.h"
#include <cmath>

int main() {
    const uint32_t N = 64;
    const float h = 0.1f;

    // Input current state and k values
    float input_y[N];
    float input_k1[N];
    float input_k2[N];
    float input_k3[N];
    float input_k4[N];

    // Output updated state
    float expect_y[N];
    float calculated_y[N];

    // Initialize inputs
    for (uint32_t i = 0; i < N; i++) {
        input_y[i] = (float)i;
        input_k1[i] = 1.0f + 0.1f * (float)i;
        input_k2[i] = 1.1f + 0.1f * (float)i;
        input_k3[i] = 1.2f + 0.1f * (float)i;
        input_k4[i] = 1.3f + 0.1f * (float)i;
    }

    // Compute expected result with CPU version
    runge_kutta_step_cpu(input_y, input_k1, input_k2, input_k3, input_k4, expect_y, h, N);

    // Compute result with accelerator version
    runge_kutta_step_dsa(input_y, input_k1, input_k2, input_k3, input_k4, calculated_y, h, N);

    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_y[i] - calculated_y[i]) > 1e-5f) {
            printf("runge_kutta_step: FAILED\n");
            return 1;
        }
    }

    printf("runge_kutta_step: PASSED\n");
    return 0;
}

