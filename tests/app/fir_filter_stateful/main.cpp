// Loom app test driver: fir_filter_stateful
#include "fir_filter_stateful.h"
#include <cstdio>
#include <cmath>

int main() {
    const uint32_t input_size = 16;
    const uint32_t num_taps = 5;

    // Input signal
    float input[input_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

    // Filter coefficients (simple averaging filter)
    float coeffs[num_taps] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

    // State arrays
    float input_state[num_taps - 1] = {0.0f, 0.0f, 0.0f, 0.0f};
    float expect_output_state[num_taps - 1] = {0.0f};
    float calculated_output_state[num_taps - 1] = {0.0f};
    float cpu_current_state[num_taps - 1] = {0.0f};
    float dsa_current_state[num_taps - 1] = {0.0f};

    // Output arrays
    float expect_output[input_size] = {0.0f};
    float calculated_output[input_size] = {0.0f};

    // Run CPU version
    fir_filter_stateful_cpu(input, coeffs, input_state,
                            expect_output, expect_output_state, cpu_current_state,
                            input_size, num_taps);

    // Run DSA version
    fir_filter_stateful_dsa(input, coeffs, input_state,
                            calculated_output, calculated_output_state, dsa_current_state,
                            input_size, num_taps);

    // Compare results
    bool passed = true;
    for (uint32_t i = 0; i < input_size; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-5f) {
            passed = false;
            break;
        }
    }

    if (passed) {
        printf("fir_filter_stateful: PASSED\n");
        return 0;
    } else {
        printf("fir_filter_stateful: FAILED\n");
        return 1;
    }
}
