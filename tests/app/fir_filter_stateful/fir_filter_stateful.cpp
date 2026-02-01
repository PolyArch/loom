// Loom kernel implementation: fir_filter_stateful
#include "fir_filter_stateful.h"
#include "loom/loom.h"

void fir_filter_stateful_cpu(const float* __restrict__ input,
                              const float* __restrict__ coeffs,
                              const float* __restrict__ input_state,
                              float* __restrict__ output,
                              float* __restrict__ output_state,
                              float* __restrict__ current_state,
                              const uint32_t input_size,
                              const uint32_t num_taps) {
    // Initialize current state from input state
    for (uint32_t k = 0; k < num_taps - 1; k++) {
        current_state[k] = input_state[k];
    }

    for (uint32_t n = 0; n < input_size; n++) {
        float sum = coeffs[0] * input[n];

        // Contribution from current state (previous samples)
        for (uint32_t k = 1; k < num_taps; k++) {
            sum += coeffs[k] * current_state[num_taps - 1 - k];
        }

        output[n] = sum;

        // Update current state: shift and insert new sample
        for (uint32_t k = num_taps - 1; k > 1; k--) {
            current_state[k - 1] = current_state[k - 2];
        }
        current_state[0] = input[n];
    }

    // Copy final state to output state
    for (uint32_t k = 0; k < num_taps - 1; k++) {
        output_state[k] = current_state[k];
    }
}

LOOM_ACCEL()
void fir_filter_stateful_dsa(const float* __restrict__ input,
                              const float* __restrict__ coeffs,
                              const float* __restrict__ input_state,
                              float* __restrict__ output,
                              float* __restrict__ output_state,
                              float* __restrict__ current_state,
                              const uint32_t input_size,
                              const uint32_t num_taps) {
    // Initialize current state from input state
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t k = 0; k < num_taps - 1; k++) {
        current_state[k] = input_state[k];
    }

    for (uint32_t n = 0; n < input_size; n++) {
        float sum = coeffs[0] * input[n];

        // Contribution from current state (previous samples)
        for (uint32_t k = 1; k < num_taps; k++) {
            sum += coeffs[k] * current_state[num_taps - 1 - k];
        }

        output[n] = sum;

        // Update current state: shift and insert new sample
        for (uint32_t k = num_taps - 1; k > 1; k--) {
            current_state[k - 1] = current_state[k - 2];
        }
        current_state[0] = input[n];
    }

    // Copy final state to output state
    for (uint32_t k = 0; k < num_taps - 1; k++) {
        output_state[k] = current_state[k];
    }
}
