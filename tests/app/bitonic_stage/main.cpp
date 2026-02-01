#include <cstdio>

#include "bitonic_stage.h"
#include <cmath>

int main() {
    const uint32_t N = 8;
    const uint32_t stage = 1;
    const uint32_t pass = 0;

    // Initial input array - same as hpp test case
    float initial_input[N] = {3.0f, 1.0f, 4.0f, 2.0f, 8.0f, 6.0f, 7.0f, 5.0f};

    // Create separate arrays for CPU and DSA versions (both in-place)
    float cpu_array[N];
    float dsa_array[N];

    // Copy initial input to both arrays
    for (uint32_t i = 0; i < N; i++) {
        cpu_array[i] = initial_input[i];
        dsa_array[i] = initial_input[i];
    }

    // Compute result with CPU version (in-place)
    bitonic_stage_cpu(cpu_array, N, stage, pass);

    // Compute result with DSA version (in-place)
    bitonic_stage_dsa(dsa_array, N, stage, pass);

    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(cpu_array[i] - dsa_array[i]) > 1e-5f) {
            printf("bitonic_stage: FAILED\n");
            return 1;
        }
    }

    printf("bitonic_stage: PASSED\n");
    return 0;
}

