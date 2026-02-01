// Main test file for AXPY kernel
// AXPY: y = alpha * x + y (out-of-place: output_y = alpha * input_x + input_y)

#include "axpy.h"
#include <cstdio>

int main() {
    const uint32_t N = 8;
    const uint32_t alpha = 3;

    // Input vectors
    uint32_t x[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t y[N] = {10, 20, 30, 40, 50, 60, 70, 80};

    // Output vectors
    uint32_t cpu_out[N];
    uint32_t dsa_out[N];

    // Compute with CPU version
    axpy_cpu(x, y, cpu_out, alpha, N);

    // Compute with DSA version
    axpy_dsa(x, y, dsa_out, alpha, N);

    // Print results
    printf("AXPY Results (y = alpha*x + y, alpha=%u):\n", alpha);
    printf("x   = [");
    for (uint32_t i = 0; i < N; i++) printf("%u%s", x[i], i < N-1 ? ", " : "");
    printf("]\n");
    printf("y   = [");
    for (uint32_t i = 0; i < N; i++) printf("%u%s", y[i], i < N-1 ? ", " : "");
    printf("]\n");
    printf("out = [");
    for (uint32_t i = 0; i < N; i++) printf("%u%s", dsa_out[i], i < N-1 ? ", " : "");
    printf("]\n");

    // Compare results
    bool passed = true;
    for (uint32_t i = 0; i < N; i++) {
        if (cpu_out[i] != dsa_out[i]) {
            printf("FAILED at index %u: cpu=%u, dsa=%u\n", i, cpu_out[i], dsa_out[i]);
            passed = false;
        }
    }

    if (passed) {
        printf("PASSED: All results correct!\n");
    }

    return passed ? 0 : 1;
}
