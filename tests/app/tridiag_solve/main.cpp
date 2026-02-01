// Loom app test driver: tridiag_solve
#include "tridiag_solve.h"
#include <cstdio>
#include <cmath>

int main() {
    const uint32_t N = 8;

    // Input diagonals and RHS
    float input_a[N] = {0.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    float input_b[N] = {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};
    float input_c[N] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0.0f};
    float input_d[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    // Output solution vectors
    float expect_x[N] = {0.0f};
    float calculated_x[N] = {0.0f};

    // Temporary buffers
    float cpu_c_prime[N] = {0.0f};
    float cpu_d_prime[N] = {0.0f};
    float dsa_c_prime[N] = {0.0f};
    float dsa_d_prime[N] = {0.0f};

    // Compute expected result with CPU version
    tridiag_solve_cpu(input_a, input_b, input_c, input_d,
                      expect_x, cpu_c_prime, cpu_d_prime, N);

    // Compute result with DSA version
    tridiag_solve_dsa(input_a, input_b, input_c, input_d,
                      calculated_x, dsa_c_prime, dsa_d_prime, N);

    // Compare results with tolerance
    bool passed = true;
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_x[i] - calculated_x[i]) > 1e-5f) {
            passed = false;
            break;
        }
    }

    if (passed) {
        printf("tridiag_solve: PASSED\n");
        return 0;
    } else {
        printf("tridiag_solve: FAILED\n");
        return 1;
    }
}
