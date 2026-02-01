// Loom kernel implementation: tridiag_solve
#include "tridiag_solve.h"
#include "loom/loom.h"

void tridiag_solve_cpu(const float* __restrict__ input_a,
                       const float* __restrict__ input_b,
                       const float* __restrict__ input_c,
                       const float* __restrict__ input_d,
                       float* __restrict__ output_x,
                       float* __restrict__ c_prime,
                       float* __restrict__ d_prime,
                       const uint32_t N) {
    // Forward elimination
    c_prime[0] = input_c[0] / input_b[0];
    d_prime[0] = input_d[0] / input_b[0];

    for (uint32_t i = 1; i < N; i++) {
        float m = input_b[i] - input_a[i] * c_prime[i - 1];
        c_prime[i] = input_c[i] / m;
        d_prime[i] = (input_d[i] - input_a[i] * d_prime[i - 1]) / m;
    }

    // Back substitution
    output_x[N - 1] = d_prime[N - 1];
    for (uint32_t i = N - 1; i > 0; i--) {
        output_x[i - 1] = d_prime[i - 1] - c_prime[i - 1] * output_x[i];
    }
}

LOOM_ACCEL()
void tridiag_solve_dsa(const float* __restrict__ input_a,
                       const float* __restrict__ input_b,
                       const float* __restrict__ input_c,
                       const float* __restrict__ input_d,
                       float* __restrict__ output_x,
                       float* __restrict__ c_prime,
                       float* __restrict__ d_prime,
                       const uint32_t N) {
    // Forward elimination
    c_prime[0] = input_c[0] / input_b[0];
    d_prime[0] = input_d[0] / input_b[0];

    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 1; i < N; i++) {
        float m = input_b[i] - input_a[i] * c_prime[i - 1];
        c_prime[i] = input_c[i] / m;
        d_prime[i] = (input_d[i] - input_a[i] * d_prime[i - 1]) / m;
    }

    // Back substitution
    output_x[N - 1] = d_prime[N - 1];
    for (uint32_t i = N - 1; i > 0; i--) {
        output_x[i - 1] = d_prime[i - 1] - c_prime[i - 1] * output_x[i];
    }
}
