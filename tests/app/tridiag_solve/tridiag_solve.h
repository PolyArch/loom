// Loom kernel: tridiag_solve
#ifndef TRIDIAG_SOLVE_H
#define TRIDIAG_SOLVE_H

#include <cstdint>

void tridiag_solve_cpu(const float* __restrict__ input_a,
                       const float* __restrict__ input_b,
                       const float* __restrict__ input_c,
                       const float* __restrict__ input_d,
                       float* __restrict__ output_x,
                       float* __restrict__ c_prime,
                       float* __restrict__ d_prime,
                       const uint32_t N);

void tridiag_solve_dsa(const float* __restrict__ input_a,
                       const float* __restrict__ input_b,
                       const float* __restrict__ input_c,
                       const float* __restrict__ input_d,
                       float* __restrict__ output_x,
                       float* __restrict__ c_prime,
                       float* __restrict__ d_prime,
                       const uint32_t N);

#endif // TRIDIAG_SOLVE_H
