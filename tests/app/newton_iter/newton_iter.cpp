// Loom kernel implementation: newton_iter
#include "newton_iter.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Newton-Raphson iteration step
// Tests complete compilation chain with floating-point division and array operations
// Test: 15 random values, x_new = x - f(x)/f'(x)

// CPU implementation of Newton-Raphson iteration step
// Computes one Newton-Raphson iteration: x_new = x_old - f(x)/f'(x)
// input_x: current x values (N elements)
// input_f: function values f(x) at current x (N elements)
// input_df: derivative values f'(x) at current x (N elements)
// output_x: updated x values (N elements)
void newton_iter_cpu(const float* __restrict__ input_x,
                     const float* __restrict__ input_f,
                     const float* __restrict__ input_df,
                     float* __restrict__ output_x,
                     const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        output_x[i] = input_x[i] - input_f[i] / input_df[i];
    }
}

// Accelerator implementation of Newton-Raphson iteration step
LOOM_ACCEL()
void newton_iter_dsa(const float* __restrict__ input_x,
                     const float* __restrict__ input_f,
                     const float* __restrict__ input_df,
                     float* __restrict__ output_x,
                     const uint32_t N) {
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < N; i++) {
        output_x[i] = input_x[i] - input_f[i] / input_df[i];
    }
}

