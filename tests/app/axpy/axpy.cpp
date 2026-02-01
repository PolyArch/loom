// AXPY kernel implementations
// AXPY: output_y[i] = alpha * input_x[i] + input_y[i]

#include "axpy.h"
#include "loom/loom.h"

// CPU reference implementation
void axpy_cpu(const uint32_t* __restrict input_x,
              const uint32_t* __restrict input_y,
              uint32_t* __restrict output_y,
              uint32_t alpha, uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        output_y[i] = alpha * input_x[i] + input_y[i];
    }
}

// DSA accelerated implementation with Loom pragmas
LOOM_ACCEL("axpy")
void axpy_dsa(const uint32_t* __restrict input_x,
              const uint32_t* __restrict input_y,
              uint32_t* __restrict output_y,
              uint32_t alpha, uint32_t N) {
    compute_loop:
    LOOM_PARALLEL(4, contiguous)
    LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
    for (uint32_t i = 0; i < N; i++) {
        output_y[i] = alpha * input_x[i] + input_y[i];
    }
}
