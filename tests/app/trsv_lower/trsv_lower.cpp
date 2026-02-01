// Loom kernel implementation: trsv_lower
#include "trsv_lower.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// CPU implementation of triangular solve (forward substitution)
// Solves L * x = b, where L is lower triangular
// L: N x N lower triangular matrix (row-major, diagonal included)
// b: input vector of length N
// x: output vector of length N
void trsv_lower_cpu(const uint32_t* __restrict__ L, 
                    const uint32_t* __restrict__ b, 
                    uint32_t* __restrict__ x, 
                    const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t sum = b[i];
        for (uint32_t j = 0; j < i; j++) {
            sum -= L[i * N + j] * x[j];
        }
        // Assume diagonal elements are non-zero (no division by zero check)
        x[i] = sum / L[i * N + i];
    }
}

// Accelerator implementation of triangular solve (forward substitution)
LOOM_ACCEL()
void trsv_lower_dsa(const uint32_t* __restrict__ L, 
                    const uint32_t* __restrict__ b, 
                    uint32_t* __restrict__ x, 
                    const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t sum = b[i];
        for (uint32_t j = 0; j < i; j++) {
            sum -= L[i * N + j] * x[j];
        }
        x[i] = sum / L[i * N + i];
    }
}

