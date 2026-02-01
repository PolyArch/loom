// Loom kernel implementation: trsv_upper
#include "trsv_upper.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>







// CPU implementation of triangular solve (backward substitution)
// Solves U * x = b, where U is upper triangular
// U: N x N upper triangular matrix (row-major, diagonal included)
// b: input vector of length N
// x: output vector of length N
void trsv_upper_cpu(const uint32_t* __restrict__ U, 
                    const uint32_t* __restrict__ b, 
                    uint32_t* __restrict__ x, 
                    const uint32_t N) {
    for (int32_t i = N - 1; i >= 0; i--) {
        uint32_t sum = b[i];
        for (uint32_t j = i + 1; j < N; j++) {
            sum -= U[i * N + j] * x[j];
        }
        x[i] = sum / U[i * N + i];
    }
}

// Accelerator implementation of triangular solve (backward substitution)
LOOM_ACCEL()
void trsv_upper_dsa(const uint32_t* __restrict__ U, 
                    const uint32_t* __restrict__ b, 
                    uint32_t* __restrict__ x, 
                    const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (int32_t i = N - 1; i >= 0; i--) {
        uint32_t sum = b[i];
        for (uint32_t j = i + 1; j < N; j++) {
            sum -= U[i * N + j] * x[j];
        }
        x[i] = sum / U[i * N + i];
    }
}



