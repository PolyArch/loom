// Loom kernel implementation: relu
#include "relu.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: ReLU activation function
// Tests complete compilation chain with max operation using float type
// Test: [-5.5,-3.2,-1,-0.1,0,0.1,1,2.5,4.8,10] → [0,0,0,0,0,0.1,1,2.5,4.8,10] (max(0, x))







// CPU implementation of ReLU activation
// ReLU(x) = max(0, x)
void relu_cpu(const float* __restrict__ input,
              float* __restrict__ output,
              const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

// ReLU: output[i] = max(0, input[i])
// Accelerator implementation of ReLU activation
LOOM_ACCEL()
void relu_dsa(const float* __restrict__ input,
              float* __restrict__ output,
              const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t i = 0; i < N; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}





