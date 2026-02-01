// Loom kernel implementation: sigmoid
#include "sigmoid.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Sigmoid activation function
// Tests complete compilation chain with transcendental operations
// Test: [0.0, 1.0, -1.0] → [0.5, 0.731, 0.269]







// CPU implementation of Sigmoid activation
// sigmoid(x) = 1 / (1 + exp(-x))
void sigmoid_cpu(const float* __restrict__ input,
                 float* __restrict__ output,
                 const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

// Sigmoid: output[i] = 1 / (1 + exp(-input[i]))
// Accelerator implementation of Sigmoid activation
LOOM_ACCEL()
void sigmoid_dsa(const float* __restrict__ input,
                 float* __restrict__ output,
                 const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t i = 0; i < N; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}



// Step 1: Read input

// Step 2: Negate input: -x

// Step 3: Exponential: exp(-x)

// Step 4: Add 1: 1 + exp(-x)

// Step 5: Reciprocal: 1 / (1 + exp(-x))

// Step 6: Write output




