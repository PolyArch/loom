// Loom kernel implementation: window_blackman
#include "window_blackman.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Blackman window function
// Tests complete compilation chain with trigonometric operations (cosine)
// Test: Apply Blackman window to signal [1.5, 2.3, 1.8, 2.7, 2.1, 1.6, 2.4, 1.9]
// Window coefficients: [~0, 0.090, 0.459, 0.920, 0.920, 0.459, 0.090, ~0]
// Expected output (input * window): [~0, 0.208, 0.827, 2.485, 1.933, 0.735, 0.217, ~0]







const float PI_BLACKMAN = 3.14159265358979323846f;

// CPU implementation of Blackman window
// w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
void window_blackman_cpu(const float* __restrict__ input,
                         float* __restrict__ output,
                         const uint32_t N) {
    for (uint32_t n = 0; n < N; n++) {
        float t = 2.0f * PI_BLACKMAN * n / (N - 1);
        float window = 0.42f - 0.5f * cosf(t) + 0.08f * cosf(2.0f * t);
        output[n] = input[n] * window;
    }
}

// Accelerator implementation of Blackman window
LOOM_ACCEL()
void window_blackman_dsa(const float* __restrict__ input,
                         float* __restrict__ output,
                         const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t n = 0; n < N; n++) {
        float t = 2.0f * PI_BLACKMAN * n / (N - 1);
        float window = 0.42f - 0.5f * cosf(t) + 0.08f * cosf(2.0f * t);
        output[n] = input[n] * window;
    }
}



