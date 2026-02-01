// Loom kernel implementation: window_hanning
#include "window_hanning.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

const float PI_HANNING = 3.14159265358979323846f;

// CPU implementation of Hanning window
// w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
void window_hanning_cpu(const float* __restrict__ input,
                        float* __restrict__ output,
                        const uint32_t N) {
    for (uint32_t n = 0; n < N; n++) {
        float window = 0.5f * (1.0f - cosf(2.0f * PI_HANNING * n / (N - 1)));
        output[n] = input[n] * window;
    }
}

// Accelerator implementation of Hanning window
LOOM_ACCEL()
void window_hanning_dsa(const float* __restrict__ input,
                        float* __restrict__ output,
                        const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t n = 0; n < N; n++) {
        float window = 0.5f * (1.0f - cosf(2.0f * PI_HANNING * n / (N - 1)));
        output[n] = input[n] * window;
    }
}

