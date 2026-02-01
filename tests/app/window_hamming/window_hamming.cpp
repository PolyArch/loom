// Loom kernel implementation: window_hamming
#include "window_hamming.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>








const float PI_HAMMING = 3.14159265358979323846f;

// CPU implementation of Hamming window
// w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
void window_hamming_cpu(const float* __restrict__ input,
                        float* __restrict__ output,
                        const uint32_t N) {
    for (uint32_t n = 0; n < N; n++) {
        float window = 0.54f - 0.46f * cosf(2.0f * PI_HAMMING * n / (N - 1));
        output[n] = input[n] * window;
    }
}

// Accelerator implementation of Hamming window
LOOM_ACCEL()
void window_hamming_dsa(const float* __restrict__ input,
                        float* __restrict__ output,
                        const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t n = 0; n < N; n++) {
        float window = 0.54f - 0.46f * cosf(2.0f * PI_HAMMING * n / (N - 1));
        output[n] = input[n] * window;
    }
}



