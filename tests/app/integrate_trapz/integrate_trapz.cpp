// Loom kernel implementation: integrate_trapz
#include "integrate_trapz.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>







// CPU implementation of trapezoidal integration
// Computes integral using trapezoidal rule: integral = sum((y[i] + y[i+1]) * (x[i+1] - x[i]) / 2)
// input_x: x coordinates (N elements)
// input_y: y coordinates (N elements)
// Returns: computed integral value
float integrate_trapz_cpu(const float* __restrict__ input_x,
                          const float* __restrict__ input_y,
                          const uint32_t N) {
    float sum = 0.0f;
    
    for (uint32_t i = 0; i < N - 1; i++) {
        float dx = input_x[i + 1] - input_x[i];
        float avg_y = (input_y[i] + input_y[i + 1]) * 0.5f;
        sum += avg_y * dx;
    }
    
    return sum;
}

// Accelerator implementation of trapezoidal integration
LOOM_ACCEL()
float integrate_trapz_dsa(const float* __restrict__ input_x,
                          const float* __restrict__ input_y,
                          const uint32_t N) {
    float sum = 0.0f;
    LOOM_PARALLEL(4)
    for (uint32_t i = 0; i < N - 1; i++) {
        float dx = input_x[i + 1] - input_x[i];
        float avg_y = (input_y[i] + input_y[i + 1]) * 0.5f;
        sum += avg_y * dx;
    }
    
    return sum;
}


