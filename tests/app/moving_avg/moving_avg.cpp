// Loom kernel implementation: moving_avg
#include "moving_avg.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Moving average with sliding window
// Tests complete compilation chain with variable-length inner loop, backward indexing, and per-window reduction
// Test: 12 random input values, window=4 → moving averages






// CPU implementation of moving average with window
// For each position i, output[i] = average of input[i-window_size+1..i]
// Positions before window_size use all available elements
void moving_avg_cpu(const float* __restrict__ input,
                    float* __restrict__ output,
                    const uint32_t N,
                    const uint32_t window_size) {
    for (uint32_t i = 0; i < N; i++) {
        // Determine window start and actual window size
        uint32_t start = (i + 1 >= window_size) ? (i + 1 - window_size) : 0;
        uint32_t actual_window = i - start + 1;
        
        // Compute sum over window
        float sum = 0.0f;
        for (uint32_t j = start; j <= i; j++) {
            sum += input[j];
        }
        
        // Compute average
        output[i] = sum / static_cast<float>(actual_window);
    }
}

// Moving average: output[i] = avg(input[start..i]) where start = max(0, i+1-window_size)
// Accelerator implementation of moving average with window
LOOM_ACCEL()
void moving_avg_dsa(const float* __restrict__ input,
                    float* __restrict__ output,
                    const uint32_t N,
                    const uint32_t window_size) {
    LOOM_PARALLEL(4)
    for (uint32_t i = 0; i < N; i++) {
        // Determine window start and actual window size
        uint32_t start = (i + 1 >= window_size) ? (i + 1 - window_size) : 0;
        uint32_t actual_window = i - start + 1;
        
        // Compute sum over window
        float sum = 0.0f;
        for (uint32_t j = start; j <= i; j++) {
            sum += input[j];
        }
        
        // Compute average
        output[i] = sum / static_cast<float>(actual_window);
    }
}




