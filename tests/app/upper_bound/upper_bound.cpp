// Loom kernel implementation: upper_bound
#include "upper_bound.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>







// CPU implementation of upper_bound (first element > value)
void upper_bound_cpu(const float* __restrict__ input_sorted,
                     const float* __restrict__ input_targets,
                     uint32_t* __restrict__ output_indices,
                     const uint32_t N,
                     const uint32_t M) {
    // For each target value, find upper bound
    for (uint32_t t = 0; t < M; t++) {
        float target = input_targets[t];
        uint32_t left = 0;
        uint32_t right = N;
        
        while (left < right) {
            uint32_t mid = left + (right - left) / 2;
            
            if (input_sorted[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        output_indices[t] = left;
    }
}

// Accelerator implementation of upper_bound (first element > value)
LOOM_ACCEL()
void upper_bound_dsa(const float* __restrict__ input_sorted,
                     const float* __restrict__ input_targets,
                     uint32_t* __restrict__ output_indices,
                     const uint32_t N,
                     const uint32_t M) {
    // For each target value, find upper bound
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t t = 0; t < M; t++) {
        float target = input_targets[t];
        uint32_t left = 0;
        uint32_t right = N;
        
        while (left < right) {
            uint32_t mid = left + (right - left) / 2;
            
            if (input_sorted[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        output_indices[t] = left;
    }
}



