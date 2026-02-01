// Loom kernel implementation: binary_search
#include "binary_search.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Binary search
// Tests complete compilation chain with while loop and complex control flow
// Test: sorted=[1,3,5,7,9,11,13,15], targets=[5,13,2,15,20], N=8, M=5 → indices=[2,6,0xFFFFFFFF,7,0xFFFFFFFF]






// CPU implementation of binary search
void binary_search_cpu(const float* __restrict__ input_sorted,
                       const float* __restrict__ input_targets,
                       uint32_t* __restrict__ output_indices,
                       const uint32_t N,
                       const uint32_t M) {
    // For each target value, perform binary search
    for (uint32_t t = 0; t < M; t++) {
        float target = input_targets[t];
        int32_t left = 0;
        int32_t right = static_cast<int32_t>(N) - 1;
        int32_t result = -1;
        
        while (left <= right) {
            int32_t mid = left + (right - left) / 2;
            
            if (input_sorted[mid] == target) {
                result = mid;
                break;
            } else if (input_sorted[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        // Store result as uint32_t (0xFFFFFFFF if not found)
        output_indices[t] = (result == -1) ? 0xFFFFFFFF : static_cast<uint32_t>(result);
    }
}

// Accelerator implementation of binary search
LOOM_ACCEL()
void binary_search_dsa(const float* __restrict__ input_sorted,
                       const float* __restrict__ input_targets,
                       uint32_t* __restrict__ output_indices,
                       const uint32_t N,
                       const uint32_t M) {
    // For each target value, perform binary search
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t t = 0; t < M; t++) {
        float target = input_targets[t];
        int32_t left = 0;
        int32_t right = static_cast<int32_t>(N) - 1;
        int32_t result = -1;
        
        while (left <= right) {
            int32_t mid = left + (right - left) / 2;
            
            if (input_sorted[mid] == target) {
                result = mid;
                break;
            } else if (input_sorted[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        // Store result as uint32_t (0xFFFFFFFF if not found)
        output_indices[t] = (result == -1) ? 0xFFFFFFFF : static_cast<uint32_t>(result);
    }
}





