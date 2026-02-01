// Loom kernel implementation: hist_bin
#include "hist_bin.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Histogram binning
// Tests complete compilation chain with scatter writes and binning logic
// Test: input=[0.5,1.5,2.5,3.5,4.5,1.2,2.8,0.1], 5 bins from 0 to 5 → [2,2,2,1,1]






// CPU implementation of histogram binning
// Bins data into num_bins bins between min_val and max_val
void hist_bin_cpu(const float* __restrict__ input,
                  uint32_t* __restrict__ output,
                  const uint32_t N,
                  const uint32_t num_bins,
                  const float min_val,
                  const float max_val) {
    // Initialize output bins to zero
    for (uint32_t i = 0; i < num_bins; i++) {
        output[i] = 0;
    }
    
    float range = max_val - min_val;
    float bin_width = range / static_cast<float>(num_bins);
    
    // Count elements in each bin
    for (uint32_t i = 0; i < N; i++) {
        float val = input[i];
        
        // Skip values outside range
        if (val < min_val || val >= max_val) {
            continue;
        }
        
        // Compute bin index
        LOOM_REDUCE(+)
        uint32_t bin = static_cast<uint32_t>((val - min_val) / bin_width);
        
        // Handle edge case where val == max_val
        if (bin >= num_bins) {
            bin = num_bins - 1;
        }
        
        output[bin]++;
    }
}

// Accelerator implementation of histogram binning
LOOM_TARGET("temporal")
LOOM_ACCEL()
void hist_bin_dsa(const float* __restrict__ input,
                  uint32_t* __restrict__ output,
                  const uint32_t N,
                  const uint32_t num_bins,
                  const float min_val,
                  const float max_val) {
    // Initialize output bins to zero
    LOOM_PARALLEL(4)
    for (uint32_t i = 0; i < num_bins; i++) {
        output[i] = 0;
    }
    
    float range = max_val - min_val;
    float bin_width = range / static_cast<float>(num_bins);
    
    // Count elements in each bin
    for (uint32_t i = 0; i < N; i++) {
        float val = input[i];
        
        // Skip values outside range
        if (val < min_val || val >= max_val) {
            continue;
        }
        
        // Compute bin index
        uint32_t bin = static_cast<uint32_t>((val - min_val) / bin_width);
        
        // Handle edge case where val == max_val
        if (bin >= num_bins) {
            bin = num_bins - 1;
        }
        
        output[bin]++;
    }
}


