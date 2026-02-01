#include <cstdio>

#include "histogram.h"

int main() {
    const uint32_t N = 2048;
    const uint32_t num_bins = 64;
    
    // Input array
    uint32_t input[N];
    
    // Histogram arrays
    uint32_t expect_hist[num_bins];
    uint32_t calculated_hist[num_bins];
    
    // Initialize input with values distributed across bins
    for (uint32_t i = 0; i < N; i++) {
        input[i] = (i * 13) % num_bins;
    }
    
    // Test basic histogram
    histogram_cpu(input, expect_hist, N, num_bins);
    histogram_dsa(input, calculated_hist, N, num_bins);
    
    for (uint32_t i = 0; i < num_bins; i++) {
        if (expect_hist[i] != calculated_hist[i]) {
            printf("histogram: FAILED\n");
            return 1;
        }
    }
    
    printf("histogram: PASSED\n");
    return 0;
}

