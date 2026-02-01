#include <cstdio>

#include "histogram_strided.h"

int main() {
    const uint32_t N = 2048;
    const uint32_t num_bins = 64;
    const uint32_t stride = 10;
    
    // Input array
    uint32_t input[N];
    
    // Histogram arrays
    uint32_t expect_hist[num_bins];
    uint32_t calculated_hist[num_bins];
    
    // Initialize input with values distributed across bins
    for (uint32_t i = 0; i < N; i++) {
        input[i] = (i * 13) % (num_bins * stride);
    }
    
    // Test strided histogram
    histogram_strided_cpu(input, expect_hist, N, num_bins, stride);
    histogram_strided_dsa(input, calculated_hist, N, num_bins, stride);
    
    for (uint32_t i = 0; i < num_bins; i++) {
        if (expect_hist[i] != calculated_hist[i]) {
            printf("histogram_strided: FAILED\n");
            return 1;
        }
    }
    
    printf("histogram_strided: PASSED\n");
    return 0;
}

