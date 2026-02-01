#include <cstdio>

#include "distance_point.h"
#include <cmath>

int main() {
    const uint32_t N = 64;
    
    // Input points (N pairs of 3D points)
    float point_a[N * 3];
    float point_b[N * 3];
    
    for (uint32_t i = 0; i < N; i++) {
        point_a[i * 3 + 0] = 0.0f;
        point_a[i * 3 + 1] = 0.0f;
        point_a[i * 3 + 2] = 0.0f;
        
        point_b[i * 3 + 0] = 3.0f + i * 0.1f;
        point_b[i * 3 + 1] = 4.0f + i * 0.1f;
        point_b[i * 3 + 2] = 0.0f;
    }
    
    // Output arrays
    float expect_distance[N];
    float calculated_distance[N];
    
    // Compute expected result with CPU version
    distance_point_cpu(point_a, point_b, expect_distance, N);
    
    // Compute result with accelerator version
    distance_point_dsa(point_a, point_b, calculated_distance, N);
    
    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_distance[i] - calculated_distance[i]) > 1e-5f) {
            printf("distance_point: FAILED\n");
            return 1;
        }
    }
    
    printf("distance_point: PASSED\n");
    return 0;
}

