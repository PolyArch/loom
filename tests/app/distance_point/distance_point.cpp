// Loom kernel implementation: distance_point
#include "distance_point.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Point-to-point Euclidean distance
// Tests complete compilation chain with sqrt operation
// Test: a=[(0,0,0),(3,0,0)], b=[(1,0,0),(0,4,0)], N=2 → distances=[1.0,5.0]

// CPU implementation of point-to-point distance
// Computes Euclidean distance between N pairs of 3D points
// Each point is 3 consecutive floats: (x, y, z)
void distance_point_cpu(const float* __restrict__ input_point_a,
                        const float* __restrict__ input_point_b,
                        float* __restrict__ output_distance,
                        const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        float ax = input_point_a[i * 3 + 0];
        float ay = input_point_a[i * 3 + 1];
        float az = input_point_a[i * 3 + 2];

        float bx = input_point_b[i * 3 + 0];
        float by = input_point_b[i * 3 + 1];
        float bz = input_point_b[i * 3 + 2];

        float dx = ax - bx;
        float dy = ay - by;
        float dz = az - bz;

        output_distance[i] = sqrtf(dx*dx + dy*dy + dz*dz);
    }
}

// Accelerator implementation of point-to-point distance
LOOM_ACCEL()
void distance_point_dsa(const float* __restrict__ input_point_a,
                        const float* __restrict__ input_point_b,
                        float* __restrict__ output_distance,
                        const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        float ax = input_point_a[i * 3 + 0];
        float ay = input_point_a[i * 3 + 1];
        float az = input_point_a[i * 3 + 2];

        float bx = input_point_b[i * 3 + 0];
        float by = input_point_b[i * 3 + 1];
        float bz = input_point_b[i * 3 + 2];

        float dx = ax - bx;
        float dy = ay - by;
        float dz = az - bz;

        output_distance[i] = sqrtf(dx*dx + dy*dy + dz*dz);
    }
}

