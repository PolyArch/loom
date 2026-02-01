// Loom kernel implementation: line_intersect
#include "line_intersect.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: 2D line segment intersection test
// Tests complete compilation chain with geometric computation (cross products, fabsf)
// Test: 6 line pairs → intersections [1,0,1,0,1,0]







// CPU implementation of 2D line segment intersection test
// Tests if N pairs of 2D line segments intersect
// Each line segment is defined by 2 points (4 floats): (x1, y1, x2, y2)
// Output: 1 if lines intersect, 0 if they don't
// Uses cross product method to determine intersection
void line_intersect_cpu(const float* __restrict__ input_line_a,
                        const float* __restrict__ input_line_b,
                        uint32_t* __restrict__ output_intersect,
                        const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        // Line A: from (ax1, ay1) to (ax2, ay2)
        float ax1 = input_line_a[i * 4 + 0];
        float ay1 = input_line_a[i * 4 + 1];
        float ax2 = input_line_a[i * 4 + 2];
        float ay2 = input_line_a[i * 4 + 3];
        
        // Line B: from (bx1, by1) to (bx2, by2)
        float bx1 = input_line_b[i * 4 + 0];
        float by1 = input_line_b[i * 4 + 1];
        float bx2 = input_line_b[i * 4 + 2];
        float by2 = input_line_b[i * 4 + 3];
        
        // Direction vectors
        float dax = ax2 - ax1;
        float day = ay2 - ay1;
        float dbx = bx2 - bx1;
        float dby = by2 - by1;
        
        // Denominator for parameter calculation
        float denom = dax * dby - day * dbx;
        
        // Check if lines are parallel
        if (fabsf(denom) < 1e-8f) {
            output_intersect[i] = 0;
            continue;
        }
        
        // Vector from A start to B start
        float dx = bx1 - ax1;
        float dy = by1 - ay1;
        
        // Calculate parameters
        float t = (dx * dby - dy * dbx) / denom;
        float u = (dx * day - dy * dax) / denom;
        
        // Check if intersection occurs within both line segments
        if (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f) {
            output_intersect[i] = 1;
        } else {
            output_intersect[i] = 0;
        }
    }
}

// Accelerator implementation of 2D line segment intersection test
LOOM_ACCEL()
void line_intersect_dsa(const float* __restrict__ input_line_a,
                        const float* __restrict__ input_line_b,
                        uint32_t* __restrict__ output_intersect,
                        const uint32_t N) {
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < N; i++) {
        // Line A: from (ax1, ay1) to (ax2, ay2)
        float ax1 = input_line_a[i * 4 + 0];
        float ay1 = input_line_a[i * 4 + 1];
        float ax2 = input_line_a[i * 4 + 2];
        float ay2 = input_line_a[i * 4 + 3];
        
        // Line B: from (bx1, by1) to (bx2, by2)
        float bx1 = input_line_b[i * 4 + 0];
        float by1 = input_line_b[i * 4 + 1];
        float bx2 = input_line_b[i * 4 + 2];
        float by2 = input_line_b[i * 4 + 3];
        
        // Direction vectors
        float dax = ax2 - ax1;
        float day = ay2 - ay1;
        float dbx = bx2 - bx1;
        float dby = by2 - by1;
        
        // Denominator for parameter calculation
        float denom = dax * dby - day * dbx;
        
        // Check if lines are parallel
        if (fabsf(denom) < 1e-8f) {
            output_intersect[i] = 0;
            continue;
        }
        
        // Vector from A start to B start
        float dx = bx1 - ax1;
        float dy = by1 - ay1;
        
        // Calculate parameters
        float t = (dx * dby - dy * dbx) / denom;
        float u = (dx * day - dy * dax) / denom;
        
        // Check if intersection occurs within both line segments
        if (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f) {
            output_intersect[i] = 1;
        } else {
            output_intersect[i] = 0;
        }
    }
}



