// Loom kernel implementation: bisection_step
#include "bisection_step.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Bisection method step
// Tests complete compilation chain with conditional interval selection
// Test: a=[0,1,2], b=[4,5,6], fa=[1,2,4], fb=[-2,-3,8], fc=[-0.5,1.5,5], N=3
// Branch coverage: [0]=if (fa*fc<0), [1]=else-if (fc*fb<0), [2]=else (default)
// Element 0: fa*fc = 1*(-0.5) = -0.5 < 0 → [a,c] = [0,2]
// Element 1: fa*fc = 2*1.5 = 3 > 0, fc*fb = 1.5*(-3) = -4.5 < 0 → [c,b] = [3,5]
// Element 2: fa*fc = 4*5 = 20 > 0, fc*fb = 5*8 = 40 > 0 → default [c,b] = [4,6]
// Expected: output_a=[0,3,4], output_b=[2,5,6]

// CPU implementation of single bisection method step
// Performs one bisection step for root finding
// input_a: left endpoints of intervals (N elements)
// input_b: right endpoints of intervals (N elements)
// input_fa: function values at left endpoints f(a) (N elements)
// input_fb: function values at right endpoints f(b) (N elements)
// input_fc: function values at midpoints f(c) where c = (a+b)/2 (N elements)
// output_a: updated left endpoints (N elements)
// output_b: updated right endpoints (N elements)
// Rule: if f(a)*f(c) < 0, new interval is [a,c]; else if f(c)*f(b) < 0, new interval is [c,b]; else [c,b]
void bisection_step_cpu(const float* __restrict__ input_a,
                        const float* __restrict__ input_b,
                        const float* __restrict__ input_fa,
                        const float* __restrict__ input_fb,
                        const float* __restrict__ input_fc,
                        float* __restrict__ output_a,
                        float* __restrict__ output_b,
                        const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        float c = (input_a[i] + input_b[i]) * 0.5f;

        // Check sign changes to determine which half contains the root
        if (input_fa[i] * input_fc[i] < 0.0f) {
            // f(a) and f(c) have opposite signs - root is in [a, c]
            output_a[i] = input_a[i];
            output_b[i] = c;
        } else if (input_fc[i] * input_fb[i] < 0.0f) {
            // f(c) and f(b) have opposite signs - root is in [c, b]
            output_a[i] = c;
            output_b[i] = input_b[i];
        } else {
            // No clear sign change detected - default to [c, b]
            output_a[i] = c;
            output_b[i] = input_b[i];
        }
    }
}

// Accelerator implementation of single bisection method step
LOOM_ACCEL()
void bisection_step_dsa(const float* __restrict__ input_a,
                        const float* __restrict__ input_b,
                        const float* __restrict__ input_fa,
                        const float* __restrict__ input_fb,
                        const float* __restrict__ input_fc,
                        float* __restrict__ output_a,
                        float* __restrict__ output_b,
                        const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t i = 0; i < N; i++) {
        float c = (input_a[i] + input_b[i]) * 0.5f;

        // Check sign changes to determine which half contains the root
        if (input_fa[i] * input_fc[i] < 0.0f) {
            // f(a) and f(c) have opposite signs - root is in [a, c]
            output_a[i] = input_a[i];
            output_b[i] = c;
        } else if (input_fc[i] * input_fb[i] < 0.0f) {
            // f(c) and f(b) have opposite signs - root is in [c, b]
            output_a[i] = c;
            output_b[i] = input_b[i];
        } else {
            // No clear sign change detected - default to [c, b]
            output_a[i] = c;
            output_b[i] = input_b[i];
        }
    }
}

