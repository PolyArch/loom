// Loom kernel implementation: runge_kutta_step
#include "runge_kutta_step.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Runge-Kutta 4th order step
// Tests complete compilation chain with RK4 formula
// Test: y=[1,2], h=0.1, k1-k4 → [1.015, 2.025]






// CPU implementation of single RK4 (Runge-Kutta 4th order) step
// Computes one RK4 step for ODE: dy/dt = f(t, y)
// input_y: current state (N elements)
// input_k1: k1 = f(t, y) (N elements)
// input_k2: k2 = f(t + h/2, y + h*k1/2) (N elements)
// input_k3: k3 = f(t + h/2, y + h*k2/2) (N elements)
// input_k4: k4 = f(t + h, y + h*k3) (N elements)
// output_y: updated state y_new = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4) (N elements)
// h: step size
void runge_kutta_step_cpu(const float* __restrict__ input_y,
                          const float* __restrict__ input_k1,
                          const float* __restrict__ input_k2,
                          const float* __restrict__ input_k3,
                          const float* __restrict__ input_k4,
                          float* __restrict__ output_y,
                          const float h,
                          const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        output_y[i] = input_y[i] + (h / 6.0f) * (input_k1[i] + 2.0f * input_k2[i] + 2.0f * input_k3[i] + input_k4[i]);
    }
}

// Accelerator implementation of single RK4 step
LOOM_ACCEL()
void runge_kutta_step_dsa(const float* __restrict__ input_y,
                          const float* __restrict__ input_k1,
                          const float* __restrict__ input_k2,
                          const float* __restrict__ input_k3,
                          const float* __restrict__ input_k4,
                          float* __restrict__ output_y,
                          const float h,
                          const uint32_t N) {
    LOOM_PARALLEL(4)
    for (uint32_t i = 0; i < N; i++) {
        output_y[i] = input_y[i] + (h / 6.0f) * (input_k1[i] + 2.0f * input_k2[i] + 2.0f * input_k3[i] + input_k4[i]);
    }
}


