// Loom kernel implementation: fft_butterfly
#include "fft_butterfly.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: FFT butterfly stages (Cooley-Tukey)
// Tests complete compilation chain with trigonometric operations (cos, sin, log2)
// Test: 16-point FFT with random complex inputs

const float PI_BUTTERFLY = 3.14159265358979323846f;

// CPU implementation of FFT butterfly stages
// Input must be bit-reversed before calling this function
// Performs Cooley-Tukey radix-2 DIT butterfly operations
void fft_butterfly_cpu(const float* __restrict__ input_real,
                       const float* __restrict__ input_imag,
                       float* __restrict__ output_real,
                       float* __restrict__ output_imag,
                       const uint32_t N) {
    // // Print output_real
    // for (uint32_t i = 0; i < N; i++) {
    //     printf("%e", input_real[i]);
    //     if (i < N - 1) printf(",");
    // }
    // printf(" \n");

    // // Print output_imag
    // for (uint32_t i = 0; i < N; i++) {
    //     printf("%e", input_imag[i]);
    //     if (i < N - 1) printf(",");
    // }
    // printf(" \n");

    // Copy input to output for working
    for (uint32_t i = 0; i < N; i++) {
        output_real[i] = input_real[i];
        output_imag[i] = input_imag[i];
    }

    // FFT butterfly computation
    for (uint32_t s = 1; s <= static_cast<uint32_t>(log2f(N)); s++) {
        uint32_t m = 1 << s;  // 2^s
        float wm_r = cosf(-2.0f * PI_BUTTERFLY / m);
        float wm_i = sinf(-2.0f * PI_BUTTERFLY / m);

        for (uint32_t k = 0; k < N; k += m) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (uint32_t j = 0; j < m / 2; j++) {
                // t = w * output[k + j + m/2]
                float t_r = w_r * output_real[k + j + m / 2] - w_i * output_imag[k + j + m / 2];
                float t_i = w_r * output_imag[k + j + m / 2] + w_i * output_real[k + j + m / 2];

                // u = output[k + j]
                float u_r = output_real[k + j];
                float u_i = output_imag[k + j];

                // output[k + j] = u + t
                output_real[k + j] = u_r + t_r;
                output_imag[k + j] = u_i + t_i;

                // output[k + j + m/2] = u - t
                output_real[k + j + m / 2] = u_r - t_r;
                output_imag[k + j + m / 2] = u_i - t_i;

                // w = w * wm
                float new_w_r = w_r * wm_r - w_i * wm_i;
                float new_w_i = w_r * wm_i + w_i * wm_r;
                w_r = new_w_r;
                w_i = new_w_i;
            }
        }
    }

    // // Print output_real
    // for (uint32_t i = 0; i < N; i++) {
    //     printf("%e", output_real[i]);
    //     if (i < N - 1) printf(",");
    // }
    // printf(" \n");

    // // Print output_imag
    // for (uint32_t i = 0; i < N; i++) {
    //     printf("%e", output_imag[i]);
    //     if (i < N - 1) printf(",");
    // }
    // printf(" \n");
}

// Accelerator implementation of FFT butterfly stages
LOOM_ACCEL()
void fft_butterfly_dsa(const float* __restrict__ input_real,
                       const float* __restrict__ input_imag,
                       float* __restrict__ output_real,
                       float* __restrict__ output_imag,
                       const uint32_t N) {
    // Copy input to output for working
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        output_real[i] = input_real[i];
        output_imag[i] = input_imag[i];
    }

    // FFT butterfly computation
    for (uint32_t s = 1; s <= static_cast<uint32_t>(log2f(N)); s++) {
        uint32_t m = 1 << s;  // 2^s
        float wm_r = cosf(-2.0f * PI_BUTTERFLY / m);
        float wm_i = sinf(-2.0f * PI_BUTTERFLY / m);

        for (uint32_t k = 0; k < N; k += m) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (uint32_t j = 0; j < m / 2; j++) {
                // t = w * output[k + j + m/2]
                float t_r = w_r * output_real[k + j + m / 2] - w_i * output_imag[k + j + m / 2];
                float t_i = w_r * output_imag[k + j + m / 2] + w_i * output_real[k + j + m / 2];

                // u = output[k + j]
                float u_r = output_real[k + j];
                float u_i = output_imag[k + j];

                // output[k + j] = u + t
                output_real[k + j] = u_r + t_r;
                output_imag[k + j] = u_i + t_i;

                // output[k + j + m/2] = u - t
                output_real[k + j + m / 2] = u_r - t_r;
                output_imag[k + j + m / 2] = u_i - t_i;

                // w = w * wm
                float new_w_r = w_r * wm_r - w_i * wm_i;
                float new_w_i = w_r * wm_i + w_i * wm_r;
                w_r = new_w_r;
                w_i = new_w_i;
            }
        }
    }
}

