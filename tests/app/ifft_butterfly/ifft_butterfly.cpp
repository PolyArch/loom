// Loom kernel implementation: ifft_butterfly
#include "ifft_butterfly.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: IFFT butterfly stages
// Tests complete compilation chain with inverse FFT (trig functions, conjugation, scaling)








const float PI_IFFT_BUTTERFLY = 3.14159265358979323846f;

// CPU implementation of IFFT butterfly stages
// Input must be bit-reversed and conjugated before calling this function
// Performs butterfly operations followed by conjugation and scaling
void ifft_butterfly_cpu(const float* __restrict__ input_real,
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
    
    // FFT butterfly computation (same as forward FFT)
    for (uint32_t s = 1; s <= static_cast<uint32_t>(log2f(N)); s++) {
        uint32_t m = 1 << s;  // 2^s
        float wm_r = cosf(-2.0f * PI_IFFT_BUTTERFLY / m);
        float wm_i = sinf(-2.0f * PI_IFFT_BUTTERFLY / m);
        
        for (uint32_t k = 0; k < N; k += m) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (uint32_t j = 0; j < m / 2; j++) {
                float t_r = w_r * output_real[k + j + m / 2] - w_i * output_imag[k + j + m / 2];
                float t_i = w_r * output_imag[k + j + m / 2] + w_i * output_real[k + j + m / 2];
                
                float u_r = output_real[k + j];
                float u_i = output_imag[k + j];
                
                output_real[k + j] = u_r + t_r;
                output_imag[k + j] = u_i + t_i;
                
                output_real[k + j + m / 2] = u_r - t_r;
                output_imag[k + j + m / 2] = u_i - t_i;
                
                float new_w_r = w_r * wm_r - w_i * wm_i;
                float new_w_i = w_r * wm_i + w_i * wm_r;
                w_r = new_w_r;
                w_i = new_w_i;
            }
        }
    }
    
    // Conjugate the output and scale by 1/N
    float scale = 1.0f / N;
    for (uint32_t i = 0; i < N; i++) {
        output_real[i] = output_real[i] * scale;
        output_imag[i] = -output_imag[i] * scale;
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

// Accelerator implementation of IFFT butterfly stages
LOOM_ACCEL()
void ifft_butterfly_dsa(const float* __restrict__ input_real,
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
    
    // FFT butterfly computation (same as forward FFT)
    for (uint32_t s = 1; s <= static_cast<uint32_t>(log2f(N)); s++) {
        uint32_t m = 1 << s;
        float wm_r = cosf(-2.0f * PI_IFFT_BUTTERFLY / m);
        float wm_i = sinf(-2.0f * PI_IFFT_BUTTERFLY / m);
        
        for (uint32_t k = 0; k < N; k += m) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (uint32_t j = 0; j < m / 2; j++) {
                float t_r = w_r * output_real[k + j + m / 2] - w_i * output_imag[k + j + m / 2];
                float t_i = w_r * output_imag[k + j + m / 2] + w_i * output_real[k + j + m / 2];
                
                float u_r = output_real[k + j];
                float u_i = output_imag[k + j];
                
                output_real[k + j] = u_r + t_r;
                output_imag[k + j] = u_i + t_i;
                
                output_real[k + j + m / 2] = u_r - t_r;
                output_imag[k + j + m / 2] = u_i - t_i;
                
                float new_w_r = w_r * wm_r - w_i * wm_i;
                float new_w_i = w_r * wm_i + w_i * wm_r;
                w_r = new_w_r;
                w_i = new_w_i;
            }
        }
    }
    
    // Conjugate the output and scale by 1/N
    float scale = 1.0f / N;
    for (uint32_t i = 0; i < N; i++) {
        output_real[i] = output_real[i] * scale;
        output_imag[i] = -output_imag[i] * scale;
    }
}



