/*
 * Activation Functions -- element-wise nonlinearity for transformer FFN.
 * Supports GeLU (polynomial approximation), SiLU (sigmoid * x), ReLU.
 * N=seq_len, D=2048 (D_ff dimension).
 *
 * Variants: gelu_poly, gelu_lut, silu, relu.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tile_utils.h"

#define SEQ_LEN    128
#define D_FF       2048
#define TILE_SIZE  256

#define SQRT_2_OVER_PI 0.7978845608f
#define GELU_COEFF     0.044715f

/* Polynomial tanh approximation (Pade approximant) */
static inline float approx_tanh(float x) {
    if (x > 5.0f) return 1.0f;
    if (x < -5.0f) return -1.0f;
    float x2 = x * x;
    float num = x * (135135.0f + x2 * (17325.0f + x2 * 378.0f));
    float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return num / den;
}

/* --- Primary kernel: GeLU with polynomial tanh --- */

void activation(float *data, int total) {
    TILE_FOR(t, 0, total, TILE_SIZE) {
        int t_end = TILE_END(t, total, TILE_SIZE);
        for (int i = t; i < t_end; i++) {
            float x = data[i];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
            data[i] = 0.5f * x * (1.0f + approx_tanh(inner));
        }
    }
}

/* --- Variants --- */

/* GeLU polynomial approximation (same as primary) */
void activation_gelu_poly(float *data, int total) {
    TILE_FOR(t, 0, total, TILE_SIZE) {
        int t_end = TILE_END(t, total, TILE_SIZE);
        for (int i = t; i < t_end; i++) {
            float x = data[i];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
            data[i] = 0.5f * x * (1.0f + approx_tanh(inner));
        }
    }
}

/* GeLU with lookup-table style: use standard tanhf as proxy for LUT */
void activation_gelu_lut(float *data, int total) {
    TILE_FOR(t, 0, total, TILE_SIZE) {
        int t_end = TILE_END(t, total, TILE_SIZE);
        for (int i = t; i < t_end; i++) {
            float x = data[i];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
            data[i] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }
}

/* SiLU (Swish): x * sigmoid(x) */
void activation_silu(float *data, int total) {
    TILE_FOR(t, 0, total, TILE_SIZE) {
        int t_end = TILE_END(t, total, TILE_SIZE);
        for (int i = t; i < t_end; i++) {
            float x = data[i];
            float sig = 1.0f / (1.0f + expf(-x));
            data[i] = x * sig;
        }
    }
}

/* ReLU: max(0, x) */
void activation_relu(float *data, int total) {
    TILE_FOR(t, 0, total, TILE_SIZE) {
        int t_end = TILE_END(t, total, TILE_SIZE);
        for (int i = t; i < t_end; i++) {
            if (data[i] < 0.0f) data[i] = 0.0f;
        }
    }
}

/* --- Reference implementations --- */

void activation_gelu_ref(float *data, int total) {
    for (int i = 0; i < total; i++) {
        float x = data[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
        data[i] = 0.5f * x * (1.0f + approx_tanh(inner));
    }
}

void activation_silu_ref(float *data, int total) {
    for (int i = 0; i < total; i++) {
        float x = data[i];
        float sig = 1.0f / (1.0f + expf(-x));
        data[i] = x * sig;
    }
}

void activation_relu_ref(float *data, int total) {
    for (int i = 0; i < total; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

/* --- Self-test --- */

int main(void) {
    int total = SEQ_LEN * D_FF;
    float *d_gelu   = (float *)malloc((size_t)total * sizeof(float));
    float *d_gelu_r = (float *)malloc((size_t)total * sizeof(float));
    float *d_silu   = (float *)malloc((size_t)total * sizeof(float));
    float *d_silu_r = (float *)malloc((size_t)total * sizeof(float));
    float *d_relu   = (float *)malloc((size_t)total * sizeof(float));
    float *d_relu_r = (float *)malloc((size_t)total * sizeof(float));

    if (!d_gelu || !d_gelu_r || !d_silu || !d_silu_r || !d_relu || !d_relu_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < total; i++) {
        float val = ((float)(i % 199) - 99.0f) / 33.0f;
        d_gelu[i] = val; d_gelu_r[i] = val;
        d_silu[i] = val; d_silu_r[i] = val;
        d_relu[i] = val; d_relu_r[i] = val;
    }

    activation_gelu_poly(d_gelu, total);
    activation_gelu_ref(d_gelu_r, total);

    activation_silu(d_silu, total);
    activation_silu_ref(d_silu_r, total);

    activation_relu(d_relu, total);
    activation_relu_ref(d_relu_r, total);

    float max_gelu = 0.0f, max_silu = 0.0f, max_relu = 0.0f;
    for (int i = 0; i < total; i++) {
        float e = fabsf(d_gelu[i] - d_gelu_r[i]);
        if (e > max_gelu) max_gelu = e;
        e = fabsf(d_silu[i] - d_silu_r[i]);
        if (e > max_silu) max_silu = e;
        e = fabsf(d_relu[i] - d_relu_r[i]);
        if (e > max_relu) max_relu = e;
    }

    printf("activation_gelu_poly: max_error = %e\n", max_gelu);
    printf("activation_silu: max_error = %e\n", max_silu);
    printf("activation_relu: max_error = %e\n", max_relu);

    int pass = (max_gelu < 1e-5f) && (max_silu < 1e-5f) && (max_relu == 0.0f);
    printf("activation: %s\n", pass ? "PASS" : "FAIL");

    free(d_gelu); free(d_gelu_r);
    free(d_silu); free(d_silu_r);
    free(d_relu); free(d_relu_r);
    return pass ? 0 : 1;
}
