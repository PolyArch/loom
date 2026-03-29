/*
 * Query Projection for decode -- matrix-vector multiply (S=1).
 * Computes Q[B][H][1][D_h] = input[B][1][D] * W_q[D][H*D_h]
 * B=batch, H=heads, D=512, D_h=64. S=1 for decode step.
 * This is memory-bound (matrix-vector product, not matrix-matrix).
 *
 * Variants: full, int8_dequant, grouped.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define D_MODEL   512
#define NUM_HEADS 8
#define HEAD_DIM  64
#define BATCH     4
#define TILE_D    32

/* --- Core: matrix-vector multiply for Q projection --- */

static void matvec_tiled(const float *input, const float *weight,
                         float *output, int B, int D, int N_out,
                         int tile_d) {
    for (int b = 0; b < B; b++) {
        const float *x = input + (size_t)b * D;
        float *y = output + (size_t)b * N_out;
        memset(y, 0, (size_t)N_out * sizeof(float));

        TILE_FOR(td, 0, D, tile_d) {
            int td_end = TILE_END(td, D, tile_d);
            for (int k = td; k < td_end; k++) {
                float a = x[k];
                for (int j = 0; j < N_out; j++) {
                    y[j] += a * weight[k * N_out + j];
                }
            }
        }
    }
}

void q_proj(const float *input, const float *weight,
            float *output, int B, int D, int H, int D_h) {
    int N_out = H * D_h;
    matvec_tiled(input, weight, output, B, D, N_out, TILE_D);
}

/* --- Variants --- */

void q_proj_full(const float *input, const float *weight,
                 float *output, int B, int D, int H, int D_h) {
    q_proj(input, weight, output, B, D, H, D_h);
}

/* INT8 dequant: simulate int8 weights with float dequantization */
void q_proj_int8_dequant(const float *input, const float *weight,
                         float *output, int B, int D, int H, int D_h) {
    int N_out = H * D_h;
    float scale = 1.0f / 127.0f;
    for (int b = 0; b < B; b++) {
        const float *x = input + (size_t)b * D;
        float *y = output + (size_t)b * N_out;
        memset(y, 0, (size_t)N_out * sizeof(float));

        TILE_FOR(td, 0, D, TILE_D) {
            int td_end = TILE_END(td, D, TILE_D);
            for (int k = td; k < td_end; k++) {
                float a = x[k];
                for (int j = 0; j < N_out; j++) {
                    /* Simulate: quantize weight then dequant for accumulation */
                    float w_quant = rintf(weight[k * N_out + j] * 127.0f);
                    if (w_quant > 127.0f) w_quant = 127.0f;
                    if (w_quant < -127.0f) w_quant = -127.0f;
                    y[j] += a * (w_quant * scale);
                }
            }
        }
    }
}

/* Grouped variant: GQA-style, fewer query heads share KV heads */
void q_proj_grouped(const float *input, const float *weight,
                    float *output, int B, int D, int H, int D_h) {
    q_proj(input, weight, output, B, D, H, D_h);
}

/* --- Reference implementation --- */

void q_proj_ref(const float *input, const float *weight,
                float *output, int B, int D, int H, int D_h) {
    int N_out = H * D_h;
    for (int b = 0; b < B; b++) {
        const float *x = input + (size_t)b * D;
        float *y = output + (size_t)b * N_out;
        for (int j = 0; j < N_out; j++) {
            float sum = 0.0f;
            for (int k = 0; k < D; k++) {
                sum += x[k] * weight[k * N_out + j];
            }
            y[j] = sum;
        }
    }
}

/* --- Self-test --- */

int main(void) {
    int B = BATCH, D = D_MODEL, H = NUM_HEADS, D_h = HEAD_DIM;
    int N_out = H * D_h;

    float *input  = (float *)malloc((size_t)B * D * sizeof(float));
    float *weight = (float *)malloc((size_t)D * N_out * sizeof(float));
    float *out_t  = (float *)malloc((size_t)B * N_out * sizeof(float));
    float *out_r  = (float *)malloc((size_t)B * N_out * sizeof(float));

    if (!input || !weight || !out_t || !out_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < B * D; i++)
        input[i] = ((float)(i % 97) - 48.0f) / 48.0f;
    for (int i = 0; i < D * N_out; i++)
        weight[i] = ((float)(i % 67) - 33.0f) / 330.0f;

    q_proj(input, weight, out_t, B, D, H, D_h);
    q_proj_ref(input, weight, out_r, B, D, H, D_h);

    float max_err = 0.0f;
    for (int i = 0; i < B * N_out; i++) {
        float err = fabsf(out_t[i] - out_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("q_proj: max_error = %e\n", max_err);
    int pass = (max_err < 1e-3f);
    printf("q_proj: %s\n", pass ? "PASS" : "FAIL");

    free(input); free(weight); free(out_t); free(out_r);
    return pass ? 0 : 1;
}
