/*
 * KV Projection -- shared K/V projection for decode step (S=1).
 * Computes K_new[B][H_kv][1][D_h] = input[B][1][D] * W_kv[D][H_kv*D_h]
 * Same structure used for both K and V projections.
 * B=batch, H_kv=KV heads (may differ from Q heads in GQA/MQA).
 *
 * Variants: mha, gqa, mqa, int8.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define D_MODEL   512
#define HEAD_DIM  64
#define BATCH     4
#define TILE_D    32

/* --- Core: matrix-vector multiply for KV projection --- */

static void kv_matvec_tiled(const float *input, const float *weight,
                            float *output, int B, int D, int H_kv, int D_h,
                            int tile_d) {
    int N_out = H_kv * D_h;
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

/* MHA: H_kv = H (standard multi-head attention, 8 KV heads) */
void kv_proj(const float *input, const float *weight,
             float *output, int B, int D, int H_kv, int D_h) {
    kv_matvec_tiled(input, weight, output, B, D, H_kv, D_h, TILE_D);
}

/* --- Variants --- */

void kv_proj_mha(const float *input, const float *weight,
                 float *output, int B, int D, int H_kv, int D_h) {
    kv_matvec_tiled(input, weight, output, B, D, H_kv, D_h, TILE_D);
}

/* GQA: grouped query attention (H_kv < H, e.g. H_kv=2) */
void kv_proj_gqa(const float *input, const float *weight,
                 float *output, int B, int D, int H_kv, int D_h) {
    kv_matvec_tiled(input, weight, output, B, D, H_kv, D_h, TILE_D);
}

/* MQA: multi-query attention (H_kv=1) */
void kv_proj_mqa(const float *input, const float *weight,
                 float *output, int B, int D, int H_kv, int D_h) {
    kv_matvec_tiled(input, weight, output, B, D, H_kv, D_h, TILE_D);
}

/* INT8 variant: simulate quantized weights */
void kv_proj_int8(const float *input, const float *weight,
                  float *output, int B, int D, int H_kv, int D_h) {
    int N_out = H_kv * D_h;
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
                    float w_quant = rintf(weight[k * N_out + j] * 127.0f);
                    if (w_quant > 127.0f) w_quant = 127.0f;
                    if (w_quant < -127.0f) w_quant = -127.0f;
                    y[j] += a * (w_quant * scale);
                }
            }
        }
    }
}

/* --- Reference implementation --- */

void kv_proj_ref(const float *input, const float *weight,
                 float *output, int B, int D, int H_kv, int D_h) {
    int N_out = H_kv * D_h;
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
    int B = BATCH, D = D_MODEL, H_kv = 8, D_h = HEAD_DIM;
    int N_out = H_kv * D_h;

    float *input  = (float *)malloc((size_t)B * D * sizeof(float));
    float *weight = (float *)malloc((size_t)D * N_out * sizeof(float));
    float *out_t  = (float *)malloc((size_t)B * N_out * sizeof(float));
    float *out_r  = (float *)malloc((size_t)B * N_out * sizeof(float));

    if (!input || !weight || !out_t || !out_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < B * D; i++)
        input[i] = ((float)(i % 89) - 44.0f) / 44.0f;
    for (int i = 0; i < D * N_out; i++)
        weight[i] = ((float)(i % 71) - 35.0f) / 350.0f;

    kv_proj(input, weight, out_t, B, D, H_kv, D_h);
    kv_proj_ref(input, weight, out_r, B, D, H_kv, D_h);

    float max_err = 0.0f;
    for (int i = 0; i < B * N_out; i++) {
        float err = fabsf(out_t[i] - out_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("kv_proj: max_error = %e\n", max_err);
    int pass = (max_err < 1e-3f);
    printf("kv_proj: %s\n", pass ? "PASS" : "FAIL");

    free(input); free(weight); free(out_t); free(out_r);
    return pass ? 0 : 1;
}
