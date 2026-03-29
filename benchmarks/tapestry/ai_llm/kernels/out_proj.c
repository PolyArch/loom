/*
 * Output Projection -- matmul to combine multi-head attention output.
 * Computes output[B][D] = attn_out[B][H*D_h] * W_o[H*D_h][D]
 * For decode (S=1): this is a matrix-vector product per batch.
 * B=batch, H=heads, D_h=64, D=512.
 *
 * Variants: full, int8_dequant, low_rank.
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
#define TILE_K    32

/* --- Core: matrix-vector multiply for output projection --- */

static void out_proj_tiled(const float *input, const float *weight,
                           float *output, int B, int K_in, int D,
                           int tile_k) {
    for (int b = 0; b < B; b++) {
        const float *x = input + (size_t)b * K_in;
        float *y = output + (size_t)b * D;
        memset(y, 0, (size_t)D * sizeof(float));

        TILE_FOR(tk, 0, K_in, tile_k) {
            int tk_end = TILE_END(tk, K_in, tile_k);
            for (int k = tk; k < tk_end; k++) {
                float a = x[k];
                for (int j = 0; j < D; j++) {
                    y[j] += a * weight[k * D + j];
                }
            }
        }
    }
}

void out_proj(const float *input, const float *weight,
              float *output, int B, int H, int D_h, int D) {
    int K_in = H * D_h;
    out_proj_tiled(input, weight, output, B, K_in, D, TILE_K);
}

/* --- Variants --- */

void out_proj_full(const float *input, const float *weight,
                   float *output, int B, int H, int D_h, int D) {
    out_proj(input, weight, output, B, H, D_h, D);
}

/* INT8 dequant: simulate int8 weights with float dequantization */
void out_proj_int8_dequant(const float *input, const float *weight,
                           float *output, int B, int H, int D_h, int D) {
    int K_in = H * D_h;
    float scale = 1.0f / 127.0f;

    for (int b = 0; b < B; b++) {
        const float *x = input + (size_t)b * K_in;
        float *y = output + (size_t)b * D;
        memset(y, 0, (size_t)D * sizeof(float));

        TILE_FOR(tk, 0, K_in, TILE_K) {
            int tk_end = TILE_END(tk, K_in, TILE_K);
            for (int k = tk; k < tk_end; k++) {
                float a = x[k];
                for (int j = 0; j < D; j++) {
                    float w_quant = rintf(weight[k * D + j] * 127.0f);
                    if (w_quant > 127.0f) w_quant = 127.0f;
                    if (w_quant < -127.0f) w_quant = -127.0f;
                    y[j] += a * (w_quant * scale);
                }
            }
        }
    }
}

/* Low-rank variant: decompose W_o into W_a[K_in][R] * W_b[R][D] */
void out_proj_low_rank(const float *input,
                       const float *W_a, const float *W_b,
                       float *output, int B, int K_in, int R, int D) {
    float *intermediate = (float *)malloc((size_t)B * R * sizeof(float));
    if (!intermediate) return;

    /* First: intermediate[B][R] = input[B][K_in] * W_a[K_in][R] */
    for (int b = 0; b < B; b++) {
        const float *x = input + (size_t)b * K_in;
        float *mid = intermediate + (size_t)b * R;
        memset(mid, 0, (size_t)R * sizeof(float));
        TILE_FOR(tk, 0, K_in, TILE_K) {
            int tk_end = TILE_END(tk, K_in, TILE_K);
            for (int k = tk; k < tk_end; k++) {
                float a = x[k];
                for (int r = 0; r < R; r++) {
                    mid[r] += a * W_a[k * R + r];
                }
            }
        }
    }

    /* Second: output[B][D] = intermediate[B][R] * W_b[R][D] */
    for (int b = 0; b < B; b++) {
        const float *mid = intermediate + (size_t)b * R;
        float *y = output + (size_t)b * D;
        memset(y, 0, (size_t)D * sizeof(float));
        for (int r = 0; r < R; r++) {
            float a = mid[r];
            for (int j = 0; j < D; j++) {
                y[j] += a * W_b[r * D + j];
            }
        }
    }

    free(intermediate);
}

/* --- Reference implementation --- */

void out_proj_ref(const float *input, const float *weight,
                  float *output, int B, int H, int D_h, int D) {
    int K_in = H * D_h;
    for (int b = 0; b < B; b++) {
        const float *x = input + (size_t)b * K_in;
        float *y = output + (size_t)b * D;
        for (int j = 0; j < D; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K_in; k++) {
                sum += x[k] * weight[k * D + j];
            }
            y[j] = sum;
        }
    }
}

/* --- Self-test --- */

int main(void) {
    int B = BATCH, H = NUM_HEADS, D_h = HEAD_DIM, D = D_MODEL;
    int K_in = H * D_h;

    float *input  = (float *)malloc((size_t)B * K_in * sizeof(float));
    float *weight = (float *)malloc((size_t)K_in * D * sizeof(float));
    float *out_t  = (float *)malloc((size_t)B * D * sizeof(float));
    float *out_r  = (float *)malloc((size_t)B * D * sizeof(float));

    if (!input || !weight || !out_t || !out_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < B * K_in; i++)
        input[i] = ((float)(i % 83) - 41.0f) / 41.0f;
    for (int i = 0; i < K_in * D; i++)
        weight[i] = ((float)(i % 59) - 29.0f) / 290.0f;

    out_proj(input, weight, out_t, B, H, D_h, D);
    out_proj_ref(input, weight, out_r, B, H, D_h, D);

    float max_err = 0.0f;
    for (int i = 0; i < B * D; i++) {
        float err = fabsf(out_t[i] - out_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("out_proj: max_error = %e\n", max_err);
    int pass = (max_err < 1e-3f);
    printf("out_proj: %s\n", pass ? "PASS" : "FAIL");

    free(input); free(weight); free(out_t); free(out_r);
    return pass ? 0 : 1;
}
