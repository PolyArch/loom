/*
 * FFN Down-Projection -- contraction matmul for transformer FFN.
 * Computes output[N][D] = input[N][D_ff] * weight[D_ff][D] + bias[D]
 * N=seq_len, D_ff=2048, D=512.
 *
 * Variants: tile32, tile64, tile128.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define SEQ_LEN   128
#define D_MODEL   512
#define D_FF      2048

/* --- Core tiled GEMM with bias --- */

static void gemm_bias_tiled(const float *input, const float *weight,
                            const float *bias, float *output,
                            int N, int K, int D,
                            int tile_n, int tile_d, int tile_k) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            output[i * D + j] = bias[j];
        }
    }

    TILE_FOR(tn, 0, N, tile_n) {
        int tn_end = TILE_END(tn, N, tile_n);
        TILE_FOR(td, 0, D, tile_d) {
            int td_end = TILE_END(td, D, tile_d);
            TILE_FOR(tk, 0, K, tile_k) {
                int tk_end = TILE_END(tk, K, tile_k);
                for (int i = tn; i < tn_end; i++) {
                    for (int j = td; j < td_end; j++) {
                        float sum = 0.0f;
                        for (int k = tk; k < tk_end; k++) {
                            sum += input[i * K + k] * weight[k * D + j];
                        }
                        output[i * D + j] += sum;
                    }
                }
            }
        }
    }
}

void ffn_down(const float *input, const float *weight, const float *bias,
              float *output, int N, int D_ff, int D) {
    gemm_bias_tiled(input, weight, bias, output, N, D_ff, D, 32, 64, 32);
}

/* --- Tile-size variants --- */

void ffn_down_tile32(const float *input, const float *weight,
                     const float *bias, float *output,
                     int N, int D_ff, int D) {
    gemm_bias_tiled(input, weight, bias, output, N, D_ff, D, 32, 32, 32);
}

void ffn_down_tile64(const float *input, const float *weight,
                     const float *bias, float *output,
                     int N, int D_ff, int D) {
    gemm_bias_tiled(input, weight, bias, output, N, D_ff, D, 64, 64, 64);
}

void ffn_down_tile128(const float *input, const float *weight,
                      const float *bias, float *output,
                      int N, int D_ff, int D) {
    gemm_bias_tiled(input, weight, bias, output, N, D_ff, D, 128, 128, 64);
}

/* --- Reference implementation --- */

void ffn_down_ref(const float *input, const float *weight, const float *bias,
                  float *output, int N, int D_ff, int D) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            float sum = bias[j];
            for (int k = 0; k < D_ff; k++) {
                sum += input[i * D_ff + k] * weight[k * D + j];
            }
            output[i * D + j] = sum;
        }
    }
}

/* --- Self-test --- */

int main(void) {
    int N = SEQ_LEN, Dff = D_FF, D = D_MODEL;

    float *input  = (float *)malloc((size_t)N * Dff * sizeof(float));
    float *weight = (float *)malloc((size_t)Dff * D * sizeof(float));
    float *bias   = (float *)malloc((size_t)D * sizeof(float));
    float *out_t  = (float *)malloc((size_t)N * D * sizeof(float));
    float *out_r  = (float *)malloc((size_t)N * D * sizeof(float));

    if (!input || !weight || !bias || !out_t || !out_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N * Dff; i++)
        input[i] = ((float)(i % 79) - 39.0f) / 390.0f;
    for (int i = 0; i < Dff * D; i++)
        weight[i] = ((float)(i % 61) - 30.0f) / 300.0f;
    for (int i = 0; i < D; i++)
        bias[i] = ((float)(i % 37) - 18.0f) / 18.0f;

    ffn_down(input, weight, bias, out_t, N, Dff, D);
    ffn_down_ref(input, weight, bias, out_r, N, Dff, D);

    float max_err = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float err = fabsf(out_t[i] - out_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("ffn_down: max_error = %e\n", max_err);
    int pass = (max_err < 1e-2f);
    printf("ffn_down: %s\n", pass ? "PASS" : "FAIL");

    free(input); free(weight); free(bias); free(out_t); free(out_r);
    return pass ? 0 : 1;
}
