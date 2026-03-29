/*
 * FFN Up-Projection -- expansion matmul for transformer FFN.
 * Computes output[N][D_ff] = input[N][D] * weight[D][D_ff] + bias[D_ff]
 * N=seq_len, D=512, D_ff=2048.
 *
 * Variants: tile32, tile64, tile128, gated (for GLU-style gating).
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
                            int N, int D, int D_ff,
                            int tile_n, int tile_d, int tile_k) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D_ff; j++) {
            output[i * D_ff + j] = bias[j];
        }
    }

    TILE_FOR(tn, 0, N, tile_n) {
        int tn_end = TILE_END(tn, N, tile_n);
        TILE_FOR(td, 0, D_ff, tile_d) {
            int td_end = TILE_END(td, D_ff, tile_d);
            TILE_FOR(tk, 0, D, tile_k) {
                int tk_end = TILE_END(tk, D, tile_k);
                for (int i = tn; i < tn_end; i++) {
                    for (int j = td; j < td_end; j++) {
                        float sum = 0.0f;
                        for (int k = tk; k < tk_end; k++) {
                            sum += input[i * D + k] * weight[k * D_ff + j];
                        }
                        output[i * D_ff + j] += sum;
                    }
                }
            }
        }
    }
}

void ffn_up(const float *input, const float *weight, const float *bias,
            float *output, int N, int D, int D_ff) {
    gemm_bias_tiled(input, weight, bias, output, N, D, D_ff, 32, 64, 32);
}

/* --- Tile-size variants --- */

void ffn_up_tile32(const float *input, const float *weight, const float *bias,
                   float *output, int N, int D, int D_ff) {
    gemm_bias_tiled(input, weight, bias, output, N, D, D_ff, 32, 32, 32);
}

void ffn_up_tile64(const float *input, const float *weight, const float *bias,
                   float *output, int N, int D, int D_ff) {
    gemm_bias_tiled(input, weight, bias, output, N, D, D_ff, 64, 64, 64);
}

void ffn_up_tile128(const float *input, const float *weight, const float *bias,
                    float *output, int N, int D, int D_ff) {
    gemm_bias_tiled(input, weight, bias, output, N, D, D_ff, 128, 128, 64);
}

/* Gated variant: for SwiGLU / GLU-style FFN (up * gate) */
void ffn_up_gated(const float *input,
                  const float *weight_up, const float *bias_up,
                  const float *weight_gate, const float *bias_gate,
                  float *output, int N, int D, int D_ff) {
    float *gate = (float *)malloc((size_t)N * D_ff * sizeof(float));
    if (!gate) return;

    gemm_bias_tiled(input, weight_up, bias_up, output, N, D, D_ff,
                    32, 64, 32);
    gemm_bias_tiled(input, weight_gate, bias_gate, gate, N, D, D_ff,
                    32, 64, 32);

    /* Element-wise multiply: output = up * sigmoid(gate) */
    int total = N * D_ff;
    for (int i = 0; i < total; i++) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        output[i] *= sig;
    }
    free(gate);
}

/* --- Reference implementation --- */

void ffn_up_ref(const float *input, const float *weight, const float *bias,
                float *output, int N, int D, int D_ff) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D_ff; j++) {
            float sum = bias[j];
            for (int k = 0; k < D; k++) {
                sum += input[i * D + k] * weight[k * D_ff + j];
            }
            output[i * D_ff + j] = sum;
        }
    }
}

/* --- Self-test --- */

int main(void) {
    int N = SEQ_LEN, D = D_MODEL, Dff = D_FF;

    float *input  = (float *)malloc((size_t)N * D * sizeof(float));
    float *weight = (float *)malloc((size_t)D * Dff * sizeof(float));
    float *bias   = (float *)malloc((size_t)Dff * sizeof(float));
    float *out_t  = (float *)malloc((size_t)N * Dff * sizeof(float));
    float *out_r  = (float *)malloc((size_t)N * Dff * sizeof(float));

    if (!input || !weight || !bias || !out_t || !out_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N * D; i++)
        input[i] = ((float)(i % 79) - 39.0f) / 39.0f;
    for (int i = 0; i < D * Dff; i++)
        weight[i] = ((float)(i % 61) - 30.0f) / 300.0f;
    for (int i = 0; i < Dff; i++)
        bias[i] = ((float)(i % 37) - 18.0f) / 18.0f;

    ffn_up(input, weight, bias, out_t, N, D, Dff);
    ffn_up_ref(input, weight, bias, out_r, N, D, Dff);

    float max_err = 0.0f;
    for (int i = 0; i < N * Dff; i++) {
        float err = fabsf(out_t[i] - out_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("ffn_up: max_error = %e\n", max_err);
    int pass = (max_err < 1e-2f);
    printf("ffn_up: %s\n", pass ? "PASS" : "FAIL");

    free(input); free(weight); free(bias); free(out_t); free(out_r);
    return pass ? 0 : 1;
}
