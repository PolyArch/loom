/*
 * FFN First Layer -- matmul for transformer feed-forward network.
 * Computes C[N][D_ff] = A[N][D] * B[D][D_ff]
 * N=seq_len, D=512, D_ff=2048.
 * Tiled: tile_N=32, tile_Dff=64, tile_D=32.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define SEQ_LEN   128
#define D_MODEL   512
#define D_FF      2048
#define TILE_N    32
#define TILE_DFF  64
#define TILE_D    32

void ffn1(const float *input, const float *weight, const float *bias,
          float *output, int N, int D, int D_ff) {
    /* Initialize output with bias */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D_ff; j++) {
            output[i * D_ff + j] = bias[j];
        }
    }

    TILE_FOR(tn, 0, N, TILE_N) {
        int tn_end = TILE_END(tn, N, TILE_N);
        TILE_FOR(td, 0, D_ff, TILE_DFF) {
            int td_end = TILE_END(td, D_ff, TILE_DFF);
            TILE_FOR(tk, 0, D, TILE_D) {
                int tk_end = TILE_END(tk, D, TILE_D);
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

void ffn1_ref(const float *input, const float *weight, const float *bias,
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

int main(void) {
    int N = SEQ_LEN, D = D_MODEL, Dff = D_FF;

    float *input   = (float *)malloc((size_t)N * D * sizeof(float));
    float *weight  = (float *)malloc((size_t)D * Dff * sizeof(float));
    float *bias    = (float *)malloc((size_t)Dff * sizeof(float));
    float *out_t   = (float *)malloc((size_t)N * Dff * sizeof(float));
    float *out_r   = (float *)malloc((size_t)N * Dff * sizeof(float));

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

    ffn1(input, weight, bias, out_t, N, D, Dff);
    ffn1_ref(input, weight, bias, out_r, N, D, Dff);

    float max_err = 0.0f;
    for (int i = 0; i < N * Dff; i++) {
        float err = fabsf(out_t[i] - out_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("ffn1: max_error = %e\n", max_err);
    int pass = (max_err < 1e-2f);
    printf("ffn1: %s\n", pass ? "PASS" : "FAIL");

    free(input); free(weight); free(bias); free(out_t); free(out_r);
    return pass ? 0 : 1;
}
